"""First-order state-estimation solver used as a baseline in the paper.

Only ``LBFGS_se`` is kept here: it is the **LBFGSNF** baseline of Table I --
L-BFGS applied to the latent-space normalizing-flow MAP objective (same
objective as LMNF), used to isolate the effect of the LM curvature
approximation / trust-region update. The other experimental first-order
optimizers from the research repo (SGD/Muon/AdamW/GDBT variants) are not part
of the paper and were removed for this release.
"""

import numpy as np
import torch
from torch.optim import LBFGS
from tqdm import tqdm

from SE_torch.optimizers.base_optimizer import SEOptimizer
from SE_torch.optimizers.se_loss import SELoss, SELossCartNFLat


class LBFGS_se(SEOptimizer):
    """L-BFGS state estimator over the NF latent code (LBFGSNF baseline).

    With ``use_prior=True`` the loss is the full latent-space NF-MAP objective
    (``SELossCartNFLat`` with the log-determinant term); ``obs_lvl`` rescales
    the likelihood/prior weighting to the current observability level.
    """

    def __init__(self, **kwargs):
        super(LBFGS_se, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = int(kwargs.get('max_iter', 200))
        self.verbose = kwargs.get('verbose', False)
        self.use_prior = kwargs.get('use_prior', False)
        self.prior_config_path = kwargs.get('prior_config_path')
        obs_lvl = float(kwargs.get('obs_lvl', 1.))
        if self.use_prior:
            self.loss_func = SELossCartNFLat(
                nf_model_config_name=self.prior_config_path,
                with_log_det=True,
                prior_scale=torch.tensor(np.log2(np.exp(1)) / 234.).to(torch.get_default_dtype()),
                ll_scale=torch.tensor(np.log2(np.exp(1)) / (obs_lvl * 726)).to(torch.get_default_dtype()),
            )
        else:
            self.loss_func = SELoss()

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        x = x0.clone().detach().to(torch.get_default_dtype())
        all_x = [x.clone().detach()]

        self.loss_func.update_params(z, v, slk_bus, h_ac, nb, norm_H)

        x_enc = self.loss_func.encode(x).detach()
        x_enc.requires_grad = True

        optimizer = LBFGS([x_enc], line_search_fn='strong_wolfe', max_iter=3)

        def closure():
            optimizer.zero_grad()
            loss_c = self.loss_func.compute_f(x_enc)
            loss_c.backward()
            return loss_c

        converged = False
        it, delta = 0, torch.inf
        with torch.no_grad():
            loss_prev = self.loss_func.compute_f(x_enc).item()
        x_enc_prev = x_enc.clone()
        pbar = tqdm(range(int(self.max_iter)), desc=f'Optimizing with LBFGS', disable=not self.verbose, leave=True, colour='green',
                    postfix={'loss': f"{loss_prev:.4f}"})
        for it in pbar:
            optimizer.step(closure)
            with torch.no_grad():
                loss = self.loss_func.compute_f(x_enc)

            delta_f = (loss_prev - loss.item()) / abs(loss_prev)
            delta_x = torch.norm(x_enc - x_enc_prev).item()
            if 0 <= delta_f <= self.tol:# and delta_x <= self.tol:
                converged = True
                pbar.set_postfix(ftol=f"{delta_f:.4e}", xtol=f"{delta_x:.4e}", loss=f"{loss.item():.4f}")
                break
            loss_prev = loss.item()
            x_enc_prev = x_enc.clone()
            pbar.set_postfix(ftol=f"{delta_f:.4e}", xtol=f"{delta_x:.4e}", loss=f"{loss.item():.4f}")
        x = self.loss_func.decode(x_enc.detach())
        T, V = x[:nb], x[nb:]

        return x, T, V, converged, it, loss.item(), all_x
