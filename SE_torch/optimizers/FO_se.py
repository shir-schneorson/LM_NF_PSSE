import json

import torch
from sympy.physics.units import momentum
from torch.optim import SGD, Muon, AdamW, LBFGS
from torch.distributions import MultivariateNormal, Normal
from tqdm import tqdm

from SE_torch.learn_prior.NF.NF import FlowModel
from SE_torch.learn_prior.VAE.VAE import AmortizedVAE
from SE_torch.optimizers.LM_opt import LMOpt
from SE_torch.optimizers.base_optimizer import SEOptimizer, WeightedSELoss, WeightedSEWithPriorLoss, WeightedSECartWithPriorLoss
from SE_torch.optimizers.se_loss import SELossNF, SELoss


class LBFGS_se(SEOptimizer):
    def __init__(self, **kwargs):
        super(LBFGS_se, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-7)
        self.max_iter = int(kwargs.get('max_iter', 500))
        self.verbose = kwargs.get('verbose', True)
        self.use_prior = kwargs.get('use_prior', False)
        self.prior_config_path = kwargs.get('prior_config_path')

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        x_polar = x0.clone()
        x_polar.requires_grad = True
        all_x = [x_polar.clone().detach()]
        non_slk_bus = (torch.arange(nb * 2) != slk_bus[0]).type(torch.get_default_dtype())
        slk_vec = torch.zeros_like(non_slk_bus)
        slk_vec[slk_bus[0]] = slk_bus[1]

        if self.use_prior:
            criterion = WeightedSEWithPriorLoss(z, v, slk_bus[0], NF_config_path=self.prior_config_path)
        else:
            criterion = WeightedSELoss(z, v)
        optimizer = LBFGS([x_polar], line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            x_curr = x_polar * non_slk_bus + slk_vec
            z_est = h_ac.estimate(x_curr)
            if self.use_prior:
                loss = criterion(z_est, x_curr)
            else:
                loss = criterion(z_est)
            loss.backward()
            return loss

        converged = False
        it, delta = 0, torch.inf
        with torch.no_grad():
            x_curr = x_polar * non_slk_bus + slk_vec
            z_est = h_ac.estimate(x_curr)
            if self.use_prior:
                loss_prev = criterion(z_est, x_curr).item()
            else:
                loss_prev = criterion(z_est).item()
        x_prev = x_polar.clone()
        pbar = tqdm(range(int(self.max_iter)), desc=f'Optimizing with LBFGS', leave=True, colour='green',
                    postfix={'loss': f"{loss_prev:.4f}"})
        for it in pbar:
            optimizer.step(closure)
            with torch.no_grad():
                x_curr = x_polar * non_slk_bus + slk_vec
                z_est = h_ac.estimate(x_curr)
                if self.use_prior:
                    loss = criterion(z_est, x_curr)
                else:
                    loss = criterion(z_est)
            delta_f = (loss_prev - loss.item()) / abs(loss_prev)
            delta_x = torch.norm(x_curr - x_prev).item()
            if 0 <= delta_f <= self.tol:# and delta_x <= self.tol:
                converged = True
                pbar.set_postfix(ftol=f"{delta_f:.4e}", xtol=f"{delta_x:.4e}", loss=f"{loss.item():.4f}")
                break
            loss_prev = loss.item()
            x_prev = x_curr.clone()
            all_x.append(x_curr.clone().detach())
            pbar.set_postfix(ftol=f"{delta_f:.4e}", xtol=f"{delta_x:.4e}", loss=f"{loss.item():.4f}")
        x = x_polar * non_slk_bus + slk_vec
        T, V = x[:nb], x[nb:]

        return x, T, V, converged, it, loss.item(), all_x


class LBFGS_se_latent(SEOptimizer):
    def __init__(self, **kwargs):
        super(LBFGS_se_latent, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = int(kwargs.get('max_iter', 1000))
        self.verbose = kwargs.get('verbose', True)
        self.z, self.v, self.R,self.slk_bus, self.h_ac, self.nb = None, None, None, None, None, None
        self.use_NF_prior = False
        self.flow_model,self.prior_dist, self.std_NF,self.m_NF = None, None, None, None
        self.use_VAE_prior = False
        self.vae_model, self.num_samples, self.sigma_p, self.edge_index = None, None, None, None
        self.lh_dist = None
        self.load_NF_prior(**kwargs)
        self.load_VAE_prior(**kwargs)
        self.same_start = kwargs.get('same_start', False)
        self.prefix = '-ss' if self.same_start else ''

    def load_NF_prior(self, **kwargs):
        self.use_NF_prior = kwargs.get('use_nf_prior', False)
        if self.use_NF_prior:
            NF_config_path = kwargs['prior_config_path']
            NF_config = json.load(open(NF_config_path))
            ckpt_path = f"../learn_prior/NF/models/{NF_config.get('ckpt_name')}"
            self.flow_model = FlowModel(**NF_config).to(device='cpu')
            self.flow_model.load_state_dict(torch.load(ckpt_path))
            self.flow_model.eval()
            self.m_NF = torch.load("../learn_prior/datasets/mean_polar.pt")
            self.std_NF = torch.load("../learn_prior/datasets/std_polar.pt")
            nb = self.m_NF.shape[0]
            self.prior_dist = MultivariateNormal(torch.zeros(nb), torch.eye(nb))

    def load_VAE_prior(self, **kwargs):
        self.use_VAE_prior = kwargs.get('use_vae_prior', False)
        if self.use_VAE_prior:
            VAE_config_path = kwargs['prior_config_path']
            VAE_config = json.load(open(VAE_config_path))
            ckpt_path = f"../learn_prior/VAE/models/{VAE_config.get('ckpt_name')}"
            self.vae_model = AmortizedVAE(**VAE_config).to(device='cpu')
            self.vae_model.load_state_dict(torch.load(ckpt_path))
            self.vae_model.eval()
            self.m_NF = torch.load("../learn_prior/datasets/mean_polar.pt").to(torch.get_default_dtype())
            self.std_NF = torch.load("../learn_prior/datasets/std_polar.pt").to(torch.get_default_dtype())
            self.num_samples = VAE_config['num_samples']
            self.sigma_p = VAE_config['sigma_p']
            self.edge_index = kwargs.get('edge_index')
            nb = self.m_NF.shape[0]
            self.prior_dist = MultivariateNormal(torch.zeros(nb), torch.eye(nb))

    def update_attr(self, *args):
        self.z = args[0]
        self.v = args[1]
        self.R = torch.diag(1.0 / torch.sqrt(self.v))
        self.slk_bus = args[2]
        self.h_ac = args[3]
        self.nb = args[4]
        self.lh_dist = MultivariateNormal(self.z, torch.diag(self.v))

    def _remove_slack(self, x, dim=0):
        s = int(self.slk_bus[0])
        if dim == 0:
            return torch.cat([x[:s], x[s + 1:]], dim=0)
        else:
            return torch.cat([x[:, :s], x[:, s + 1:]], dim=1)

    def _insert_at_slack(self, vec, num_to_insert=0.):
        s = int(self.slk_bus[0])
        return torch.cat([vec[:s], torch.ones(1) * num_to_insert, vec[s:]], dim=0)

    def _insert_slack_angle(self, vec):
        s = int(self.slk_bus[0])
        angle = torch.tensor([self.slk_bus[1]])
        return torch.cat([vec[:s], angle, vec[s:]], dim=0)

    def vae_decode(self, eps):
        x_norm = self.vae_model.decode(eps.unsqueeze(0), edge_index=self.edge_index).squeeze(0)
        x = ((x_norm * self.std_NF) + self.m_NF)
        x = self._insert_slack_angle(x)
        return x

    def vae_encode(self, x):
        x_red = self._remove_slack(x)
        x_norm = ((x_red - self.m_NF) / self.std_NF).unsqueeze(0)
        eps, _ = self.vae_model.encode(x_norm.unsqueeze(0), edge_index=self.edge_index)
        return eps.squeeze(0)

    def compute_composition_vae(self, eps):
        x = self.vae_decode(eps)
        z_est = self.h_ac.estimate(x)
        return z_est

    def nf_prior_eps(self, x):
        x_red = self._remove_slack(x)
        x_norm = ((x_red - self.m_NF) / self.std_NF).unsqueeze(0)
        eps = self.flow_model.inverse(x_norm).squeeze(0)
        return eps

    def nf_forward_eps(self, eps):
        x_norm = self.flow_model(eps.unsqueeze(0)).squeeze(0)
        x = ((x_norm * self.std_NF) + self.m_NF)
        x = self._insert_slack_angle(x)
        return x

    def compute_composition_nf(self, eps):
        x = self.nf_forward_eps(eps)
        z_est = self.h_ac.estimate(x)
        return z_est

    def se_loss_nf(self, eps):
        z_est = self.compute_composition_nf(eps)
        # lh = torch.norm(self.R @ (self.z - z_est)).pow(2)
        lh = -self.lh_dist.log_prob(z_est)
        # prior = torch.norm(eps).pow(2)
        prior = -self.prior_dist.log_prob(eps)
        return lh + prior

    def se_loss_vae(self, eps):
        z_est = self.compute_composition_vae(eps)
        # lh = torch.norm(self.R @ (self.z - z_est)).pow(2)
        lh = -self.lh_dist.log_prob(z_est)
        # prior = torch.norm(eps).pow(2)
        prior = -self.prior_dist.log_prob(eps)
        return lh + prior

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        self.update_attr(z, v, slk_bus, h_ac, nb)
        if self.same_start:
            x = x0.clone()
            if self.use_NF_prior:
                eps = self.nf_prior_eps(x).detach()
            else:
                eps = self.vae_encode(x).detach()
        else:
            eps = torch.zeros(nb * 2 - 1)

        all_x = [x0.clone().detach()]
        eps.requires_grad = True

        if self.use_NF_prior:
            criterion = self.se_loss_nf
        else:
            criterion = self.se_loss_vae
        optimizer = LBFGS([eps], line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            curr_loss = criterion(eps)
            curr_loss.backward()
            return curr_loss

        converged = False
        it, delta = 0, torch.inf
        with torch.no_grad():
            loss_prev = criterion(eps).item()
        pbar = tqdm(range(int(self.max_iter)), desc=f'Optimizing with LBFGS latent{self.prefix}', leave=True, colour='green',
                    postfix={'loss': f"{loss_prev:.4f}"})
        for it in pbar:
            optimizer.step(closure)
            if self.use_NF_prior:
                all_x.append(self.nf_forward_eps(eps).detach())
            else:
                all_x.append(self.vae_decode(eps).detach().clone())
            with torch.no_grad():
                loss = criterion(eps)
            delta = loss_prev - loss.item()
            if 0 <= delta < self.tol:
                converged = True
                break
            loss_prev = loss.item()
            pbar.set_postfix(delta=f"{delta:.4e}", loss=f"{loss.item():.4f}")
        if self.use_NF_prior:
            x = self.nf_forward_eps(eps)
        else:
            x = self.vae_decode(eps)
        T, V = x[:nb], x[nb:]
        return x, T, V, converged, it, loss.item(), all_x


class LBFGS_se_cart(SEOptimizer):
    def __init__(self, **kwargs):
        super(LBFGS_se_cart, self).__init__(**kwargs)
        self.slk_vec = None
        self.non_slk_bus = None
        self.u = None
        self.slk_angle = None
        self.slk_idx_imag = None
        self.slk_idx_real = None
        self.norm_H = None
        self.nb = None
        self.h_ac = None
        self.slk_bus = None
        self.R = None
        self.v = None
        self.z = None
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = int(kwargs.get('max_iter', 500))
        self.verbose = kwargs.get('verbose', True)
        self.use_prior = kwargs.get('use_prior', False)
        self.prior_config_path = kwargs.get('prior_config_path')
        self.device = kwargs.get('device', 'mps')
        self.dtype = kwargs.get('dtype', torch.get_default_dtype())

    def update_attr(self, *args):
        self.z = args[0].to(self.device, self.dtype)
        self.v = args[1].to(self.device, self.dtype)
        self.R = torch.diag(1.0 / self.v).to(self.device, self.dtype)
        self.slk_bus = args[2]
        self.h_ac = args[3]
        self.h_ac._update_device_dtype(self.device, self.dtype)
        self.nb = args[4]
        self.norm_H = (args[5] if args[5] is not None else torch.ones_like(self.z)).to(self.device, self.dtype)

        self.slk_idx_real = self.slk_bus[0]
        self.slk_idx_imag = self.slk_bus[0] + self.nb

        self.slk_angle = torch.tensor([self.slk_bus[1]]).to(self.device, self.dtype)
        self.u = torch.zeros(self.nb * 2).to(self.device, self.dtype)
        self.u[[self.slk_idx_real, self.slk_idx_imag]] = torch.tensor([torch.cos(self.slk_angle), torch.sin(self.slk_angle)]).to(self.device, self.dtype)
        self.non_slk_bus = (~torch.isin(torch.arange(self.nb * 2),
                                  torch.tensor([self.slk_idx_real, self.slk_idx_imag]))).to(self.device, self.dtype)
        self.slk_vec = 1 - self.non_slk_bus

    def _project_x(self, x):
        x_proj = (x * self.non_slk_bus) + (x * self.slk_vec).dot(self.u) * self.u
        return x_proj

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        x_cart = x0.clone().to('mps', torch.get_default_dtype())
        x_cart.requires_grad_(True)
        self.update_attr(z, v, slk_bus, h_ac, nb, norm_H)
        non_slk_bus = (torch.arange(nb) != slk_bus[0]).type(torch.get_default_dtype())
        slk_vec = torch.zeros_like(non_slk_bus)
        slk_vec[slk_bus[0]] = slk_bus[1]

        if self.use_prior:
            criterion = WeightedSECartWithPriorLoss(self.z, self.R, self.slk_idx_imag,
                                                    norm_H=self.norm_H, NF_config_path=self.prior_config_path)
        else:
            criterion = WeightedSELoss(self.z, self.R, norm_H=self.norm_H)
        optimizer = LBFGS([x_cart], line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            x_proj = self._project_x(x_cart)
            z_est_curr = self.h_ac.estimate(x_proj)
            if self.use_prior:
                loss_curr = criterion(z_est_curr, x_proj)
            else:
                loss_curr = criterion(z_est_curr)
            loss_curr.backward()
            return loss_curr

        converged = False
        it, delta = 0, torch.inf
        loss_prev = torch.inf
        pbar = tqdm(range(int(self.max_iter)), desc=f'Optimizing with LBFGS', leave=False, colour='green')
        for it in pbar:
            optimizer.step(closure)
            with torch.no_grad():
                x_proj = self._project_x(x_cart)
                z_est = self.h_ac.estimate(x_proj)
                if self.use_prior:
                    loss = criterion(z_est, x_proj)
                else:
                    loss = criterion(z_est)
            delta = loss_prev - loss.item()
            if 0 <= delta < self.tol:
                converged = True
                break
            loss_prev = loss.item()
            if self.verbose and it % (self.max_iter / 100) == 0:
                print(f'LBFGS - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}')
        x_cart = self._project_x(x_cart.detach())
        if self.verbose and it > 0:
            print(f'LBFGS - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}')

        x_cart = x_cart.to('cpu', torch.get_default_dtype())
        Vc = x_cart[:self.nb] + 1j * x_cart[self.nb:]
        T, V = Vc.angle(), Vc.abs()
        return x_cart, T, V, converged, it


class GDBT_se(SEOptimizer):
    def __init__(self, **kwargs):
        super(GDBT_se, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-7)
        self.beta = kwargs.get('beta', 0.5)
        self.t0 = kwargs.get('t0', 1e-8)
        self.c = kwargs.get('c', 0.2)
        self.t_min = kwargs.get('t_min', 1e-12)
        self.max_iter = int(kwargs.get('max_iter', 2000))
        self.verbose = kwargs.get('verbose', True)
        self.use_prior = kwargs.get('use_prior', False)
        self.prior_config_path = kwargs.get('prior_config_path')

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        x = x0.clone()
        x.requires_grad = True
        all_x = [x.clone().detach()]
        non_slk_bus = (torch.arange(nb * 2) != slk_bus[0]).type(torch.get_default_dtype())
        slk_vec = torch.zeros_like(non_slk_bus)
        slk_vec[slk_bus[0]] = slk_bus[1]

        if self.use_prior:
            criterion = WeightedSEWithPriorLoss(z, v, slk_bus[0], NF_config_path=self.prior_config_path, norm_H=norm_H)
            final_opt = LMOpt(loss_func=SELossNF(NF_config_path=self.prior_config_path, with_log_det=True), verbose=False, delta_init=None)
        else:
            criterion = WeightedSELoss(z, v, norm_H=norm_H)
            final_opt = LMOpt(loss_func=SELoss(), verbose=False, delta_init=None)

        loss_prev = torch.inf
        z_est = h_ac.estimate(x)
        loss = criterion(z_est, x)
        x_prev = x
        converged = False
        k = 0
        pbar = tqdm(range(int(self.max_iter)), desc=f'Optimizing with GDBT', leave=True, colour='green',
                    postfix={'loss': f"{loss_prev:.4f}"})
        for k in pbar:

            z_est = h_ac.estimate(x)
            loss = criterion(z_est, x)
            loss.backward()
            g = x.grad
            g[slk_bus[0]] = 0
            norm_g = torch.linalg.norm(g)
            if norm_g == 0.0:
                delta_f = (loss_prev - loss.item()) / abs(loss_prev)
                converged = 0 <= delta_f <= self.tol
                x, T, V, _, k_lm, loss, all_x_lm = final_opt(x, z, v, slk_bus, h_ac, nb)
                all_x.extend(all_x_lm)
                k += k_lm
                pbar.set_postfix(loss=f"{loss:.4f}", k_lm=f"{k_lm}")
                return x, T, V, converged, k, loss.item(), all_x
            t = self.t0
            t_prev = self.t0

            while t >= self.t_min:
                x_new = (x - t * g).detach()
                if not torch.isfinite(x_new).all():
                    t_prev = t
                    t *= self.beta
                    continue
                z_est = h_ac.estimate(x_new)
                loss_new = criterion(z_est, x_new)
                if loss_new <= (1 - self.c * t) * loss :
                    if t >= t_prev:
                        t_prev = t
                        t /= self.beta
                        continue
                    loss_prev = loss
                    loss = loss_new
                    x_prev = x.detach()
                    x = x_new.detach()
                    all_x.append(x.detach())
                    x.requires_grad = True
                    break
                t_prev = t
                t *= self.beta

            delta_f = (loss_prev - loss.item()) / abs(loss_prev)
            pbar.set_postfix(ftol=f"{delta_f.item():.4e}", loss=f"{loss.item():.4f}", t=f"{t:.4e}")
            if t < self.t_min:
                x, T, V, _, k_lm, loss, all_x_lm = final_opt(x, z, v, slk_bus, h_ac, nb)
                all_x.extend(all_x_lm)
                k += k_lm
                pbar.set_postfix(loss=f"{loss:.4f}", k_lm=f"{k_lm}")
                return x, T, V, converged, k, loss, all_x

            converged = 0 < delta_f <= self.tol
            if converged:
                x, T, V, _, k_lm, loss, all_x_lm = final_opt(x, z, v, slk_bus, h_ac, nb)
                all_x.extend(all_x_lm)
                k += k_lm
                pbar.set_postfix(loss=f"{loss:.4f}", k_lm=f"{k_lm}")
                return x, T, V, converged, k, loss, all_x

        x, T, V, _, k_lm, loss, all_x_lm = final_opt(x, z, v, slk_bus, h_ac, nb)
        all_x.extend(all_x_lm)
        k += k_lm
        pbar.set_postfix(loss=f"{loss:.4f}", k_lm=f"{k_lm}")
        return x, T, V, converged, k, loss, all_x


class SGD_se(SEOptimizer):
    def __init__(self, **kwargs):
        super(SGD_se, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = int(kwargs.get('max_iter', 1e+6))
        self.lr = kwargs.get('lr', 1e-4)
        self.start_factor = kwargs.get('start_factor', 1e-5)
        self.end_factor = kwargs.get('end_factor', 1.5e-3)
        self.verbose = kwargs.get('verbose', True)
        self.use_prior = kwargs.get('use_prior', False)

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        T = x0[:nb].clone()
        T[slk_bus[0]] = slk_bus[1]
        V = x0[nb:].clone()
        T.requires_grad_(True)
        V.requires_grad_(True)
        non_slk_bus = (torch.arange(nb) != slk_bus[0]).type(torch.get_default_dtype())
        slk_vec = torch.zeros_like(non_slk_bus)
        slk_vec[slk_bus[0]] = slk_bus[1]

        R = torch.diag(1. / v)
        if self.use_prior:
            criterion = WeightedSEWithPriorLoss(z, R, slk_bus[0], norm_H=norm_H)
        else:
            criterion = WeightedSELoss(z, R, norm_H=norm_H)
        optimizer = SGD([T, V], lr=self.lr, momentum=.9, nesterov=True)

        scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.start_factor, end_factor=self.end_factor, total_iters=self.max_iter)
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                patience=0,
                                                                factor=1/1.001**2,
                                                                threshold_mode='abs',
                                                                threshold=0,
                                                                eps=1e-12)
        converged = False
        it, delta = 0, torch.inf
        loss_prev = torch.inf
        for it in tqdm(range(int(self.max_iter)), desc=f'Optimizing with SGD', leave=False, colour='green'):
            optimizer.zero_grad()
            T_non_slk = T * non_slk_bus + slk_vec
            z_est = h_ac.estimate(T_non_slk, V)
            if self.use_prior:
                x = torch.concat([T_non_slk, V]).unsqueeze(0)
                loss = criterion(z_est, x)
            else:
                loss = criterion(z_est)
            loss.backward()
            optimizer.step()
            delta = loss_prev - loss.item()
            if 0 <= delta < self.tol:
                converged = True
                break
            loss_prev = loss.item()
            scheduler1.step(epoch=None)
            scheduler2.step(loss_prev)

            if self.verbose and it % (self.max_iter / 100) == 0:
                print(
                    f'SGD - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}, lr: {scheduler2.get_last_lr()[0]:.5e}')
        T = T.detach() * non_slk_bus + slk_vec
        x = torch.concat([T, V], dim=0)

        if self.verbose and it > 0:
            print(f'SGD - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}')
        return x, T, V, converged, it


class Muon_se(SEOptimizer):
    def __init__(self, **kwargs):
        super(Muon_se, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = int(kwargs.get('max_iter', 1e+6))
        self.lr = kwargs.get('lr', 1e-3)
        self.patience = kwargs.get('patience', 1)
        self.factor = kwargs.get('factor', .99)
        self.threshold = kwargs.get('threshold', 0)
        self.eps = kwargs.get('eps', 1e-12)
        self.verbose = kwargs.get('verbose', True)
        self.use_prior = kwargs.get('use_prior', False)

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        T = x0[:nb].clone().reshape(-1, 1)
        T[slk_bus[0]] = slk_bus[1]
        V = x0[nb:].clone().reshape(-1, 1)
        T.requires_grad_(True)
        V.requires_grad_(True)
        non_slk_bus = (torch.arange(nb) != slk_bus[0]).type(torch.get_default_dtype()).reshape(-1, 1)
        slk_vec = torch.zeros_like(non_slk_bus).reshape(-1, 1)
        slk_vec[slk_bus[0]] = slk_bus[1]

        R = torch.diag(1.0 / v)
        if self.use_prior:
            criterion = WeightedSEWithPriorLoss(z, R, slk_bus[0], norm_H=norm_H)
        else:
            criterion = WeightedSELoss(z, R, norm_H=norm_H)
        optimizer = Muon([T, V], lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=self.patience,
                                                               factor=self.factor,
                                                               threshold=self.threshold,
                                                               eps=self.eps)
        converged = False
        it, delta = 0, torch.inf
        loss_prev = torch.inf
        for it in tqdm(range(int(self.max_iter)), desc=f'Optimizing with Muon', leave=False, colour='green'):
            optimizer.zero_grad()
            T_non_slk = T * non_slk_bus + slk_vec
            z_est = h_ac.estimate(T_non_slk, V)
            if self.use_prior:
                x = torch.concat([T_non_slk, V]).unsqueeze(0)
                loss = criterion(z_est, x)
            else:
                loss = criterion(z_est)
            loss.backward()
            optimizer.step()
            delta = loss_prev - loss.item()
            if 0 <= delta < self.tol:
                converged = True
                break
            loss_prev = loss.item()
            scheduler.step(loss_prev)

            if self.verbose and it % (self.max_iter / 100) == 0:
                print(
                    f'Muon - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.5e}')
        T = T.detach() * non_slk_bus + slk_vec
        T = T.flatten()
        V = V.flatten()
        x = torch.concat([T, V], dim=0)
        if self.verbose and it > 0:
            print(f'Muon - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.5e}')
        return x, T, V, converged, it


class AdamW_se(SEOptimizer):
    def __init__(self, **kwargs):
        super(AdamW_se, self).__init__(**kwargs)
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = int(kwargs.get('max_iter', 1e+6))
        self.lr = kwargs.get('lr', 1e-3)
        self.patience = kwargs.get('patience', 1)
        self.factor = kwargs.get('factor', .99)
        self.threshold = kwargs.get('threshold', 0)
        self.eps = kwargs.get('eps', 1e-12)
        self.verbose = kwargs.get('verbose', True)
        self.use_prior = kwargs.get('use_prior', False)

    def __call__(self, x0, z, v, slk_bus, h_ac, nb ,norm_H=None):
        T = x0[:nb].clone()
        T[slk_bus[0]] = slk_bus[1]
        V = x0[nb:].clone()
        T.requires_grad_(True)
        V.requires_grad_(True)
        non_slk_bus = (torch.arange(nb) != slk_bus[0]).type(torch.float32)
        slk_vec = torch.zeros_like(non_slk_bus)
        slk_vec[slk_bus[0]] = slk_bus[1]

        R = torch.diag(1.0 / v)
        if self.use_prior:
            criterion = WeightedSEWithPriorLoss(z, R, slk_bus[0], norm_H=norm_H)
        else:
            criterion = WeightedSELoss(z, R, norm_H=norm_H)
        optimizer = AdamW([T, V], lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=self.patience,
                                                               factor=self.factor,
                                                               threshold=self.threshold,
                                                               eps=self.eps)
        converged = False
        it, delta = 0, torch.inf
        loss_prev = torch.inf
        for it in tqdm(range(int(self.max_iter)), desc=f'Optimizing with AdamW', leave=False, colour='green'):
            optimizer.zero_grad()
            T_non_slk = T * non_slk_bus + slk_vec
            z_est = h_ac.estimate(T_non_slk, V)
            if self.use_prior:
                x = torch.concat([T_non_slk, V]).unsqueeze(0)
                loss = criterion(z_est, x)
            else:
                loss = criterion(z_est)
            loss.backward()
            optimizer.step()
            delta = loss_prev - loss.item()
            if 0 <= delta < self.tol:
                converged = True
                break
            loss_prev = loss.item()
            scheduler.step(loss_prev)

            if self.verbose and it % (self.max_iter / 100) == 0:
                print(
                    f'AdamW - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.5e}')
        T = T.detach() * non_slk_bus + slk_vec
        x = torch.concat([T, V], dim=0)
        if self.verbose and it > 0:
            print(f'AdamW - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.5e}')
        return x, T, V, converged, it

