"""MAP loss functions for LM/L-BFGS state estimation.

Each `SELoss` subclass supplies the residual, objective, and Jacobian of a
data-fit term plus an optional prior term, in either state space or a prior's
latent space. The variants used by the paper are:

  * `SELossCart`        -- data-fit only (model-only LM).
  * `SELossGNCartLat`   -- Gaussian latent prior (LMGS).
  * `SELossCartNFLat`   -- normalizing-flow latent prior; `with_log_det`
                           toggles the change-of-variables term (LMNF vs LMNF-NLD).
  * `SELossFM` / `SELossFlowerCart` -- flow-matching prior (FLOWER).

The `load_*_prior` helpers build the prior tensors/models from the checkpoints
and normalization stats under `learn_prior/models` and `data_parser/data`.
"""

import json
import os
from pathlib import Path
import re

from abc import ABC

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal
from torch.func import jacrev, vmap, jacfwd
from SE_torch.learn_prior.NF.nf_model import (
    create_multiscale_flow, create_voltage_flow,
    MinMaxTransform, Normalize,
    _load_topology,
)
from SE_torch.learn_prior.NF.NF import FlowModel
from SE_torch.learn_prior.NF.NF_compact_strong import FlowModel as FlowModelCompactStrong

from SE_torch.learn_prior.FM.FM import FlowMatching

DEFAULT_NB = 118
SCALE_V = 0.025
SCALE_T = torch.sqrt(torch.tensor(torch.pi)) / 2.
_DATASETS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data_parser", "data", "time_series4"))
_NF_MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "learn_prior", "models"))
_NF_CONFIGS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "learn_prior", "configs"))
_NETS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "nets"))


def _resolve_net_file(path):
    """
    Resolve the path to a network .mat file referenced in an NF config.
    Configs typically use a path relative to `SE_torch/learn_prior/NF/`
    (e.g. ``"../../../nets/ieee118_186.mat"``), which doesn't work when
    se_loss.py is imported from a different CWD. We try, in order:
      1. the path as-is,
      2. `<_NETS_DIR>/<basename(path)>`  — fixed nets dir at the project root,
      3. `<_NF_CONFIGS_DIR>/<path>`      — relative to the configs dir,
      4. the original string (so downstream raises a clear error).
    """
    if path is None:
        return None
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join(_NETS_DIR, os.path.basename(path)),
        os.path.normpath(os.path.join(_NF_CONFIGS_DIR, path)),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return path


def _build_nf_model(nf_config, device='cpu'):
    model_type = nf_config.get("model_type", "legacy_realnvp")
    device = torch.device(device)
    if model_type in {"legacy_realnvp", "realnvp", "nf"}:
        return FlowModel(**nf_config).to(device=device, dtype=torch.get_default_dtype())
    if model_type in {"compact_strong", "nf_compact_strong"}:
        return FlowModelCompactStrong(**nf_config).to(device=device, dtype=torch.get_default_dtype())
    raise ValueError(f"Unsupported NF model_type '{model_type}'")


def _reduce_GN_prior(m, Q, nb, slk_idx):
    if m.ndim != 1:
        raise ValueError(f"GN prior mean must be 1D, got shape {tuple(m.shape)}")
    if Q.ndim != 2:
        raise ValueError(f"GN prior covariance must be 2D, got shape {tuple(Q.shape)}")
    if Q.shape[0] != Q.shape[1] or Q.shape[0] != m.shape[0]:
        raise ValueError(f"GN prior mean/cov mismatch: m={tuple(m.shape)}, Q={tuple(Q.shape)}")

    d_full = nb * 2
    d_red = d_full - 1

    if m.shape[0] == d_red:
        return m, Q
    if m.shape[0] != d_full:
        raise ValueError(
            f"Unsupported GN prior dimension {m.shape[0]}; expected {d_red} (reduced) or {d_full} (full)."
        )

    mask = torch.ones(m.shape[0], dtype=torch.bool, device=m.device)
    mask[slk_idx] = False
    return m[mask], Q[mask][:, mask]


def _make_GN_precision_factors(Q_red, tol=1e-6):
    # mask = torch.diag(Q_red.diag() <= 1e-6).to(torch.get_default_dtype()) * 1e-6
    # Q_red += mask
    Q_red += torch.eye(Q_red.shape[0]) * tol
    evals, evecs = torch.linalg.eigh(Q_red)
    evals = torch.clamp(evals, min=0)
    inv = torch.zeros_like(evals)
    inv_sqrt = torch.zeros_like(evals)
    keep = evals > tol
    inv[keep] = 1.0 / evals[keep]
    inv_sqrt[keep] = torch.rsqrt(evals[keep])
    # Q_inv = evecs.T @ torch.diag(inv) @ evecs
    # L = torch.diag(inv_sqrt) @ evecs.T
    L = torch.linalg.cholesky(Q_red)
    L_inv = torch.linalg.inv(L)
    return L, L_inv


def _make_nf_normalization_factors(mean, cov, std, normalization, tol=1e-6, data_min=None, data_max=None):
    dim = mean.shape[0]
    eye = torch.eye(dim, dtype=mean.dtype, device=mean.device)

    if normalization == "chol":
        return mean, *_make_GN_precision_factors(cov, tol=tol)
    if normalization == "diag":
        std_diag = torch.diag(std.diag().clamp_min(1e-6))
        std_inv = torch.diag(1.0 / std_diag.diag())
        return mean, std_diag, std_inv
    if normalization == "minmax":
        if data_min is None or data_max is None:
            raise ValueError("Min-max normalization requires both 'data_min' and 'data_max'.")
        center = 0.5 * (data_max + data_min)
        half_range = (0.5 * (data_max - data_min)).clamp_min(1e-6)
        scale = torch.diag(half_range)
        scale_inv = torch.diag(1.0 / half_range)
        return center, scale, scale_inv
    if normalization == "mean":
        return mean, eye, eye
    if normalization == "none":
        return torch.zeros_like(mean), eye, eye
    raise ValueError(f"Unsupported data_normalization '{normalization}'")


def load_gaussian_model_prior(**kwargs):
    nb = kwargs.get('nb', DEFAULT_NB)
    slk_idx = kwargs.get('slk_idx', 68)
    prior_tol = kwargs.get('prior_tol', 1e-6)
    cart = kwargs.get('cart', False)
    device = torch.device(kwargs.get('device', 'cpu'))
    slk_idx = slk_idx + nb if cart else slk_idx
    prefix = 'vri' if cart else 'vam'
    mean_path = kwargs.get('mean_path', os.path.join(_DATASETS_DIR, f"mean_gaussian_{prefix}.pt"))
    cov_path = kwargs.get('cov_path', os.path.join(_DATASETS_DIR, f"cov_{prefix}.pt"))
    m = torch.load(mean_path).to(device=device, dtype=torch.get_default_dtype())
    Q = torch.load(cov_path).to(device=device, dtype=torch.get_default_dtype())

    m_red, Q_red = _reduce_GN_prior(m, Q, nb=nb, slk_idx=slk_idx)
    L, L_inv = _make_GN_precision_factors(Q_red, tol=prior_tol)

    return m_red, Q_red, L, L_inv


def load_GN_data_prior(**kwargs):
    nb = kwargs.get('nb', DEFAULT_NB)
    slk_idx = kwargs.get('slk_idx', 68)
    prior_tol = kwargs.get('prior_tol', 1e-6)
    cart = kwargs.get('cart', False)
    prefix = '_cart' if cart else '_polar'
    mean_path = kwargs.get('mean_path', os.path.join(_DATASETS_DIR, f"mean_NF{prefix}.pt"))
    cov_path = kwargs.get('cov_path', os.path.join(_DATASETS_DIR, f"cov_NF{prefix}.pt"))
    m = torch.load(mean_path).to(torch.get_default_dtype())
    Q = torch.load(cov_path).to(torch.get_default_dtype())

    m_red, Q_red = _reduce_GN_prior(m, Q, nb=nb, slk_idx=slk_idx)
    L, L_inv = _make_GN_precision_factors(Q_red, tol=prior_tol)

    return m_red, Q_red, L, L_inv


def load_PCA_Gaussian_prior(**kwargs):
    prior_tol = kwargs.get('prior_tol', 0.)
    prior_path = kwargs.get('PCA_prior_path', os.path.join(_DATASETS_DIR, "pca_gaussian_polar.pt"))
    components_path = kwargs.get('PCA_components_path', os.path.join(_DATASETS_DIR, "pca_components_polar.pt"))
    data_mean_path = kwargs.get('PCA_data_mean_path', os.path.join(_DATASETS_DIR, "mean_PCA_data_polar.pt"))
    latent_mean_path = kwargs.get('PCA_mean_path', os.path.join(_DATASETS_DIR, "mean_PCA_polar.pt"))
    latent_cov_path = kwargs.get('PCA_cov_path', os.path.join(_DATASETS_DIR, "cov_PCA_polar.pt"))

    if os.path.exists(prior_path):
        prior_bundle = torch.load(prior_path, map_location='cpu')
        data_mean = prior_bundle["data_mean"].to(torch.get_default_dtype())
        components = prior_bundle["components"].to(torch.get_default_dtype())
        latent_mean = prior_bundle["latent_mean"].to(torch.get_default_dtype())
        latent_cov = prior_bundle["latent_cov"].to(torch.get_default_dtype())
    else:
        data_mean = torch.load(data_mean_path, map_location='cpu').to(torch.get_default_dtype())
        components = torch.load(components_path, map_location='cpu').to(torch.get_default_dtype())
        latent_mean = torch.load(latent_mean_path, map_location='cpu').to(torch.get_default_dtype())
        latent_cov = torch.load(latent_cov_path, map_location='cpu').to(torch.get_default_dtype())

    latent_cov_factor, latent_precision_factor = _make_GN_precision_factors(latent_cov, tol=prior_tol)
    return data_mean, components, latent_mean, latent_cov, latent_cov_factor, latent_precision_factor


def load_NF_prior(**kwargs):
    with_log_det = kwargs.get('with_log_det', False)
    edge_index  = kwargs.get('edge_index', None)
    NF_config = kwargs.get('NF_config', json.load(open(kwargs['NF_config_path'], 'r')))
    cart = NF_config.get('cart', False)
    prefix = '_cart' if cart else '_polar'
    device = torch.device(kwargs.get('device', NF_config.get('device', 'cpu')))
    NF_config['device'] = str(device)
    NF_config['edge_index'] = edge_index
    ckpt_name = NF_config.get('ckpt_name')
    ckpt_path = NF_config.get('ckpt_path', os.path.join(_NF_MODELS_DIR, ckpt_name))
    flow_model = _build_nf_model(NF_config, device=device)
    flow_model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    flow_model.to(device=device, dtype=torch.get_default_dtype())
    flow_model.eval()
    mean_path = NF_config.get('mean_path', os.path.join(_DATASETS_DIR, f"mean_NF{prefix}.pt"))
    cov_path = NF_config.get('cov_path', os.path.join(_DATASETS_DIR, f"cov_NF{prefix}.pt"))
    std_path = NF_config.get('std_path', os.path.join(_DATASETS_DIR, f"std_NF{prefix}.pt"))
    min_path = NF_config.get('min_path', os.path.join(_DATASETS_DIR, f"min_NF{prefix}.pt"))
    max_path = NF_config.get('max_path', os.path.join(_DATASETS_DIR, f"max_NF{prefix}.pt"))
    normalization = NF_config.get('data_normalization', 'diag')
    prior_tol = kwargs.get('prior_tol', 1e-6)
    m_NF = torch.load(mean_path, map_location=device).to(torch.get_default_dtype())
    cov_NF = torch.load(cov_path, map_location=device).to(torch.get_default_dtype())
    std_NF = torch.load(std_path, map_location=device).to(torch.get_default_dtype())
    data_min = data_max = None
    if normalization == "minmax":
        data_min = torch.load(min_path, map_location=device).to(torch.get_default_dtype())
        data_max = torch.load(max_path, map_location=device).to(torch.get_default_dtype())
    m_NF, L_NF, L_inv_NF = _make_nf_normalization_factors(
        mean=m_NF,
        cov=cov_NF,
        std=std_NF,
        normalization=normalization,
        tol=prior_tol,
        data_min=data_min,
        data_max=data_max,
    )

    return m_NF, L_NF, L_inv_NF, flow_model, with_log_det


def _pick_best_nf_ckpt(epochs_path, model_name):
    """
    Locate the best NF checkpoint under `epochs_path` (searched recursively),
    preferring (in order):

      1. `<model_name>_best.pth` — written directly by `VoltageFlowTrainer`
         (the new raw-torch trainer in `nf_model.py`).
      2. Any `<model_name>_epoch*.pth` — pick the one with the lowest
         recorded `val_bpd` (per-epoch checkpoints from `VoltageFlowTrainer`
         carry that field). Files that don't carry it are treated as +inf.
      3. Legacy PyTorch Lightning `*.ckpt` files — pick by lowest reported
         `val_bpd` (read from the `ModelCheckpoint` callback dict, falling
         back to a top-level `val_bpd` key if present).
      4. Final fallback (only if no val score is available anywhere):
         the highest-epoch Lightning ckpt — i.e. the previous behaviour of
         this loader.
    """
    root = Path(epochs_path)

    # 1) explicit best from the new trainer
    best_matches = list(root.rglob(f"{model_name}_best.pth"))
    if best_matches:
        return best_matches[0]

    # # 2) per-epoch torch checkpoints from VoltageFlowTrainer
    pth_files = list(root.rglob(f"{model_name}_epoch*.pth"))
    if pth_files:
        scored = []
        for p in pth_files:
            try:
                st = torch.load(p, map_location='cpu')
                v = float(st.get('val_bpd', float('inf'))) \
                    if isinstance(st, dict) else float('inf')
            except Exception:
                v = float('inf')
            scored.append((p, v))
        return min(scored, key=lambda pv: pv[1])[0]

    # 3) legacy Lightning .ckpt files
    ckpts = list(root.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No NF checkpoint (.pth or .ckpt) found under {root}"
        )

    scored = []
    for p in ckpts:
        val = None
        try:
            st = torch.load(p, map_location='cpu')
            if isinstance(st, dict):
                cb = st.get('callbacks', {})
                if isinstance(cb, dict):
                    for k, v in cb.items():
                        if 'ModelCheckpoint' in str(k) and isinstance(v, dict):
                            cand = v.get('best_model_score', v.get('current_score'))
                            if cand is not None:
                                val = float(cand.item()) if hasattr(cand, 'item') else float(cand)
                            break
                if val is None and st.get('val_bpd') is not None:
                    val = float(st['val_bpd'])
        except Exception:
            val = None
        scored.append((p, val))

    with_val = [(p, v) for p, v in scored if v is not None]
    if with_val:
        return min(with_val, key=lambda pv: pv[1])[0]

    # 4) ultimate fallback — highest epoch number in filename
    ep_re = re.compile(r"epoch=(\d+)")
    with_ep = [(p, int(ep_re.search(p.name).group(1)))
               for p in ckpts if ep_re.search(p.name)]
    if with_ep:
        return max(with_ep, key=lambda pe: pe[1])[0]
    return ckpts[-1]


def load_nf_model_prior(**kwargs):
    with_log_det = kwargs.get('with_log_det', False)
    nf_model_config_name = kwargs.get('nf_model_config_name', 'nf_model_config_v0.1.json')
    nf_config_path = os.path.join(_NF_CONFIGS_DIR, nf_model_config_name + '.json')
    nf_model_config = json.load(open(nf_config_path, 'r'))
    cart = nf_model_config.get('cart', False)
    prefix = 'vri' if cart else 'vam'
    device = kwargs.get('device', 'cpu')
    checkpoint_path = str(nf_model_config.get('checkpoint_path', 'nf_model_config_v0.1'))
    model_name = str(nf_model_config.get('model_name', 'VoltageFlow_v0.0'))
    epochs_path = os.path.relpath(os.path.join(_NF_MODELS_DIR, checkpoint_path, model_name), os.getcwd())
    ckpt_path = _pick_best_nf_ckpt(epochs_path, model_name)
    print(f"[load_nf_model_prior] loading best NF checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Topology for the smart bus mask
    # ------------------------------------------------------------------
    # When the config was trained with `mask_type='smart'`, the coupling
    # layers need the same `edge_index` / `Y` (admittance matrix) that were
    # used at training time — otherwise the mask is built from random data
    # and the loaded weights won't make sense.
    #
    # The caller may pass `edge_index` and/or `Y` explicitly via kwargs (we
    # prefer those). Otherwise, if the config requests smart masks, we load
    # the topology from the .mat net file referenced in the config (resolved
    # via `_resolve_net_file` so a relative `file` path works from any CWD).
    flow_kwargs = dict(nf_model_config)
    caller_edge_index = kwargs.get('edge_index', None)
    caller_Y = kwargs.get('Y', None)
    if caller_edge_index is not None:
        flow_kwargs['edge_index'] = caller_edge_index
    if caller_Y is not None:
        flow_kwargs['Y'] = caller_Y

    need_topology = (
        flow_kwargs.get('mask_type') == 'smart'
        or bool(flow_kwargs.get('load_topology', False))
    )
    if need_topology and ('edge_index' not in flow_kwargs
                          or 'Y' not in flow_kwargs):
        topo_cfg = dict(nf_model_config)
        topo_cfg['file'] = _resolve_net_file(topo_cfg.get('file'))
        edge_index, Ybus, slk_bus = _load_topology(topo_cfg)
        flow_kwargs.setdefault('edge_index', edge_index)
        flow_kwargs.setdefault('Y', Ybus)
        print(f"[load_nf_model_prior] loaded topology for smart mask from "
              f"'{topo_cfg['file']}': h={Ybus.shape[0]}, "
              f"E={edge_index.shape[1]}, slack_bus(orig)={slk_bus}")

    device = torch.device(device)
    flow_model = create_voltage_flow(device=device, **flow_kwargs)
    ckpt = torch.load(ckpt_path, map_location=device)
    flow_model.load_state_dict(ckpt['state_dict'])
    flow_model.to(device=device, dtype=torch.get_default_dtype())
    flow_model.eval()
    mean_path = nf_model_config.get('mean_path', os.path.join(_DATASETS_DIR, f"mean_{prefix}.pt"))
    std_path = nf_model_config.get('std_path', os.path.join(_DATASETS_DIR, f"std_{prefix}.pt"))
    mean_val = torch.load(mean_path).to(device=device, dtype=torch.get_default_dtype())
    std_val = torch.load(std_path).to(device=device, dtype=torch.get_default_dtype())
    scaler = Normalize(mean_val, std_val)

    return flow_model, scaler, with_log_det


def load_FM_prior(**kwargs):
    FM_config_path = kwargs['FM_config_path']
    FM_config = json.load(open(FM_config_path))
    ckpt_path = f"../learn_prior/FM/models/{FM_config.get('ckpt_name')}"
    flow_model = FlowMatching(**FM_config).to(device='cpu')
    flow_model.load_state_dict(torch.load(ckpt_path))
    flow_model.eval()
    m_FM = torch.load("../learn_prior/datasets/mean_NF_polar.pt").to(torch.get_default_dtype())
    std_FM = torch.load("../learn_prior/datasets/std_NF_polar.pt").to(torch.get_default_dtype())

    return m_FM, std_FM, flow_model


class LossFunction(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def x_init(self, x):
        pass

    def reshape_x(self, x):
        return x

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def update_params(self, *args):
        pass

    def compute_residuals(self, x):
        raise NotImplementedError

    def compute_f(self, x):
        raise NotImplementedError

    def compute_J(self, x):
        raise NotImplementedError

    def compute_grad(self, x):
        raise NotImplementedError

    def likelihood_loss(self, x):
        raise NotImplementedError

    def prior_loss(self, x):
        raise NotImplementedError


class SELoss(LossFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Device the loss operates on. Defaults to CPU for backwards
        # compatibility. The LM optimiser moves its own tensors to
        # this same device at the top of `__call__`.
        self.device = torch.device(kwargs.get('device', 'cpu'))
        self.z = kwargs.get('z')
        self.v = kwargs.get('v')
        self.R = kwargs.get('R')
        self.h_ac = kwargs.get('h_ac')
        self.slk_bus = kwargs.get('slk_bus')
        self.nb = kwargs.get('nb')
        self.norm_H = kwargs.get('norm_H')
        self.prior_scale = kwargs.get('prior_scale', torch.tensor(1.)).to(
            device=self.device, dtype=torch.get_default_dtype())
        self.ll_scale = kwargs.get("ll_scale", torch.tensor(1.)).to(
            device=self.device, dtype=torch.get_default_dtype())
        self.lh_dist = kwargs.get('lh_dist')
        self.A = kwargs.get('A')
        self.B = kwargs.get('B')
        self.b = kwargs.get('b')

    def x_init(self, x):
        return x.clone().to(torch.get_default_dtype())

    def update_params(self, *args):
        self.z = args[0].to(self.device)
        self.v = args[1].to(self.device)
        self.slk_bus = args[2]
        self.h_ac = args[3]
        self.nb = args[4]
        self.R = torch.diag(1. / torch.sqrt(self.v))
        self.norm_H = (args[5].to(self.device) if args[5] is not None
                       else torch.ones_like(self.z))
        self.B = self._build_remove_slack_angle_operator(self.nb * 2)
        self.A, self.b = self._build_insert_slack_angle_operator(self.nb * 2)
        self.lh_dist = MultivariateNormal(self.z, torch.diag(self.v))

    def reshape_x(self, x):
        return self._remove_slack(x)

    def _split_x(self, x):
        T = x[:self.nb]
        V = x[self.nb:]
        return T, V

    def _remove_slack(self, x, dim=0):
        if dim == 0:
            return self.B @ x
        else:
            return x @ self.B.T

    def _insert_zero_at_slack(self, vec, dim=0):
        if dim == 0:
            return self.A @ vec
        else:
            return vec @ self.A.T

    def _build_remove_slack_angle_operator(self, n_full, s=None):
        if s is None:
            s = int(self.slk_bus[0])
        B = torch.eye(n_full, device=self.device)
        B = torch.cat([B[:s], B[s + 1:]], dim=0).to(
            device=self.device, dtype=torch.get_default_dtype())
        return B

    def _build_insert_slack_angle_operator(self, n_full: int):
        s = int(self.slk_bus[0])
        A = self.B.T
        b = torch.zeros((n_full,), device=self.device)
        b[s] = torch.as_tensor(self.slk_bus[1],
                                dtype=torch.get_default_dtype(),
                                device=self.device)
        return A, b

    def _insert_slack_angle_linear(self, vec):
        return self.A @ vec + self.b

    def _insert_slack_angle(self, vec):
        s = int(self.slk_bus[0])
        angle = torch.tensor([self.slk_bus[1]]).to(
            device=self.device, dtype=torch.get_default_dtype())
        return torch.cat([vec[:s], angle, vec[s:]], dim=0)

    def update_x(self, x, step):
        x_new = x.clone()
        if step is not None:
            step_curr = step.clone()
            if len(step) < len(x):
                step_curr = self._insert_zero_at_slack(step_curr)
            x_new = x + step_curr
        return x_new

    def se_res(self, x):
        z_est = self.h_ac.estimate(x)
        res = (self.R @ (self.z - z_est)) / self.norm_H
        return torch.sqrt(self.ll_scale) * res

    def se_jacobian(self, x, remove_slack=True):
        J = self.h_ac.jacobian(x)
        J = -(self.R @ J) / self.norm_H[:, None]
        if remove_slack:
            J = self._remove_slack(J, dim=1)
        return torch.sqrt(self.ll_scale) * J

    def se_f(self, x):
        z_est = self.h_ac.estimate(x)
        res_h = self.R @ (self.z - z_est) / self.norm_H
        f_h = .5 * torch.norm(res_h).pow(2)
        return self.ll_scale * f_h

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step)
        res = self.se_res(x)
        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)
        return f_h

    def compute_J(self, x, step=None):
        x = self.update_x(x, step)
        J = self.se_jacobian(x)
        return J

    def compute_grad(self, x, step=None):
        grad = torch.zeros_like(x)
        grad = self._remove_slack(grad)
        return grad

    def likelihood_loss(self, x):
        return self.se_f(x)

    def prior_loss(self, x):
        return torch.tensor(0.)


class SELossCart(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.x_slk = None
        self.non_slk_imag = None
        self.non_slk_real = None

        self.slk_idx_real = None
        self.slk_idx_imag = None
        self.slk_angle = None
        self.u = None

    def update_params(self, *args):
        self.z = args[0].to(self.device)
        self.v = args[1].to(self.device)
        self.R = torch.pinverse(torch.linalg.cholesky(torch.diag(self.v)))
        self.slk_bus = args[2]
        self.h_ac = args[3]
        self.nb = args[4]
        self.norm_H = (args[5].to(self.device) if args[5] is not None
                       else torch.ones_like(self.z))
        self.slk_idx_real = self.slk_bus[0]
        self.slk_idx_imag = self.slk_bus[0] + self.nb

        dev = self.device
        dt = torch.get_default_dtype()
        self.slk_angle = torch.tensor([self.slk_bus[1]],
                                       device=dev, dtype=dt)
        self.slk_mag = torch.tensor([self.slk_bus[2]],
                                      device=dev, dtype=dt)
        self.not_slk = torch.ones(self.nb * 2, device=dev)
        self.not_slk[[self.slk_idx_real, self.slk_idx_imag]] = 0.
        self.u = torch.zeros(self.nb * 2, device=dev)
        self.u[[self.slk_idx_real, self.slk_idx_imag]] = torch.tensor(
            [torch.cos(self.slk_angle), torch.sin(self.slk_angle)],
            device=dev, dtype=dt)
        self.non_slk_real = torch.concat([torch.arange(self.slk_idx_real, device=dev),
                                          torch.arange(self.slk_idx_real + 1, self.nb, device=dev)])
        self.non_slk_imag = torch.concat([torch.arange(self.nb - 1, self.slk_idx_imag - 1, device=dev),
                                          torch.arange(self.slk_idx_imag, self.nb * 2 - 1, device=dev)])
        self.slk_bus = (self.slk_idx_imag, torch.sin(self.slk_angle), torch.cos(self.slk_angle))
        self.B = self._build_remove_slack_angle_operator(self.nb * 2)
        self.B_imag = self._build_remove_slack_angle_operator(self.nb * 2, s=self.slk_idx_imag)
        self.B_real = self._build_remove_slack_angle_operator(self.nb * 2 - 1, s=self.slk_idx_real)
        self.A, self.b = self._build_insert_slack_angle_operator(self.nb * 2)


    def project_x(self, x):
        x_proj = self.not_slk * x + (x.dot(self.u) * self.u)
        return x_proj

    def decode(self, x):
        x = self.project_x(x)
        x_r, x_i = x[:self.nb], x[self.nb:]
        xc = x_r + 1j * x_i
        T, V = xc.angle(), xc.abs()
        x_polar = torch.cat([T, V])
        return x_polar

    def encode(self, x):
        T, V = x[:self.nb], x[self.nb:]
        xc = torch.polar(V, T)
        x_r, x_i = xc.real, xc.imag
        x_cart = torch.cat([x_r, x_i])
        x_cart = self.project_x(x_cart)
        return x_cart

    def update_x(self, x, step):
        if step is not None:
            if len(step) < len(x):
                step = self._insert_zero_at_slack(step)
            x = x + step
        x = self.project_x(x)
        return x


class SELossCartPrior(SELossCart):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prior_scale = kwargs.get('prior_scale', torch.tensor(1.)).to(torch.get_default_dtype())

    def prior_res(self, x):
        raise NotImplementedError

    def prior_jacobian(self, x, remove_slack=True):
        raise NotImplementedError

    def prior_f(self, x):
        raise NotImplementedError

    def prior_grad(self, x, remove_slack=True):
        raise NotImplementedError

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step)
        se_res = self.se_res(x)
        prior_res = self.prior_res(x)
        res = torch.cat([se_res, prior_res], dim=0)
        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        se_f = self.se_f(x)
        prior_f = self.prior_f(x)
        f = se_f + prior_f
        return f

    def compute_J(self, x, step=None):
        x = self.update_x(x, step)
        se_J = self.se_jacobian(x)
        prior_J = self.prior_jacobian(x)
        J = torch.cat([se_J, prior_J], dim=0)
        return J

    def compute_grad(self, x, step=None):
        x = self.update_x(x, step)
        prior_grad = self.prior_grad(x)
        return prior_grad

    def prior_loss(self, x):
        return self.prior_f(x)


class SELossCartPriorLat(SELossCartPrior):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _encode(self, x):
        raise NotImplementedError

    def encode(self, x):
        T, V = x[:self.nb], x[self.nb:]
        xc = torch.polar(V, T)
        x_r, x_i = xc.real, xc.imag
        x = torch.cat([x_r, x_i])
        x = self.project_x(x)
        return self._encode(x)

    def _decode(self, eps):
        raise NotImplementedError

    def decode(self, eps):
        x = self._decode(eps)
        x = self.project_x(x)
        x_r, x_i = x[:self.nb], x[self.nb:]
        xc = x_r + 1j * x_i
        T, V = xc.angle(), xc.abs()
        x = torch.cat([T, V])
        return x

    def reshape_x(self, eps):
        return eps

    def update_x(self, eps, step):
        if step is not None:
            eps = eps + step
        return eps

    def compute_residuals(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self._decode(eps)
        se_res = self.se_res(x)
        se_prior = self.prior_res(eps)
        res = torch.cat([se_res, se_prior], dim=0)
        return res

    def compute_f(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self._decode(eps)
        se_f = self.se_f(x)
        prior_f = self.prior_f(eps)
        return se_f + prior_f

    def compute_J(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self._decode(eps)
        se_J = self.se_jacobian(x, remove_slack=False)
        decode_J = jacrev(self._decode)(eps)
        se_decode_J = se_J @ decode_J
        prior_J = self.prior_jacobian(eps)
        J = torch.cat([se_decode_J, prior_J], dim=0)
        return J

    def likelihood_loss(self, eps):
        x = self._decode(eps)
        return self.se_f(x)


class SELossGN(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_red, self.Q_red, _, self.L = load_GN_data_prior(**kwargs)

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step)
        res_h = self.se_res(x)

        x_red = self._remove_slack(x)
        res_prior = self.L @ (x_red - self.m_red)
        res = torch.cat([res_h, torch.sqrt(self.prior_scale) * res_prior], dim=0)

        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)

        x_red = self._remove_slack(x)
        res_prior = self.L @ (x_red - self.m_red)
        f_prior = .5 * torch.norm(res_prior).pow(2)
        f = f_h + (self.prior_scale * f_prior)
        return f

    def compute_J(self, x, step=None):
        x = self.update_x(x, step)
        J_h = self.se_jacobian(x)
        J = torch.cat([J_h, torch.sqrt(self.prior_scale) * self.L], dim=0)
        return J


class SELossNFNorm(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_NF, self.L_NF, self.L_inv_NF, self.flow_model, self.with_log_det = load_NF_prior(**kwargs)
        self.cache_J_prior = None

    def reshape_x(self, x_norm):
        return x_norm

    def encode(self, x):
        x_red = self._remove_slack(x)
        x_norm = (x_red - self.m_NF) @ self.L_inv_NF.T
        return x_norm

    def decode(self, x_norm):
        x_red = ((self.L_NF @ x_norm) + self.m_NF)
        x = self._insert_slack_angle_linear(x_red)
        return x

    def nf_prior_eps(self, x_norm):
        eps = self.flow_model.inverse(x_norm.unsqueeze(0)).squeeze(0)
        return eps

    def nf_prior_log_det(self, x_norm):
        log_det = self.flow_model.log_det_inv_jacobian(x_norm.unsqueeze(0)).squeeze(0)
        return -log_det

    def compute_grad(self, x_norm, step=None):
        if self.with_log_det:
            x_for_grad = x_norm.clone().requires_grad_(True)
            logdet = self.nf_prior_log_det(x_for_grad)
            grad_log_det = torch.autograd.grad(logdet, x_for_grad, create_graph=False, retain_graph=False)[0]
            return self.prior_scale * grad_log_det

        return torch.zeros(self.nb * 2 - 1).to(torch.get_default_dtype())

    def compute_residuals(self, x_norm, step=None):
        x_norm = self.update_x(x_norm, step)
        x = self.decode(x_norm)
        res_h = self.se_res(x)
        res_prior = self.nf_prior_eps(x_norm)
        res = torch.cat([res_h, torch.sqrt(self.prior_scale) * res_prior], dim=0)

        return res

    def compute_f(self, x_norm, step=None):
        x_norm = self.update_x(x_norm, step)
        x = self.decode(x_norm)
        f_h = self.se_f(x)
        res_prior = self.nf_prior_eps(x_norm)
        f_prior = .5 * torch.norm(res_prior).pow(2)
        if self.with_log_det:
            log_det = self.nf_prior_log_det(x_norm)
            f_prior += log_det
        f = f_h + (self.prior_scale * f_prior)
        return f

    def compute_J(self, x_norm, step=None):
        x_norm = self.update_x(x_norm, step)
        x = self.decode(x_norm)
        J_h = self.se_jacobian(x) @ self.L_NF
        J_prior = jacrev(self.nf_prior_eps)(x_norm)
        self.cache_J_prior = J_prior
        J = torch.cat([J_h, torch.sqrt(self.prior_scale) * J_prior], dim=0)
        return J


class SELossNF(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_NF, self.L_NF, self.L_inv_NF, self.flow_model, self.with_log_det = load_NF_prior(**kwargs)
        self.cache_J_prior = None

    def reshape_x(self, x_red):
        return x_red

    def encode(self, x):
        x_red = self._remove_slack(x)
        return x_red

    def decode(self, x_red):
        x = self._insert_slack_angle_linear(x_red)
        return x

    def nf_prior_eps(self, x_red):
        x_norm = (x_red - self.m_NF) @ self.L_inv_NF.T
        eps = self.flow_model.inverse(x_norm.unsqueeze(0)).squeeze(0)
        return eps

    def nf_prior_log_det(self, x_red):
        x_norm = (x_red - self.m_NF) @ self.L_inv_NF.T
        log_det = self.flow_model.log_det_inv_jacobian(x_norm.unsqueeze(0)).squeeze(0)
        return -log_det

    def compute_grad(self, x_red, step=None):
        if self.with_log_det:
            grad_log_det = jacrev(self.nf_prior_log_det)(x_red)
            return self.prior_scale * grad_log_det

        return torch.zeros(self.nb * 2 - 1).to(torch.get_default_dtype())

    def compute_residuals(self, x_red, step=None):
        x_red = self.update_x(x_red, step)
        x = self.decode(x_red)
        res_h = self.se_res(x)
        res_prior = self.nf_prior_eps(x_red)
        res = torch.cat([res_h, torch.sqrt(self.prior_scale) * res_prior], dim=0)

        return res

    def compute_f(self, x_red, step=None):
        x_red = self.update_x(x_red, step)
        x = self.decode(x_red)
        f_h = self.se_f(x)
        res_prior = self.nf_prior_eps(x_red)
        f_prior = .5 * torch.norm(res_prior).pow(2)
        if self.with_log_det:
            log_det = self.nf_prior_log_det(x_red)
            f_prior += log_det
        f = f_h + (self.prior_scale * f_prior)
        return f

    def compute_J(self, x_red, step=None):
        x_red = self.update_x(x_red, step)
        x = self.decode(x_red)
        J_h = self.se_jacobian(x)
        J_prior = jacrev(self.nf_prior_eps)(x_red)
        self.cache_J_prior = J_prior
        J = torch.cat([J_h, torch.sqrt(self.prior_scale) * J_prior], dim=0)
        return J


class SELossNFLat(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_NF, self.L_NF, self.L_inv_NF, self.flow_model, self.with_log_det = load_NF_prior(**kwargs)

    def reshape_x(self, x):
        return x

    def nf_prior_log_det(self, eps):
        log_det = self.flow_model.log_det_jacobian(eps.unsqueeze(0)).squeeze(0)
        return log_det

    def compute_grad(self, eps, step=None):
        if self.with_log_det:
            grad_log_det = jacfwd(self.nf_prior_log_det)(eps)
            return self.prior_scale * grad_log_det
        return torch.zeros(self.nb * 2 - 1).to(torch.get_default_dtype())

    def decode(self, eps):
        x_norm = self.flow_model(eps.unsqueeze(0)).squeeze(0)
        x_red = ((self.L_NF @ x_norm) + self.m_NF)
        x = self._insert_slack_angle_linear(x_red)
        return x

    def encode(self, x):
        x_red = self._remove_slack(x)
        x_norm = (self.L_inv_NF @ (x_red - self.m_NF)).unsqueeze(0)
        eps = self.flow_model.inverse(x_norm).squeeze(0)
        return eps

    def compute_residuals(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self.decode(eps)
        res_h = self.se_res(x)
        res =  torch.cat([res_h, torch.sqrt(self.prior_scale) * eps], dim=0)
        return res

    def compute_f(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self.decode(eps)
        f_h = self.se_f(x)
        f_prior = .5 * torch.norm(eps).pow(2)
        if self.with_log_det:
            log_det = self.nf_prior_log_det(eps)
            f_prior += log_det
        f = f_h + self.prior_scale * f_prior
        return f

    def compute_J(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self.decode(eps)
        J_h = self.h_ac.jacobian(x)
        J_f = jacfwd(self.decode)(eps)
        J_h_f = -(self.R @ J_h @ J_f)
        J_prior = torch.eye((self.nb * 2 - 1))
        J = torch.cat([J_h_f, torch.sqrt(self.prior_scale) * J_prior], dim=0)

        return J


class SELossCartNF(SELossCart):
    """
    NF prior loss over cartesian state. The primal variable `x` is the full
    (2*nb,) data-space cartesian vector, identical to the representation used
    by `SELossCart` / `SELossGNCart`.

    Data-space (not normalized-space) is the primal on purpose: the LM
    optimizer's linear model `J @ step ≈ Δres` only matches the actual update
    when `update_x` is a simple affine step + projection. The previous version
    did `add step → un-normalize → project → re-normalize`, which is a
    nonlinear round-trip and broke LM's trust-region acceptance criterion.

    Normalization to the flow's input space happens only inside
    `nf_prior_eps` / `nf_prior_log_det`, and `torch.func.jacrev` differentiates
    through it automatically.

    Note on the slack bus: the NF was trained on data with the slack bus
    dropped entirely (see `load_data` in nf_model.py), so both the real and
    imaginary slack coordinates are removed by `B_real @ B_imag` before the
    flow sees the state. The LM step, however, lives in the standard reduced
    space (one slack coordinate dropped, consistent with SELossCart).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flow_model, self.scaler, self.with_log_det = load_nf_model_prior(**kwargs)

    def x_init(self, x):
        # x0 = torch.randn(10000, 2, 117)
        # # x_non_slack = self.B_real @ self.B_imag @ x0
        # # x_chan = x_non_slack.reshape(2, self.nb - 1)
        # x_new, _ = self.flow_model.decode(x0)
        # x_new = x_new.mean(dim=0)
        # x_enc = self._to_nf_input(self.encode(x))
        # x_new = self.scaler.inverse(((x_new + x_enc) / 2.).squeeze(0)).reshape(-1)
        # x_new = self.B_imag.T @ self.B_real.T @ x_new
        # x_new = x_new + self.slk_mag * self.u
        # return self.decode(x_new)
        return x.clone()

    def _to_nf_input(self, x):
        x_non_slack = self.B_real @ self.B_imag @ x
        x_channels = x_non_slack.reshape(2, -1)
        x_norm = self.scaler(x_channels)
        return x_norm.unsqueeze(0)

    def nf_prior_eps(self, x):
        return self.flow_model.inverse(self._to_nf_input(x)).reshape(-1)

    def nf_prior_log_det(self, x):
        return self.flow_model.log_det_inv_jacobian(self._to_nf_input(x)).squeeze(0)

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step)
        res_h = self.se_res(x)
        res_prior = self.nf_prior_eps(x)
        return torch.cat([res_h, torch.sqrt(self.prior_scale) * res_prior], dim=0)

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)
        res_prior = self.nf_prior_eps(x)
        f_prior = 0.5 * torch.norm(res_prior).pow(2)
        if self.with_log_det:
            f_prior += self.nf_prior_log_det(x)
        return f_h + self.prior_scale * f_prior

    def compute_J(self, x, step=None):
        x = self.update_x(x, step)
        J_h = self.se_jacobian(x, remove_slack=False)
        J_prior = jacrev(self.nf_prior_eps)(x)
        J_full = torch.cat([J_h, torch.sqrt(self.prior_scale) * J_prior], dim=0)

        return self._remove_slack(J_full, dim=1)

    def compute_grad(self, x, step=None):
        x = self.update_x(x, step)
        if self.with_log_det:
            grad_log_det = jacrev(self.nf_prior_log_det)(x)
        else:
            grad_log_det = torch.zeros_like(x)
        return self.prior_scale * self._remove_slack(grad_log_det)

    def likelihood_loss(self, x):
        return self.se_f(x)

    def prior_loss(self, x):
        res_prior = self.nf_prior_eps(x)
        res_prior = 0.5 * torch.norm(res_prior).pow(2)
        if self.with_log_det:
            res_prior += self.nf_prior_log_det(x)
        return res_prior


class SELossCartNFNorm(SELossCart):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flow_model, self.scaler, self.with_log_det = load_nf_model_prior(**kwargs)

    def x_init(self, x):
        return x.clone()

    def reshape_x(self, x_norm):
        return x_norm

    def _encode(self, x):
        x_non_slack = self.B_real @ self.B_imag @ x        # (2*(nb-1),)
        x_chan = x_non_slack.reshape(2, self.nb - 1)        # (2, nb-1)
        x_norm = self.scaler(x_chan)                        # (2, nb-1)
        return x_norm.reshape(-1)

    def encode(self, x):
        T, V = x[:self.nb], x[self.nb:]
        xc = torch.polar(V, T)
        x_r, x_i = xc.real, xc.imag
        x = torch.cat([x_r, x_i])
        x = self.project_x(x)
        return self._encode(x)

    def _decode(self, x_norm):
        x_unnorm = self.scaler.inverse(x_norm.reshape(2, self.nb - 1))                   # (1, 2, nb-1)
        x_non_slack = x_unnorm.reshape(-1)
        x_full = self.B_imag.T @ self.B_real.T @ x_non_slack
        x = x_full + self.slk_mag * self.u
        return x

    def decode(self, x_norm):
        x = self._decode(x_norm)
        x = self.project_x(x)
        x_r, x_i = x[:self.nb], x[self.nb:]
        xc = x_r + 1j * x_i
        T, V = xc.angle(), xc.abs()
        x = torch.cat([T, V])
        return x

    def update_x(self, x, step):
        if step is not None:
            x = x + step
        return x

    def _to_nf_input(self, x_norm):
        return x_norm.reshape(2, self.nb - 1).unsqueeze(0)

    def nf_prior_eps(self, x_norm):
        return self.flow_model.inverse(self._to_nf_input(x_norm)).reshape(-1)

    def nf_prior_log_det(self, x_norm):
        return self.flow_model.log_det_inv_jacobian(self._to_nf_input(x_norm)).squeeze(0)

    def compute_residuals(self, x_norm, step=None):
        x_norm = self.update_x(x_norm, step)
        x = self._decode(x_norm)
        res_h = self.se_res(x)
        res_prior = self.nf_prior_eps(x_norm)
        return torch.cat([res_h, torch.sqrt(self.prior_scale) * res_prior], dim=0)

    def compute_f(self, x_norm, step=None):
        x_norm = self.update_x(x_norm, step)
        x = self._decode(x_norm)
        f_h = self.se_f(x)
        res_prior = self.nf_prior_eps(x_norm)
        f_prior = 0.5 * torch.norm(res_prior).pow(2)
        if self.with_log_det:
            f_prior = f_prior + self.nf_prior_log_det(x_norm)
        return f_h + self.prior_scale * f_prior

    def compute_J(self, x_norm, step=None):
        x_norm = self.update_x(x_norm, step)
        x = self._decode(x_norm)
        J_h = self.se_jacobian(x, remove_slack=False)
        J_s = jacrev(self._decode)(x_norm)
        J_h = J_h @ J_s
        J_prior = jacrev(self.nf_prior_eps)(x_norm)
        J_full = torch.cat([J_h, torch.sqrt(self.prior_scale) * J_prior], dim=0)

        return J_full

    def compute_grad(self, x_norm, step=None):
        if self.with_log_det:
            grad_log_det = jacrev(self.nf_prior_log_det)(x_norm)
            return self.prior_scale * grad_log_det
        return torch.zeros_like(x_norm).to(torch.get_default_dtype())

    def likelihood_loss(self, x_norm):
        x = self._decode(x_norm)
        return self.se_f(x)

    def prior_loss(self, x_norm):
        res_prior = self.nf_prior_eps(x_norm)
        res_prior = 0.5 * torch.norm(res_prior).pow(2)
        if self.with_log_det:
            res_prior += self.nf_prior_log_det(x_norm).item()
        return res_prior


class SELossCartNFLat(SELossCartPriorLat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flow_model, self.scaler, self.with_log_det = load_nf_model_prior(**kwargs)
        self._iter = 0
        self.t_max_iter = int(kwargs.get('t_max_iter',
                                         kwargs.get('max_iter', 100)))
        self.t_schedule = kwargs.get('t_schedule', 'linear')
        self.t_min = float(kwargs.get('t_min', 0.0))
        self.t_max = float(kwargs.get('t_max', 1.0))

    def set_iter(self, k):
        """Tell the loss what optimizer iteration is starting. Called
        once per LM outer iteration. Subsequent `compute_*` calls use
        the corresponding `t = get_t()`."""
        self._iter = int(k)

    @property
    def iter(self):
        return self._iter

    def get_t(self) -> float:
        """Mix weight `t in [t_min, t_max]` at the current iteration."""
        if callable(self.t_schedule):
            t = float(self.t_schedule(self._iter, self.t_max_iter))
        else:
            denom = max(1, self.t_max_iter)
            u = self._iter / denom
            u = max(0.0, min(1.0, u))
            sched = (self.t_schedule.lower()
                     if isinstance(self.t_schedule, str) else 'linear')
            if sched == 'cosine':
                import math as _m
                t = 0.5 * (1.0 - _m.cos(_m.pi * u))
            elif sched == 'ld_only':
                t = 1.0
            elif sched == 'no_ld_only':
                t = 0.0
            else:                       # 'linear' default
                t = u
        return max(self.t_min, min(self.t_max, float(t)))

    @property
    def _eps_dim(self):
        return 2 * (self.nb - 1)

    def _eps_to_grid(self, eps):
        return eps.reshape(2, self.nb - 1).unsqueeze(0)

    def nf_prior_log_det(self, eps):
        # t = self.get_t()
        ldj = self.flow_model.log_det_jacobian(self._eps_to_grid(eps))

        return ldj

    def prior_f(self, eps):
        f_prior = 0.5 * torch.norm(eps).pow(2)
        if self.with_log_det:
            f_prior = f_prior + self.nf_prior_log_det(eps)
        return self.prior_scale * f_prior

    def prior_res(self, eps):
        return torch.sqrt(self.prior_scale) * eps

    def prior_jacobian(self, eps, remove_slack=True):
        J_prior = torch.eye(self._eps_dim)
        return torch.sqrt(self.prior_scale) * J_prior

    def prior_grad(self, eps, remove_slack=True):
        if self.with_log_det:
            grad_log_det = jacrev(self.nf_prior_log_det)(eps).reshape(-1)
            return self.prior_scale * grad_log_det
        return torch.zeros(self._eps_dim).to(torch.get_default_dtype())

    def _encode(self, x):
        x_non_slack = self.B_real @ self.B_imag @ x        # (2*(nb-1),)
        x_chan = x_non_slack.reshape(2, self.nb - 1)        # (2, nb-1)
        x_norm = self.scaler(x_chan)                        # (2, nb-1)
        eps = self.flow_model.inverse(x_norm.unsqueeze(0))  # (1, 2, nb-1)
        return eps.reshape(-1)

    def _decode(self, eps):
        x_norm, _ = self.flow_model.decode(self._eps_to_grid(eps))
        x_unnorm = self.scaler.inverse(x_norm.squeeze(0))
        x_non_slack = x_unnorm.reshape(-1)
        x_full = self.B_imag.T @ self.B_real.T @ x_non_slack
        x = x_full + self.slk_mag * self.u
        return x


class SELossGNCartLat(SELossCartPriorLat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['cart'] = True
        self.m_red, self.Q_red, self.L, self.L_inv = load_gaussian_model_prior(**kwargs)
        self.with_log_det = False

    def _decode(self, eps):
        x_red = (self.L @ eps) + self.m_red
        x = self.B_imag.T @ x_red
        x = self.not_slk * x + self.slk_mag * self.u
        return x

    def _encode(self, x):
        x_red = self._remove_slack(x)
        eps = self.L_inv @ (x_red - self.m_red)
        return eps

    def prior_res(self, eps):
        return torch.sqrt(self.prior_scale) * eps

    def prior_f(self, eps):
        res = self.prior_res(eps)
        return self.prior_scale * 0.5 * torch.norm(res).pow(2)

    def prior_jacobian(self, eps, remove_slack=True):
        J_prior = torch.eye(eps.shape[0])
        return torch.sqrt(self.prior_scale) * J_prior

    def prior_grad(self, eps, remove_slack=True):
        return torch.zeros_like(eps).to(torch.get_default_dtype())


class SELossGNCart(SELossCartPrior):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['cart'] = True
        self.m_red, self.Q_red, _, self.L = load_gaussian_model_prior(**kwargs)

    def prior_f(self, x):
        prior_res = self.prior_res(x)
        prior_f = .5 * torch.norm(prior_res).pow(2)
        return prior_f

    def prior_res(self, x):
        x_red = self._remove_slack(x)
        res_prior = self.L @ (x_red - self.m_red)
        return torch.sqrt(self.prior_scale) * res_prior

    def prior_jacobian(self, x, remove_slack=True):
        return torch.sqrt(self.prior_scale) * self.L

    def prior_grad(self, x, remove_slack=True):
        return self._remove_slack(torch.zeros_like(x))


class SELossCartNFGS(SELossCartNFLat):
    """Latent-space combined NF + Gaussian prior, with iteration-scheduled mixing.

    The primal variable is the flow's latent `eps in R^{2(nb-1)}` (same
    convention as `SELossCartNFLat`). The total objective is

        f(eps) = 0.5 ||res_h(x(eps))||^2
               + lamda * (     t * f_NF(eps)
                          + (1-t) * f_GS(x(eps)) )

    with

        x(eps)      = decode(eps)             (NF latent -> cartesian)
        f_NF(eps)   = 0.5 ||eps||^2  [+ log|det J_f(eps)|  if `with_log_det`]
        f_GS(x_red) = 0.5 ||L (x_red - m_red)||^2

    Two priors at the same time: the proper NF density on the latent
    (with optional change-of-variables term) and a Gaussian on the
    reduced cartesian state, blended by `t in [0, 1]`.

    The default schedule starts at `t = 0` (full Gaussian prior, very
    well-behaved) and ramps linearly to `t = 1` (full NF prior) by
    iteration `t_max_iter`. The intuition is to use the smooth
    Gaussian to pull the iterate into the typical operating region
    first, then let the NF's local geometry take over once we're
    close enough for its density estimate to be informative.

    Iteration tracking
    ------------------
    The loss exposes :meth:`set_iter(k)`. The shipped LM optimizers
    (`LMOpt`, `LMOptLogDet`) call it automatically once per outer
    iteration. If the optimizer doesn't, the schedule stays at
    `t = 0` (full GS) — explicit external control is then the way to
    advance it.

    Use with the optimizer's ``latent='ss'`` flag so `process_x`
    encodes the warm-start cartesian state into eps:

        LMOpt(loss_func=SELossCartNFGS(...), latent='ss', ...)

    Constructor kwargs (beyond the usual SELoss / SELossCart ones)
    --------------------------------------------------------------
    with_log_det    : add log|det J_f(eps)| to the NF block (read from
                      the NF model config by `load_nf_model_prior`).
    t_schedule      : "linear" (default), "cosine", "nf_only",
                      "gs_only", or a callable `f(iter, max_iter) -> t`.
    t_max_iter      : iteration at which `t` reaches `t_max` (default 100).
    t_min, t_max    : clamp the schedule output (default 0.0, 1.0).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        gs_kwargs = dict(kwargs)
        gs_kwargs['cart'] = True
        self.m_red, self.Q_red, _, self.L = load_gaussian_model_prior(**gs_kwargs)

        # Iteration tracking + mixing schedule.
        self._iter = 0
        self.t_max_iter = int(kwargs.get('t_max_iter',
                                         kwargs.get('max_iter', 100)))
        self.t_schedule = kwargs.get('t_schedule', 'linear')
        self.t_min = float(kwargs.get('t_min', 0.0))
        self.t_max = float(kwargs.get('t_max', 1.0))

    # --- iteration / schedule -----------------------------------------
    def set_iter(self, k):
        """Tell the loss what optimizer iteration is starting. Called
        once per LM outer iteration. Subsequent `compute_*` calls use
        the corresponding `t = get_t()`."""
        self._iter = int(k)

    @property
    def iter(self):
        return self._iter

    def get_t(self) -> float:
        """Mix weight `t in [t_min, t_max]` at the current iteration."""
        if callable(self.t_schedule):
            t = float(self.t_schedule(self._iter, self.t_max_iter))
        else:
            denom = max(1, self.t_max_iter)
            u = self._iter / denom
            u = max(0.0, min(1.0, u))
            sched = (self.t_schedule.lower()
                     if isinstance(self.t_schedule, str) else 'linear')
            if sched == 'cosine':
                import math as _m
                t = 0.5 * (1.0 - _m.cos(_m.pi * u))
            elif sched == 'nf_only':
                t = 1.0
            elif sched == 'gs_only':
                t = 0.0
            else:                       # 'linear' default
                t = u
        return max(self.t_min, min(self.t_max, float(t)))

    def _sqrt_w(self, w: float):
        """sqrt(lamda * max(0, w)) as a scalar tensor."""
        w_clamped = max(0.0, float(w))
        return torch.sqrt(self.prior_scale * torch.as_tensor(
            w_clamped, dtype=torch.get_default_dtype()))

    def prior_f(self, eps):
        x = self._decode(eps)
        t = self.get_t()

        f_nf = 0.5 * torch.norm(eps).pow(2)
        if self.with_log_det:
            f_nf = f_nf + self.nf_prior_log_det(eps)

        x_red = self._remove_slack(x)
        res_gs = self.L @ (x_red - self.m_red)
        f_gs = 0.5 * torch.norm(res_gs).pow(2)

        return self.prior_scale * (t * f_nf + (1.0 - t) * f_gs)

    def prior_res(self, eps):
        x = self._decode(eps)
        t = self.get_t()
        res_nf = eps
        x_red = self._remove_slack(x)
        res_gs = self.L @ (x_red - self.m_red)
        res_prior = torch.cat([self._sqrt_w(t) * res_nf, self._sqrt_w(1.0 - t) * res_gs], dim=0)
        return res_prior

    def prior_jacobian(self, eps, remove_slack=True):
        t = self.get_t()

        J_decode = jacrev(self._decode)(eps)
        J_gs = self.L @ (self.B @ J_decode)
        J_nf = torch.eye(self._eps_dim)

        prior_J = torch.cat([self._sqrt_w(t) * J_nf, self._sqrt_w(1.0 - t) * J_gs], dim=0)
        return prior_J


class SELossFlowerCart(SELossCart):
    def __init__(self, x_hat_fm, nu_t, **kwargs):
        super().__init__(**kwargs)
        kwargs['cart'] = True
        self.m = x_hat_fm
        self.L = (1. / nu_t) * torch.eye(x_hat_fm.shape[0])

    def x_init(self, x):
        # new_x = self.decode(self._insert_slack_angle(self.m_red))
        return x

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step)
        res_h = self.se_res(x)

        res_prior = self.L @ (x - self.m)
        res_prior = self._remove_slack(res_prior)
        res = torch.cat([res_h, torch.sqrt(self.prior_scale) * res_prior], dim=0)

        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)

        res_prior = self.L @ (x - self.m)
        res_prior = self._remove_slack(res_prior)
        f_prior = .5 * torch.norm(res_prior).pow(2)
        f = f_h + (self.prior_scale * f_prior)
        return f

    def compute_J(self, x, step=None):
        x = self.update_x(x, step)
        J_h = self.se_jacobian(x)
        J_prior = self._remove_slack(self.L, dim=1)
        J_prior = self._remove_slack(J_prior)
        J = torch.cat([J_h, torch.sqrt(self.prior_scale) * J_prior], dim=0)
        return J


    def prior_loss(self, x):
        res_prior = self.L @ (x - self.m)
        res_prior = self._remove_slack(res_prior)
        f_prior = .5 * torch.norm(res_prior).pow(2)
        return f_prior


class SELossFM(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_FM, self.std_FM, self.flow_model = load_FM_prior(**kwargs)

    def fm_prior_eps(self, x_red):
        x_norm = ((x_red - self.m_FM) / self.std_FM).unsqueeze(0)
        eps = self.flow_model.run_flow_matching(x_norm, inverse=True).squeeze(0)
        return eps

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step)
        res_h = self.se_res(x)

        # x_red = self._remove_slack(x)
        res_prior = self.fm_prior_eps(x)

        res = torch.cat([res_h, torch.sqrt(self.prior_scale) * res_prior], dim=0)

        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)
        # x_red = self._remove_slack(x)
        res_prior = self.fm_prior_eps(x)
        f_prior = .5 * torch.norm(res_prior).pow(2)
        f = f_h + (self.prior_scale * f_prior)

        return f

    def compute_J(self, x, step=None):
        x = self.update_x(x, step)
        J_h = self.se_jacobian(x)
        # x_red = self._remove_slack(x)
        J_prior = jacfwd(self.fm_prior_eps)(x)
        J_prior = self._remove_slack(J_prior, dim=1)
        J = torch.cat([J_h, torch.sqrt(self.prior_scale) * J_prior], dim=0)

        return J


class SELossFMLat(SELossFM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.with_log_det = False

    def reshape_x(self, x):
        return x

    def decode(self, eps):
        x_norm = self.flow_model.run_flow_matching(eps.unsqueeze(0)).squeeze(0)
        x = ((x_norm * self.std_FM) + self.m_FM)
        x = self._insert_slack_angle_linear(x)
        return x

    def encode(self, x):
        x_red = self._remove_slack(x)
        x_red = self.m_FM
        x_norm = ((x_red - self.m_FM) / self.std_FM).unsqueeze(0)
        eps = self.flow_model.run_flow_matching(x_norm, inverse=True).squeeze(0)
        return eps

    def compute_composition(self, eps):
        x = self.decode(eps)
        z_est = self.h_ac.estimate(x)
        return z_est

    def compute_residuals(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self.decode(eps)
        res_h = self.se_res(x)
        res =  torch.cat([res_h, torch.sqrt(self.prior_scale) * eps], dim=0)

        return res

    def compute_f(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self.decode(eps)
        f_h = self.se_f(x)
        f_prior = .5 * torch.norm(eps).pow(2)
        f = f_h + f_prior
        return f

    def compute_J(self, eps, step=None):
        eps = self.update_x(eps, step)
        J_f = jacfwd(self.decode)(eps)

        x = self.decode(eps)
        J_h = self.h_ac.jacobian(x)
        J_h_f = -(self.R @ J_h @ J_f)

        J_prior = torch.eye((self.nb * 2) - 1)
        J = torch.cat([J_h_f, torch.sqrt(self.prior_scale) * J_prior], dim=0)

        return J


class SELossPCAGaussian(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        (self.pca_data_mean,
         self.pca_components,
         self.pca_latent_mean,
         self.pca_latent_cov,
         self.pca_cov_factor,
         self.pca_precision_factor) = load_PCA_Gaussian_prior(**kwargs)

    def pca_latent(self, x_red):
        return self.pca_components @ (x_red - self.pca_data_mean)

    def pca_prior_residual(self, x_red):
        z = self.pca_latent(x_red)
        return self.pca_precision_factor @ (z - self.pca_latent_mean)

    def pca_prior_jacobian(self):
        return self.pca_precision_factor @ self.pca_components

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step)
        res_h = self.se_res(x)
        x_red = self._remove_slack(x)
        res_prior = self.pca_prior_residual(x_red)
        res = torch.cat([res_h, torch.sqrt(self.prior_scale) * res_prior], dim=0)
        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)
        x_red = self._remove_slack(x)
        res_prior = self.pca_prior_residual(x_red)
        f_prior = 0.5 * torch.norm(res_prior).pow(2)
        f = f_h + (self.prior_scale * f_prior)
        return f

    def compute_J(self, x, step=None):
        x = self.update_x(x, step)
        J_h = self.se_jacobian(x)
        J_prior = self.pca_prior_jacobian()
        J = torch.cat([J_h, torch.sqrt(self.prior_scale) * J_prior], dim=0)
        return J
