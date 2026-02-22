import json
import math
from abc import ABC
import math as mt
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal
import torch.nn.functional as F
from torch.func import jacrev, vmap
from tqdm import tqdm

from SE_torch.learn_prior.NF.NF import FlowModel
from SE_torch.learn_prior.NF.NF2 import FlowModel as FlowModel2

from SE_torch.learn_prior.FM.FM import FlowMatching
from SE_torch.learn_prior.VAE.VAE import AmortizedVAE
from SE_torch.learn_prior.GAN.GAN import Generator
from SE_torch.optimizers.base_optimizer import SEOptimizer
from SE_torch.learn_prior.NF.NF import DATA_DIM

DEFAULT_NB = 118
SCALE_V = 0.025
SCALE_T = mt.sqrt(mt.pi) / 2


def load_GN_prior(**kwargs):
    nb = kwargs.get('nb', DEFAULT_NB)
    m = kwargs.get('m', torch.concat([torch.zeros(nb),
                                      torch.ones(nb)])).to(torch.get_default_dtype())
    Q = kwargs.get('Q', torch.diag(
        torch.concat([torch.ones(nb) * SCALE_T,
                      torch.ones(nb) * SCALE_V]))).to(torch.get_default_dtype())
    slk_idx = kwargs.get('slk_idx', 0)

    if len(m) == nb * 2 - 1:
        m_red = m
        Q_red = Q
    else:
        mask = torch.ones_like(m, dtype=torch.bool, device=m.device)
        mask[slk_idx] = False

        m_red = m[mask]
        Q_red = Q[mask][:, mask]

    Q_inv = torch.linalg.pinv(Q_red)
    L = torch.linalg.cholesky(Q_red)
    L = torch.linalg.pinv(L)

    return m_red, Q_red, Q_inv, L


def load_NF_prior(**kwargs):
    with_log_det = kwargs.get('with_log_det', False)
    NF_config_path = kwargs['NF_config_path']
    NF_config = json.load(open(NF_config_path))
    NF_config['device'] = 'cpu'
    ckpt_path = f"../learn_prior/NF/models/{NF_config.get('ckpt_name')}"
    flow_model = FlowModel(**NF_config).to(device='cpu')
    flow_model.load_state_dict(torch.load(ckpt_path))
    flow_model.eval()
    m_NF = torch.load("../learn_prior/datasets/mean_polar.pt").to(torch.get_default_dtype())
    std_NF = torch.load("../learn_prior/datasets/std_polar.pt").to(torch.get_default_dtype())

    return m_NF, std_NF, flow_model, with_log_det


def load_NF2_prior(**kwargs):
    with_log_det = kwargs.get('with_log_det', False)
    NF_config_path = kwargs['NF_config_path']
    NF_config = json.load(open(NF_config_path))
    NF_config['device'] = 'cpu'
    ckpt_path = f"../learn_prior/NF/models/{NF_config.get('ckpt_name')}"
    flow_model = FlowModel2(**NF_config).to(device='cpu')
    flow_model.load_state_dict(torch.load(ckpt_path))
    flow_model.eval()
    m_NF = torch.load("../learn_prior/datasets/mean_NF_polar.pt").to(torch.get_default_dtype())
    std_NF = torch.load("../learn_prior/datasets/std_NF_polar.pt").to(torch.get_default_dtype())

    return m_NF, std_NF, flow_model, with_log_det


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


def load_VAE_prior(**kwargs):
    VAE_config_path = kwargs['VAE_config_path']
    VAE_config = json.load(open(VAE_config_path))
    ckpt_path = f"../learn_prior/VAE/models/{VAE_config.get('ckpt_name')}"
    vae_model = AmortizedVAE(**VAE_config).to(device='cpu')
    vae_model.load_state_dict(torch.load(ckpt_path))
    vae_model.eval()
    m_VAE = torch.load("../learn_prior/datasets/mean_polar.pt").to(torch.get_default_dtype())
    std_VAE = torch.load("../learn_prior/datasets/std_polar.pt").to(torch.get_default_dtype())
    num_samples = VAE_config['num_samples']
    sigma_p = VAE_config['sigma_p']
    return m_VAE, std_VAE, vae_model, num_samples, sigma_p


def load_GAN_prior(**kwargs):
    GAN_config_path = kwargs['GAN_config_path']
    GAN_config = json.load(open(GAN_config_path))
    ckpt_path = f"../learn_prior/GAN/models/{GAN_config.get('ckpt_name')}_G_e30.pt"
    gan_model = Generator(**GAN_config).to(device='cpu')
    gan_model.load_state_dict(torch.load(ckpt_path))
    gan_model.eval()
    m_GAN = torch.load("../learn_prior/datasets/mean_polar.pt").to(torch.get_default_dtype())
    std_GAN = torch.load("../learn_prior/datasets/std_polar.pt").to(torch.get_default_dtype())
    num_samples = GAN_config['num_samples']
    sigma_p = GAN_config['sigma_p']
    return m_GAN, std_GAN, gan_model, num_samples, sigma_p


class LossFunction(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

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


class SELoss(LossFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.z = kwargs.get('z')
        self.v = kwargs.get('v')
        self.R = kwargs.get('R')
        self.h_ac = kwargs.get('h_ac')
        self.slk_bus = kwargs.get('slk_bus')
        self.nb = kwargs.get('nb')
        self.norm_H = kwargs.get('norm_H')
        self.lamda = kwargs.get('prior_scale', 1.)

    def update_params(self, *args):
        self.z = args[0]
        self.v = args[1]
        self.slk_bus = args[2]
        self.h_ac = args[3]
        self.nb = args[4]
        self.R = torch.diag(1. / torch.sqrt(self.v))
        self.norm_H = args[5] if args[5] is not None else torch.ones_like(self.z)

    def reshape_x(self, x):
        return self._remove_slack(x)

    def _split_x(self, x):
        T = x[:self.nb]
        V = x[self.nb:]
        return T, V

    def _remove_slack(self, x, dim=0):
        s = int(self.slk_bus[0])
        if dim == 0:
            return torch.cat([x[:s], x[s + 1:]], dim=0)
        else:
            return torch.cat([x[:, :s], x[:, s + 1:]], dim=1)

    def _insert_zero_at_slack(self, vec, dim=0):
        s = int(self.slk_bus[0])
        if dim == 0:
            return torch.cat([vec[:s], torch.zeros(1), vec[s:]], dim=0)
        else:
            return torch.cat([vec[:, :s], torch.zeros(vec.shape[0], 1), vec[:, s + 1:]], dim=1)

    def _build_insert_slack_angle_operator(self, n_full: int):
        s = int(self.slk_bus[0])
        A = torch.zeros((n_full, n_full - 1))
        if s > 0:
            A[0:s, 0:s] = torch.eye(s)
        if s < n_full - 1:
            A[s + 1:, s:] = torch.eye(n_full - s - 1)
        b = torch.zeros((n_full,))
        b[s] = torch.as_tensor(self.slk_bus[1], dtype=torch.get_default_dtype())
        return A, b

    def _insert_slack_angle_linear(self, vec):
        n_full = vec.shape[0] + 1
        A, b = self._build_insert_slack_angle_operator(n_full)
        return A @ vec + b

    def _insert_slack_angle(self, vec):
        s = int(self.slk_bus[0])
        angle = torch.tensor([self.slk_bus[1]]).to(dtype=torch.get_default_dtype())
        return torch.cat([vec[:s], angle, vec[s:]], dim=0)

    def update_x(self, x, step):
        if step is not None:
            if len(step) < len(x):
                step = self._insert_zero_at_slack(step)
            x = x + step
        return x

    def se_res(self, x):
        z_est = self.h_ac.estimate(x)
        res = (self.R @ (self.z - z_est))
        return res

    def se_jacobian(self, x):
        J = self.h_ac.jacobian(x)
        J = -(self.R @ J)
        J = self._remove_slack(J, dim=1)
        return J

    def se_f(self, x):
        z_est = self.h_ac.estimate(x).detach()
        res_h = self.R @ (self.z - z_est) / self.norm_H
        f_h = .5 * torch.norm(res_h).pow(2)
        return f_h

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step).detach()
        res = self.se_res(x)
        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)
        return f_h

    def compute_J(self, x, step=None):
        x = self.update_x(x, step).detach()
        J = self.se_jacobian(x)
        return J

    def compute_grad(self, x, step=None):
        return torch.zeros(self.nb * 2 - 1)


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
        self.z = args[0]
        self.v = args[1]
        self.R = torch.pinverse(torch.linalg.cholesky(torch.diag(self.v)))
        self.slk_bus = args[2]
        self.h_ac = args[3]
        self.nb = args[4]
        self.norm_H = args[5] if args[5] is not None else torch.ones_like(self.z)

        self.slk_idx_real = self.slk_bus[0]
        self.slk_idx_imag = self.slk_bus[0] + self.nb

        self.slk_angle = torch.tensor([self.slk_bus[1]])
        self.u = torch.tensor([torch.cos(self.slk_angle), torch.sin(self.slk_angle)])
        self.non_slk_real = torch.concat([torch.arange(self.slk_idx_real),
                                          torch.arange(self.slk_idx_real + 1, self.nb)])
        self.non_slk_imag = torch.concat([torch.arange(self.nb - 1, self.slk_idx_imag - 1),
                                          torch.arange(self.slk_idx_imag, self.nb * 2 - 1)])


    def project_x(self, x):
        x[[self.slk_idx_real, self.slk_idx_imag]] = x[[self.slk_idx_real, self.slk_idx_imag]].dot(self.u) * self.u
        return x

    def decode(self, x):
        x = self.project_x(x)
        x_r, x_i = x[:self.nb], x[self.nb:]
        xc = x_r + 1j * x_i
        T, V = xc.angle(), xc.abs()
        x_polar = torch.cat([T, V])
        return x_polar

    def encode(self, x):
        T, V = x[:self.nb], x[self.nb:]
        xc = V * torch.exp(1j * T)
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


class SELossGN(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_red, self.Q_red, self.Q_inv, self.L = load_GN_prior(**kwargs)

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step).detach()
        res_h = self.se_res(x)

        x_red = self._remove_slack(x)
        res_prior = self.L @ (x_red - self.m_red)
        res = torch.cat([res_h, mt.sqrt(self.lamda) * res_prior], dim=0)

        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)

        x_red = self._remove_slack(x)
        res_prior = self.L @ (x_red - self.m_red)
        f_prior = .5 * torch.norm(res_prior).pow(2)
        f = f_h + (self.lamda * f_prior)
        return f

    def compute_J(self, x, step=None):
        x = self.update_x(x, step).detach()
        J_h = self.se_jacobian(x)
        J = torch.cat([J_h, mt.sqrt(self.lamda) * self.L], dim=0)
        return J


class SELossNF(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_NF, self.std_NF, self.flow_model, self.with_log_det = load_NF_prior(**kwargs)

    def nf_prior_eps(self, x_red):
        x_norm = ((x_red - self.m_NF) / self.std_NF).unsqueeze(0)
        eps = self.flow_model.inverse(x_norm).squeeze(0)
        return eps

    def nf_prior_log_det(self, x_red):
        x_norm = ((x_red - self.m_NF) / self.std_NF).unsqueeze(0)
        log_det = self.flow_model.log_det_inv_jacobian(x_norm).squeeze(0)
        return log_det

    def compute_grad(self, x, step=None):
        if self.with_log_det:
            x_red = self._remove_slack(x).detach()
            grad_log_det = jacrev(self.nf_prior_log_det)(x_red)
            return - self.lamda * grad_log_det

        return torch.zeros(self.nb * 2 - 1)

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step)
        res_h = self.se_res(x)

        x_red = self._remove_slack(x)
        res_prior = self.nf_prior_eps(x_red)

        res = torch.cat([res_h, mt.sqrt(self.lamda) * res_prior], dim=0)

        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)
        x_red = self._remove_slack(x)
        res_prior = self.nf_prior_eps(x_red)
        f_prior = .5 * torch.norm(res_prior).pow(2)
        if self.with_log_det:
            log_det = self.nf_prior_log_det(x_red)
            f_prior -= log_det

        f = f_h + (self.lamda * f_prior)

        return f

    def compute_J(self, x, step=None):
        x = self.update_x(x, step)
        J_h = self.se_jacobian(x)
        x_red = self._remove_slack(x)
        J_prior = jacrev(self.nf_prior_eps)(x_red)
        J = torch.cat([J_h, mt.sqrt(self.lamda) * J_prior], dim=0)

        return J


class SELossNF2(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_NF, self.std_NF, self.flow_model, self.with_log_det = load_NF2_prior(**kwargs)

    def nf_prior_eps(self, x_red):
        x_norm = ((x_red - self.m_NF) / self.std_NF).unsqueeze(0)
        eps = self.flow_model.inverse(x_norm).squeeze(0)
        return eps

    def nf_prior_log_det(self, x_red):
        x_norm = ((x_red - self.m_NF) / self.std_NF).unsqueeze(0)
        log_det = self.flow_model.log_det_inv_jacobian(x_norm).squeeze(0)
        return log_det

    def compute_grad(self, x, step=None):
        if self.with_log_det:
            # x_red = self._remove_slack(x).detach()
            grad_log_det = jacrev(self.nf_prior_log_det)(x)
            grad_log_det = self._remove_slack(grad_log_det)
            return - self.lamda * grad_log_det

        return torch.zeros(self.nb * 2 - 1)

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step)
        res_h = self.se_res(x)

        res_prior = self.nf_prior_eps(x)
        res_prior = self._remove_slack(res_prior)

        res = torch.cat([res_h, mt.sqrt(self.lamda) * res_prior], dim=0)

        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)
        res_prior = self.nf_prior_eps(x)
        res_prior = self._remove_slack(res_prior)
        f_prior = .5 * torch.norm(res_prior).pow(2)
        if self.with_log_det:
            log_det = self.nf_prior_log_det(x)
            f_prior -= log_det

        f = f_h + (self.lamda * f_prior)

        return f

    def compute_J(self, x, step=None):
        x = self.update_x(x, step)
        J_h = self.se_jacobian(x)
        # x_red = self._remove_slack(x)
        J_prior = jacrev(self.nf_prior_eps)(x)
        J_prior = self._remove_slack(J_prior, dim=1)
        J_prior = self._remove_slack(J_prior, dim=0)
        J = torch.cat([J_h, mt.sqrt(self.lamda) * J_prior], dim=0)

        return J


class SELossNFLat(SELossNF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reshape_x(self, x):
        return x

    def nf_prior_log_det(self, eps):
        log_det = self.flow_model.log_det_jacobian(eps.unsqueeze(0)).squeeze(0)
        return log_det

    def nf_prior_log_det2(self, eps):
        x_norm = self.flow_model(eps.unsqueeze(0)).squeeze(0)
        log_det = self.flow_model.log_det_inv_jacobian(x_norm.unsqueeze(0)).squeeze(0)
        return -log_det

    def compute_grad(self, eps, step=None):
        if self.with_log_det:
            grad_log_det = jacrev(self.nf_prior_log_det)(eps)
            return self.lamda * grad_log_det

        return torch.zeros(self.nb * 2 - 1)

    def decode(self, eps):
        x_norm = self.flow_model(eps.unsqueeze(0)).squeeze(0)
        x = ((x_norm * self.std_NF) + self.m_NF)
        x = self._insert_slack_angle_linear(x)
        return x

    def encode(self, x):
        x_red = self._remove_slack(x)
        x_norm = ((x_red - self.m_NF) / self.std_NF).unsqueeze(0)
        eps = self.flow_model.inverse(x_norm).squeeze(0)
        return eps

    def compute_composition(self, eps):
        x = self.decode(eps)
        z_est = self.h_ac.estimate(x)
        return z_est

    def compute_residuals(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self.decode(eps)
        res_h = self.se_res(x)
        res =  torch.cat([res_h, mt.sqrt(self.lamda) * eps], dim=0)

        return res

    def compute_f(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self.decode(eps)
        f_h = self.se_f(x)
        f_prior = .5 * torch.norm(eps).pow(2)
        if self.with_log_det:
            log_det = self.nf_prior_log_det(eps)
            f_prior += log_det
        f = f_h + f_prior
        return f

    def compute_J(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self.decode(eps)
        J_h = self.h_ac.jacobian(x)
        J_f = jacrev(self.decode)(eps)
        J_h_f = -(self.R @ J_h @ J_f)
        J_prior = torch.eye((self.nb * 2 - 1))
        J = torch.cat([J_h_f, mt.sqrt(self.lamda) * J_prior], dim=0)

        return J


class SELossNF2Lat(SELossNF2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.with_log_det = False

    def reshape_x(self, x):
        return x

    def compute_grad(self, x, step=None):
        return torch.zeros_like(x)

    def decode(self, eps):
        x_norm = self.flow_model(eps.unsqueeze(0)).squeeze(0)
        x = ((x_norm * self.std_NF) + self.m_NF)
        # x = self._insert_slack_angle_linear(x)
        return x

    def encode(self, x):
        # x_red = self._remove_slack(x)
        x_norm = ((x - self.m_NF) / self.std_NF).unsqueeze(0)
        eps = self.flow_model.inverse(x_norm).squeeze(0)
        return eps

    def compute_composition(self, eps):
        x = self.decode(eps)
        z_est = self.h_ac.estimate(x)
        return z_est

    def compute_residuals(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self.decode(eps)
        res_h = self.se_res(x)
        res =  torch.cat([res_h, mt.sqrt(self.lamda) * eps], dim=0)

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
        x = self.decode(eps)
        J_h = self.h_ac.jacobian(x)
        J_f = jacrev(self.decode)(eps)
        J_h_f = -(self.R @ J_h @ J_f)
        # J_h_f = self._remove_slack(J_h_f, dim=1)
        J_prior = torch.eye((self.nb * 2))
        # J_prior = self._remove_slack(J_prior, dim=1)
        J = torch.cat([J_h_f, mt.sqrt(self.lamda) * J_prior], dim=0)

        return J


class SELossVAE(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_VAE, self.std_VAE, self.vae_model, self.num_samples, self.sigma_p = load_VAE_prior(**kwargs)

    def vae_decode(self, eps):
        x_norm = self.vae_model.decode(eps.unsqueeze(0)).squeeze(0)
        x = ((x_norm * self.std_VAE) + self.m_VAE)
        return x

    def vae_encode(self, x_red):
        x_norm = ((x_red - self.m_VAE) / self.std_VAE)
        mu, logvar= self.vae_model.encode(x_norm.unsqueeze(0))
        mu = mu.squeeze(0)
        logvar = logvar.squeeze(0).clamp(-20.0, 10.0)
        return mu, logvar

    def vae_prior_residual(self, x_red):
        mu, logvar = self.vae_encode(x_red)
        x_rec = self.vae_decode(mu)
        res_rec = (x_red - x_rec) / self.sigma_p
        return torch.cat([res_rec, mu], dim=0)

    def compute_residuals(self, x, step=None):
        x = self.update_x(x, step)
        res_h = self.se_res(x)
        x_red = self._remove_slack(x)
        res_prior = self.vae_prior_residual(x_red)
        res = torch.cat([res_h, mt.sqrt(self.lamda) * res_prior], dim=0)
        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)
        x_red = self._remove_slack(x)
        res_prior = self.vae_prior_residual(x_red)
        f_prior = 0.5 * torch.norm(res_prior).pow(2)
        f = f_h + self.lamda * f_prior

        return f

    def compute_J(self, x, step=None):
        x = self.update_x(x, step)
        J_h = self.se_jacobian(x)
        x_red = self._remove_slack(x)
        J_prior = jacrev(self.vae_prior_residual)(x_red)
        J = torch.cat([J_h, mt.sqrt(self.lamda) * J_prior], dim=0)

        return J


class SELossVAELat(SELossVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reshape_x(self, x):
        return x

    def decode(self, eps):
        x_norm = self.vae_model.decode(eps.unsqueeze(0)).squeeze(0)
        x = ((x_norm * self.std_VAE) + self.m_VAE)
        x = self._insert_slack_angle_linear(x)
        return x

    def encode(self, x):
        x_red = self._remove_slack(x)
        x_red = self.m_VAE
        x_norm = ((x_red - self.m_VAE) / self.std_VAE).unsqueeze(0)
        eps, _ = self.vae_model.encode(x_norm)
        return eps.squeeze(0)

    def compute_composition(self, eps):
        x = self.decode(eps)
        z_est = self.h_ac.estimate(x)
        return z_est

    def compute_residuals(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self.decode(eps)
        res_h = self.se_res(x)
        res =  torch.cat([res_h, mt.sqrt(self.lamda) * eps], dim=0)
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
        x = self.decode(eps)
        J_h = self.h_ac.jacobian(x)
        J_f = jacrev(self.decode)(eps)
        J_h_f = -(self.R @ J_h @ J_f)

        J_prior = torch.eye((self.nb * 2) - 1)
        J = torch.cat([J_h_f, mt.sqrt(self.lamda) * J_prior], dim=0)
        return J


class SELossGANLat(SELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_GAN, self.std_GAN, self.gan_model, self.num_samples, self.sigma_p = load_GAN_prior(**kwargs)

    def reshape_x(self, x):
        return x

    def decode(self, eps):
        x_norm = self.gan_model(eps.unsqueeze(0)).squeeze(0)
        x = ((x_norm * self.std_GAN) + self.m_GAN)
        x = self._insert_slack_angle(x)
        return x

    def encode(self, x):
        x_red = self._remove_slack(x)
        return torch.zeros_like(x_red)

    def compute_composition(self, eps):
        x = self.decode(eps)
        z_est = self.h_ac.estimate(x)
        return z_est

    def compute_residuals(self, eps, step=None):
        eps = self.update_x(eps, step)
        x = self.decode(eps)
        res_h = self.se_res(x)
        res =  torch.cat([res_h, mt.sqrt(self.lamda) * eps], dim=0)
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
        x = self.decode(eps)
        J_h = self.h_ac.jacobian(x)
        J_f = jacrev(self.decode)(eps)
        J_h_f = -(self.R @ J_h @ J_f)

        J_prior = torch.eye((self.nb * 2) - 1)
        J = torch.cat([J_h_f, mt.sqrt(self.lamda) * J_prior], dim=0)
        return J


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

        res = torch.cat([res_h, mt.sqrt(self.lamda) * res_prior], dim=0)

        return res

    def compute_f(self, x, step=None):
        x = self.update_x(x, step)
        f_h = self.se_f(x)
        # x_red = self._remove_slack(x)
        res_prior = self.fm_prior_eps(x)
        f_prior = .5 * torch.norm(res_prior).pow(2)
        f = f_h + (self.lamda * f_prior)

        return f

    def compute_J(self, x, step=None):
        x = self.update_x(x, step)
        J_h = self.se_jacobian(x)
        # x_red = self._remove_slack(x)
        J_prior = jacrev(self.fm_prior_eps)(x)
        J_prior = self._remove_slack(J_prior, dim=1)
        J = torch.cat([J_h, mt.sqrt(self.lamda) * J_prior], dim=0)

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
        res =  torch.cat([res_h, mt.sqrt(self.lamda) * eps], dim=0)

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
        J_f = jacrev(self.decode)(eps)

        x = self.decode(eps)
        J_h = self.h_ac.jacobian(x)
        J_h_f = -(self.R @ J_h @ J_f)

        J_prior = torch.eye((self.nb * 2) - 1)
        J = torch.cat([J_h_f, mt.sqrt(self.lamda) * J_prior], dim=0)

        return J