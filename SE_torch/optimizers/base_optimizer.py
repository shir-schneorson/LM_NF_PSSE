import json
import math

import torch
from torch import nn as nn
from torch.distributions import MultivariateNormal

from SE_torch.learn_prior.NF.NF import FlowModel
from SE_torch.learn_prior.NF.NF_cart import FlowModel as FlowModel_cart
from SE_torch.learn_prior.NF.load_data import DATA_DIM, CHANNELS


class SEOptimizer:
    def __init__(self, **kwargs):
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = kwargs.get('max_iter', 100)

    def __call__(self, x0, z, v, slk_bus, h_ac, nb):
        raise NotImplementedError


class WeightedSELoss(nn.Module):
    def __init__(self, z, v, norm_H=None):
        super().__init__()
        self.z = z
        self.R = torch.diag(1./ v)
        self.norm_H = norm_H if norm_H is not None else torch.ones_like(z)

    def forward(self, z_est, x):
        error = ((self.z - z_est) / self.norm_H).reshape(-1, 1)
        weighted_squared_error = (error.T @ self.R @ error)
        return weighted_squared_error


class WeightedSEWithPriorLoss(nn.Module):
    def __init__(self, z, v, slk_bus, norm_H=None, NF_config_path='../learn_prior/configs/NF_4_0_128.json'):
        super().__init__()
        self.z = z
        self.v = v
        self.R = torch.diag(1./ torch.sqrt(v))
        self.slk_bus = slk_bus
        self.norm_H = norm_H if norm_H is not None else torch.ones_like(z)
        NF_config =  json.load(open(NF_config_path))
        NF_config['device'] = 'cpu'
        ckpt_path = f"../learn_prior/NF/models/{NF_config.get('ckpt_name')}"
        self.flow_model = FlowModel(**NF_config)
        self.flow_model.load_state_dict(torch.load(ckpt_path))
        self.mean = torch.load("../learn_prior/datasets/mean_polar.pt").to(dtype=torch.get_default_dtype())
        self.std = torch.load("../learn_prior/datasets/std_polar.pt").to(dtype=torch.get_default_dtype())
        self.prior_dist = MultivariateNormal(torch.zeros(DATA_DIM * CHANNELS -1), torch.eye(DATA_DIM * CHANNELS -1))
        self.lh_dist = MultivariateNormal(self.z, torch.diag(self.v))

    def forward(self, z_est, x):
        self.flow_model.eval()
        x = torch.concat([x[:self.slk_bus], x[self.slk_bus + 1:]])
        x_norm = ((x - self.mean) /self.std).unsqueeze(0).to(device='cpu', non_blocking=False)
        eps = self.flow_model.inverse(x_norm).to(device='cpu').squeeze(0)
        prior = .5 * torch.norm(eps).pow(2)
        log_det = self.flow_model.log_det_inv_jacobian(x_norm).to(device='cpu')
        pl = prior - log_det
        res_h = (self.R @ (self.z - z_est)) / self.norm_H
        lh = .5 * torch.norm(res_h).pow(2)
        loss = lh + pl
        return loss

class WeightedSECartWithPriorLoss(nn.Module):
    def __init__(self, z, R, slk_bus, NF_config_path='../learn_prior/configs_cart/NF_2_0_8.json', norm_H=None):
        super().__init__()
        self.z = z
        self.R = R
        self.v = 1 / torch.diag(self.R)
        self.slk_bus = slk_bus
        NF_config =  json.load(open(NF_config_path))
        ckpt_path = f"../learn_prior/NF/models/{NF_config.get('ckpt_name')}"
        self.device = NF_config.get('device', 'mps')
        self.dtype = NF_config.get('dtype', torch.get_default_dtype())
        self.flow_model = FlowModel_cart(**NF_config)
        self.flow_model.load_state_dict(torch.load(ckpt_path))
        self.mean = torch.load("../learn_prior/datasets/mean_cart.pt").to(device=self.device, dtype=self.dtype)
        self.std = torch.load("../learn_prior/datasets/std_cart.pt").to(device=self.device, dtype=self.dtype)
        self.norm_H = norm_H if norm_H is not None else torch.ones_like(z)
        self.lamda_iter =  iter(torch.linspace(1, 1e-2, steps=50))

    def forward(self, z_est, x):
        self.flow_model.eval()
        x_red = torch.concat([x[:self.slk_bus], x[self.slk_bus + 1:]])
        x_norm = ((x_red - self.mean) /self.std).unsqueeze(0)
        z = self.flow_model.inverse(x_norm)
        log_prob = .5 * torch.norm(z).pow(2)
        log_det_inv_jacobian = self.flow_model.log_det_inv_jacobian(x_norm)
        loss_prior = - (log_prob + log_det_inv_jacobian)
        error = ((self.z - z_est) / self.norm_H).reshape(-1, 1)
        loss_meas = .5 * (error.T @ self.R @ error)
        loss = loss_meas + loss_prior
        return loss