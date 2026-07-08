import json
import os
import time
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision import transforms

import pytorch_lightning as pl


class VoltageDataset(data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x

class MinMaxTransform(nn.Module):
    def __init__(self, min_val, max_val):
        super(MinMaxTransform, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x_t = x.permute(*torch.arange(x.ndim - 1, -1, -1))
        x_t_scaled = ((x_t - self.min_val) / (self.max_val - self.min_val))
        return x_t_scaled.permute(*torch.arange(x_t_scaled.ndim - 1, -1, -1))

    def inverse(self, x_norm):
        x_t = x_norm.permute(*torch.arange(x_norm.ndim - 1, -1, -1))
        x_t_unscaled = (x_t * (self.max_val - self.min_val)) + self.min_val
        return x_t_unscaled.permute(*torch.arange(x_t_unscaled.ndim - 1, -1, -1))

    def jac(self, x):
        n = x.shape[-1]
        s =  1. / (self.max_val - self.min_val)
        j = torch.diag(torch.cat([torch.ones(n) * s[0], torch.ones(n) * s[1]], dim=0))
        return j.to(torch.get_default_dtype())

    def jac_inv(self, x):
        n = x.shape[-1]
        s =  (self.max_val - self.min_val)
        j = torch.diag(torch.cat([torch.ones(n) * s[0], torch.ones(n) * s[1]], dim=0))
        return j.to(torch.get_default_dtype())


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        x_t = x.permute(*torch.arange(x.ndim - 1, -1, -1))
        x_t_scaled = ((x_t - self.mean) / self.std)
        return x_t_scaled.permute(*torch.arange(x_t_scaled.ndim - 1, -1, -1))

    def inverse(self, x_norm):
        x_t = x_norm.permute(*torch.arange(x_norm.ndim - 1, -1, -1))
        x_t_unscaled = (x_t * self.std) + self.mean
        return x_t_unscaled.permute(*torch.arange(x_t_unscaled.ndim - 1, -1, -1))

    def jac(self, x):
        n = x.shape[-1]
        s =  1. / self.std
        j = torch.diag(torch.cat([torch.ones(n) * s[0], torch.ones(n) * s[1]], dim=0))
        return j.to(torch.get_default_dtype())

    def jac_inv(self, x):
        n = x.shape[-1]
        s =  self.std
        j = torch.diag(torch.cat([torch.ones(n) * s[0], torch.ones(n) * s[1]], dim=0))
        return j.to(torch.get_default_dtype())


def _load_topology(config):
    from SE_torch.net_preprocess.process_net_data import (
        parse_ieee_mat, System, Branch,
    )

    file = config.get('file')
    if file is None:
        raise ValueError(
            "config['file'] must point to a .mat net file when "
            "mask_type='smart' is used."
        )

    parsed = parse_ieee_mat(file)
    sys = System(parsed['data']['system'])
    branch = Branch(sys.branch)

    slk_bus = int(sys.slk_bus[0])
    nb = int(sys.nb)

    src = branch.i.to(torch.long)
    dst = branch.j.to(torch.long)
    keep = (src != dst) & (src != slk_bus) & (dst != slk_bus)
    src, dst = src[keep], dst[keep]

    shift = (torch.arange(nb) > slk_bus).to(torch.long)
    src = src - shift[src]
    dst = dst - shift[dst]
    edge_index = torch.stack([src, dst], dim=0)

    keep_buses = torch.arange(nb) != slk_bus
    Ybus = sys.Ybus[keep_buses][:, keep_buses]

    return edge_index, Ybus, slk_bus


def load_data(config, cart=False):
    prefix = 'vri' if cart else 'vam'
    train_data = torch.load(
        f"../../data_parser/data/time_series4/train_dataset_{prefix}.pt",
        weights_only=False).to(torch.get_default_dtype()).squeeze(2)
    test_data = torch.load(
        f"../../data_parser/data/time_series4/test_dataset_{prefix}.pt",
        weights_only=False).to(torch.get_default_dtype()).squeeze(2)
    train_data = torch.cat([train_data[:, :, :68], train_data[:, :, 69:]], dim=-1)
    test_data = torch.cat([test_data[:, :, :68], test_data[:, :, 69:]], dim=-1)
    mean = torch.load(f"../../data_parser/data/time_series4/mean_{prefix}.pt").to(torch.get_default_dtype())
    std = torch.load(f"../../data_parser/data/time_series4/std_{prefix}.pt").to(torch.get_default_dtype())
    transform = transforms.Compose([
        Normalize(mean, std),
    ])
    train_dataset = VoltageDataset(train_data, transform=transform)

    n = len(train_dataset)
    t, v = int(0.9 * n), n - int(0.9 * n)

    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [t, v])
    test_set = VoltageDataset(test_data, transform=transform)

    batch_size = config.get('batch_size', 1)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = data.DataLoader(val_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

    edge_index, Ybus = None, None
    need_topology = (
        config.get('mask_type') == 'smart'
        or bool(config.get('load_topology', False))
    )
    if need_topology:
        edge_index, Ybus, slk_bus = _load_topology(config)
        h_data = train_data.shape[-1]
        if Ybus.shape[0] != h_data:
            raise ValueError(
                f"Topology bus count {Ybus.shape[0]} doesn't match data bus "
                f"count {h_data}. Check that the slack-bus drop is consistent."
            )
        print(f"[load_data] loaded topology from '{config.get('file')}': "
              f"h={Ybus.shape[0]}, E={edge_index.shape[1]}, "
              f"slack_bus(orig)={slk_bus}")

    return (train_loader, test_loader, val_loader,
            train_set, test_set, val_set,
            edge_index, Ybus)


class VoltageFlow(pl.LightningModule):

    def __init__(self, flows, import_samples=8):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
            import_samples - Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.import_samples = import_samples
        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, imgs):
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(imgs)

    def encode(self, imgs):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, ldj = imgs, torch.zeros(imgs.shape[0], device=self.device)
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def decode(self, z):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        imag, ldj = z, torch.zeros(z.shape[0], device=self.device)
        for flow in reversed(self.flows):
            imag, ldj = flow(imag, ldj, reverse=True)
        return imag, ldj

    def inverse(self, imgs):
        z, _ = self.encode(imgs)
        return z

    def log_det_inv_jacobian(self, imgs):
        _, ldj = self.encode(imgs)
        return -ldj

    def log_det_jacobian(self, eps):
        _, ldj = self.decode(eps)
        return ldj

    def _get_likelihood(self, imgs, return_ll=False):
        z, ldj = self.encode(imgs)
        log_pz = self.prior.log_prob(z).sum(dim=[1, 2])
        log_px = ldj + log_pz
        nll = -log_px
        bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        return bpd.mean() if not return_ll else log_px

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = self._get_likelihood(batch)
        self.log('train_bpd', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_likelihood(batch)
        self.log('val_bpd', loss)

    def test_step(self, batch, batch_idx):
        samples = []
        for _ in range(self.import_samples):
            img_ll = self._get_likelihood(batch, return_ll=True)
            samples.append(img_ll)
        img_ll = torch.stack(samples, dim=-1)
        img_ll = torch.logsumexp(img_ll, dim=-1) - np.log(self.import_samples)

        bpd = -img_ll * np.log2(np.exp(1)) / np.prod(batch.shape[1:])
        bpd = bpd.mean()

        self.log('test_bpd', bpd)


class CouplingLayer(nn.Module):
    S_FAC_MAX = 1.5

    def __init__(self, network, mask, c_in):

        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.full((c_in,), -1.0))
        self.register_buffer('mask', mask)

    def forward(self, z, ldj, reverse=False, orig_img=None):
        z_in = z * self.mask
        if orig_img is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(torch.cat([z_in, orig_img], dim=1))
        s, t = nn_out.chunk(2, dim=1)

        s_fac = self.scaling_factor.exp().clamp_max(self.S_FAC_MAX).view(1, -1, 1)
        s = torch.tanh(s / s_fac) * s_fac

        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        if not reverse:
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1, 2])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1, 2])

        return z, ldj

def create_checkerboard_mask(h, i, invert=False):
    x = torch.arange(h, dtype=torch.int32)
    indices = torch.multinomial(torch.ones(h), h // 2, replacement=True, generator=torch.Generator().manual_seed(i))
    mask = torch.isin(x, indices)

    mask = mask.to(torch.float32).view(1, 1, h)
    if invert:
        mask = 1 - mask
    return mask


def create_bus_mask(h, i, invert=False):
    g = torch.Generator().manual_seed(int(i))
    perm = torch.randperm(h, generator=g)
    half = h // 2
    mask = torch.zeros(h, dtype=torch.float32)
    mask[perm[:half]] = 1.0
    mask = mask.view(1, 1, h)
    if invert:
        mask = 1.0 - mask
    return mask


def _build_adjacency(h, edge_index=None, Y=None):
    if edge_index is None and Y is None:
        raise ValueError(
            "Need either `edge_index` or `Y` to build a smart bus mask."
        )

    A = torch.zeros(h, h, dtype=torch.get_default_dtype())
    if edge_index is not None:
        ei = torch.as_tensor(edge_index, dtype=torch.long)
        if ei.dim() != 2 or ei.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(ei.shape)}."
            )
        src, dst = ei[0], ei[1]
        keep = src != dst
        src, dst = src[keep], dst[keep]
        A[src, dst] = 1.0
        A[dst, src] = 1.0
    else:
        Yt = torch.as_tensor(Y)
        if Yt.shape != (h, h):
            raise ValueError(
                f"Y must have shape ({h}, {h}), got {tuple(Yt.shape)}."
            )
        W = Yt.abs().to(torch.float64) if torch.is_complex(Yt) \
            else Yt.to(torch.float64).abs()
        W = W.clone()
        idx = torch.arange(h)
        W[idx, idx] = 0.0
        A = 0.5 * (W + W.T)

    return A


def create_smart_bus_mask(h, i, edge_index=None, Y=None, invert=False,
                          method='spectral'):
    A = _build_adjacency(h, edge_index=edge_index, Y=Y)
    half = h // 2

    if method == 'spectral':
        deg = A.sum(dim=1)
        L = torch.diag(deg) - A
        eigvals, eigvecs = torch.linalg.eigh(L)
        nontrivial_rank = min((i // 2) + 1, h - 1)
        v = eigvecs[:, nontrivial_rank]
        order = torch.argsort(v, descending=True)
        mask = torch.zeros(h, dtype=torch.float32)
        mask[order[:half]] = 1.0

    elif method == 'greedy':
        g = torch.Generator().manual_seed(int(i))
        order = torch.randperm(h, generator=g)
        color = -torch.ones(h, dtype=torch.long)
        for u in order.tolist():
            neigh = torch.nonzero(A[u] > 0, as_tuple=False).flatten().tolist()
            used = {int(color[v_].item()) for v_ in neigh if color[v_].item() >= 0}
            color[u] = 0 if 0 not in used else 1

        ones_idx = torch.nonzero(color == 0, as_tuple=False).flatten()
        zeros_idx = torch.nonzero(color == 1, as_tuple=False).flatten()
        if len(ones_idx) > half and len(zeros_idx) > 0:
            excess = len(ones_idx) - half
            scores = (A[ones_idx][:, zeros_idx].sum(dim=1)
                      - A[ones_idx][:, ones_idx].sum(dim=1))
            move = torch.topk(scores, excess).indices
            color[ones_idx[move]] = 1
        elif len(ones_idx) < half and len(zeros_idx) > 0:
            deficit = half - len(ones_idx)
            scores = (A[zeros_idx][:, ones_idx].sum(dim=1)
                      - A[zeros_idx][:, zeros_idx].sum(dim=1))
            move = torch.topk(scores, deficit).indices
            color[zeros_idx[move]] = 0
        mask = (color == 0).to(torch.float32)

    else:
        raise ValueError(
            f"unknown method '{method}', expected 'spectral' or 'greedy'."
        )

    mask = mask.view(1, 1, h)
    if invert:
        mask = 1.0 - mask
    return mask

def create_channel_mask(c_in, invert=False):
    mask = torch.cat([torch.ones(c_in//2, dtype=torch.float32),
                      torch.zeros(c_in-c_in//2, dtype=torch.float32)])
    mask = mask.view(1, c_in, 1)
    if invert:
        mask = 1 - mask
    return mask


class ConcatELU(nn.Module):
    def forward(self, x):
        return torch.cat([F.gelu(x), F.gelu(-x)], dim=1)


class LayerNormChannels(nn.Module):

    def __init__(self, c_in, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, c_in, 1))
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        y = y * self.gamma + self.beta
        return y


class GatedConv(nn.Module):

    def __init__(self, c_in, c_hidden):

        super().__init__()
        self.net = nn.Sequential(
            ConcatELU(),
            nn.Conv1d(2 * c_in, c_hidden, kernel_size=3, padding=1),
            ConcatELU(),
            nn.Conv1d(2 * c_hidden, 2 * c_in, kernel_size=1)
        )

    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)


class GatedConvNet(nn.Module):

    def __init__(self, c_in, c_hidden=32, c_out=-1, num_layers=3):
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = []
        layers += [nn.Conv1d(c_in, c_hidden, kernel_size=3, padding=1)]
        for layer_index in range(num_layers):
            layers += [GatedConv(c_hidden, c_hidden),
                       LayerNormChannels(c_hidden)]
        layers += [ConcatELU(),
                   nn.Conv1d(2 * c_hidden, c_out, kernel_size=3, padding=1)]
        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)

def _maybe_spec_norm(linear, enable):
    if not enable:
        return linear
    return nn.utils.parametrizations.spectral_norm(linear)


class GatedLinear(nn.Module):
    def __init__(self, d_in, d_hidden, spectral=True):
        super().__init__()
        self.net = nn.Sequential(
            ConcatELU(),
            _maybe_spec_norm(nn.Linear(2 * d_in, d_hidden), spectral),
            ConcatELU(),
            _maybe_spec_norm(nn.Linear(2 * d_hidden, 2 * d_in), spectral),
        )

    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=-1)
        return x + val * torch.sigmoid(gate)


class GatedMLP(nn.Module):
    def __init__(self, c_in, h, c_hidden=256, c_out=-1, num_layers=3,
                 spectral=True):
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        self.c_in = c_in
        self.c_out = c_out
        self.h = h
        in_dim = c_in * h
        out_dim = c_out * h

        layers = [_maybe_spec_norm(nn.Linear(in_dim, c_hidden), spectral)]
        for _ in range(num_layers):
            layers += [GatedLinear(c_hidden, c_hidden, spectral=spectral)]#, nn.LayerNorm(c_hidden)]

        layers += [ConcatELU(),
                   nn.Linear(2 * c_hidden, out_dim)]
        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        out = self.nn(x_flat)
        return out.reshape(B, self.c_out, self.h)


class SqueezeFlow(nn.Module):

    def forward(self, z, ldj, reverse=False):
        B, C, H, W = z.shape
        if not reverse:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, C, H, W // 2, 2)
            z = z.permute(0, 1, 4, 2, 3)
            z = z.reshape(B, 2 * C, H, W // 2)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, C // 2, 2, H, W)
            z = z.permute(0, 1, 3, 4, 2)
            z = z.reshape(B, C // 2, H, W * 2)
        return z, ldj


class SplitFlow(nn.Module):

    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z, z_split = z.chunk(2, dim=1)
            ldj += self.prior.log_prob(z_split).sum(dim=[1, 2, 3])
        else:
            z_split = self.prior.sample(sample_shape=z.shape).to(self.device)
            z = torch.cat([z, z_split], dim=1)
            ldj -= self.prior.log_prob(z_split).sum(dim=[1, 2, 3])
        return z, ldj


def create_voltage_flow(device, **kwargs):
    c_hidden = kwargs.get("c_hidden", 32)
    c_in = kwargs.get("c_in", 2)
    h = kwargs.get("h", 117)
    num_layers = kwargs.get("num_layers", 4)
    mlp_depth = kwargs.get("mlp_depth", 3)
    spectral = bool(kwargs.get("spectral_norm", False))

    # --- bus-mask configuration ---
    mask_type = kwargs.get("mask_type", "random")  # 'random' | 'smart'
    edge_index = kwargs.get("edge_index", None)
    Y = kwargs.get("Y", None)
    smart_method = kwargs.get("smart_method", "spectral")

    def _bus_mask(layer_i, invert):
        if mask_type == "smart":
            return create_smart_bus_mask(h=h, i=layer_i,
                                         edge_index=edge_index, Y=Y,
                                         invert=invert,
                                         method=smart_method)
        return create_bus_mask(h=h, i=layer_i, invert=invert)

    flow_layers = []
    for i in range(num_layers):
        flow_layers += [CouplingLayer(network=GatedMLP(c_in=c_in, h=h,
                                                      c_hidden=c_hidden,
                                                      num_layers=mlp_depth,
                                                      spectral=spectral),
                                      mask=_bus_mask(i, invert=(i % 2 == 1)),
                                      c_in=c_in)]
        # Channel-axis coupling: real <-> imag, conditioner is still MLP
        flow_layers += [CouplingLayer(network=GatedMLP(c_in=c_in, h=h,
                                                      c_hidden=c_hidden,
                                                      num_layers=mlp_depth,
                                                      spectral=spectral),
                                      mask=create_channel_mask(c_in=c_in, invert=(i % 2 == 1)),
                                      c_in=c_in)]

    flow_model = VoltageFlow(flow_layers).to(device)
    return flow_model


def create_multiscale_flow(device, **kwargs):
    c_hidden = kwargs.get("c_hidden", 32)
    c_in = kwargs.get("c_in", 2)
    h = kwargs.get("h", 1)
    w = kwargs.get("w", 118)
    num_layers = kwargs.get("num_layers", 4)
    flow_layers = []

    flow_layers += [CouplingLayer(network=GatedConvNet(c_in=c_in, c_hidden=c_hidden),
                                  mask=create_checkerboard_mask(h=h, w=w, invert=(i % 2 == 1)),
                                  c_in=c_in) for i in range(2)]
    flow_layers += [SqueezeFlow()]
    for i in range(num_layers // 2):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=2 * c_in, c_hidden=c_hidden + 16),
                                      mask=create_channel_mask(c_in=2 * c_in, invert=(i % 2 == 1)),
                                      c_in=2 * c_in)]
    flow_layers += [SplitFlow(device),
                    SqueezeFlow()]
    for i in range(num_layers):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=2 * c_in, c_hidden=c_hidden + 32),
                                      mask=create_channel_mask(c_in=2 * c_in, invert=(i % 2 == 1)),
                                      c_in=2 * c_in)]

    flow_model = VoltageFlow(flow_layers).to(device)
    return flow_model


class VoltageFlowTrainer:
    def __init__(self, model, loaders, config, device,
                 checkpoint_path, model_name):
        self.model = model
        (self.train_loader_seq, self.test_loader, self.val_loader,
         self.train_set, self.test_set, self.val_set,
         self.edge_index, self.Ybus) = loaders

        # The loader returned by `load_data` is shuffle=False; for training
        # we want shuffled minibatches with drop_last + pin_memory.
        bs = int(config.get('batch_size', 128))
        self.train_loader = data.DataLoader(
            self.train_set, batch_size=bs, shuffle=True,
            drop_last=True, pin_memory=True,
            num_workers=int(config.get('num_workers', 4)),
        )

        self.num_epochs = int(config.get('num_epochs', 200))
        self.lr = float(config.get('learning_rate', 1e-3))
        self.grad_clip = float(config.get('grad_clip', 1.0))
        self.device = device
        self.config = config

        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.ckpt_dir = os.path.join('..', 'models', checkpoint_path,
                                     model_name, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.last_ckpt = os.path.join(self.ckpt_dir, f"{model_name}_last.pth")
        self.best_ckpt = os.path.join(self.ckpt_dir, f"{model_name}_best.pth")

        weight_decay = float(config.get('weight_decay', 1e-4))
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=weight_decay)
        sched_type = str(config.get('scheduler', 'step')).lower()
        if sched_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1,
                gamma=float(config.get('lr_gamma', 0.99)),
            )


    def _bpd(self, batch):
        return self.model._get_likelihood(batch.to(self.device))

    def _save_ckpt(self, epoch, val_loss, is_best):
        state = {
            'epoch': epoch,
            'val_bpd': val_loss,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
        }
        epoch_ckpt = os.path.join(
            self.ckpt_dir, f"{self.model_name}_epoch{epoch:03d}.pth"
        )
        torch.save(state, epoch_ckpt)
        torch.save(state, self.last_ckpt)
        if is_best:
            torch.save(state, self.best_ckpt)

    def _maybe_resume(self):
        if not os.path.isfile(self.last_ckpt):
            return 1, float('inf')
        state = torch.load(self.last_ckpt, map_location=self.device)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        start_epoch = int(state['epoch']) + 1
        best_val = float(state.get('val_bpd', float('inf')))
        if os.path.isfile(self.best_ckpt):
            try:
                best_state = torch.load(self.best_ckpt, map_location='cpu')
                best_val = min(best_val, float(best_state.get('val_bpd', best_val)))
            except Exception:
                pass
        print(f"[VoltageFlowTrainer] resuming from epoch {state['epoch']} "
              f"(val_bpd={state.get('val_bpd', float('nan')):.4f}, "
              f"best_val={best_val:.4f})")
        return start_epoch, best_val


    def train_epoch(self, epoch):
        self.model.train()
        running = 0.0
        n = 0
        pbar = tqdm(self.train_loader,
                    desc=f"VoltageFlow [Epoch {epoch}/{self.num_epochs}] 🟢 Training",
                    colour='green', leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()
            loss = self._bpd(batch)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            running += loss.item()
            n += 1
            pbar.set_postfix(bpd=f"{loss.item():.4f}")
        self.scheduler.step()
        return running / max(n, 1)

    def validate_epoch(self, epoch):
        self.model.eval()
        agg, n = 0.0, 0
        pbar = tqdm(self.val_loader,
                    desc=f"VoltageFlow [Epoch {epoch}/{self.num_epochs}] 🔵 Validating",
                    colour='blue', leave=False)
        with torch.no_grad():
            for batch in pbar:
                loss = self._bpd(batch)
                agg += loss.item()
                n += 1
                pbar.set_postfix(bpd=f"{loss.item():.4f}")
        return agg / max(n, 1)

    def test_epoch(self):
        self.model.eval()
        agg, n = 0.0, 0
        pbar = tqdm(self.test_loader, desc="VoltageFlow 🟣 Test",
                    colour='magenta', leave=False)
        with torch.no_grad():
            for batch in pbar:
                loss = self._bpd(batch)
                agg += loss.item()
                n += 1
                pbar.set_postfix(bpd=f"{loss.item():.4f}")
        return agg / max(n, 1)

    def train(self):
        start_epoch, best_val = self._maybe_resume()
        train_losses, val_losses = [], []

        epoch_bar = tqdm(range(start_epoch, self.num_epochs + 1),
                        desc='Epochs', colour='cyan', disable=True)
        for epoch in epoch_bar:
            t_loss = self.train_epoch(epoch)
            v_loss = self.validate_epoch(epoch)

            train_losses.append(t_loss)
            val_losses.append(v_loss)

            is_best = v_loss < best_val
            if is_best:
                best_val = v_loss
            self._save_ckpt(epoch, v_loss, is_best=is_best)

            epoch_bar.set_postfix(train=f"{t_loss:.4f}",
                                  val=f"{v_loss:.4f}",
                                  best=f"{best_val:.4f}")
            print(f"Epoch {epoch:3d} | train_bpd={t_loss:.4f} | "
                  f"val_bpd={v_loss:.4f} | best_val={best_val:.4f}"
                  + ("  *new best*" if is_best else ""))

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val': best_val,
            'last_ckpt': self.last_ckpt,
            'best_ckpt': self.best_ckpt,
        }


def train_flow(loaders, flow, checkpoint_path, device,
               model_name="VoltageFlow", config=None):
    config = config or {}
    trainer = VoltageFlowTrainer(
        model=flow, loaders=loaders, config=config, device=device,
        checkpoint_path=checkpoint_path, model_name=model_name,
    )

    print(f"[train_flow] start training '{model_name}' "
          f"for {trainer.num_epochs} epochs on {device}")
    history = trainer.train()

    if os.path.isfile(trainer.best_ckpt):
        best_state = torch.load(trainer.best_ckpt, map_location=device)
        flow.load_state_dict(best_state['state_dict'])
    start_time = time.time()
    test_bpd = trainer.test_epoch()
    duration = time.time() - start_time

    result = {
        "test_bpd": test_bpd,
        "best_val_bpd": history['best_val'],
        "train_losses": history['train_losses'],
        "val_losses": history['val_losses'],
        "test_time_per_batch": duration / max(len(trainer.test_loader), 1),
        "last_ckpt": history['last_ckpt'],
        "best_ckpt": history['best_ckpt'],
    }
    print(f"[train_flow] test_bpd={test_bpd:.4f}  "
          f"best_val_bpd={history['best_val']:.4f}")
    return flow, result

def print_num_params(model):
    num_params = sum([np.prod(p.shape) for p in model.parameters()])
    print("Number of parameters: {:,}".format(num_params))

def main(config_path):
    config = json.load(open(config_path))
    cart = config.get('cart', False)

    loaders = load_data(config, cart=cart)
    (train_loader, test_loader, val_loader,
     train_set, test_set, val_set,
     edge_index, Ybus) = loaders

    checkpoint_path = config.get('checkpoint_path', "../models")
    os.makedirs(checkpoint_path, exist_ok=True)
    model_name = config.get('model_name', "VoltageFlow")

    pl.seed_everything(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cpu") if not torch.mps.is_available() else torch.device("mps")
    print("Using device", device)

    flow_kwargs = dict(config)
    if edge_index is not None:
        flow_kwargs['edge_index'] = edge_index
    if Ybus is not None:
        flow_kwargs['Y'] = Ybus
    flow = create_voltage_flow(device, **flow_kwargs)
    print_num_params(flow)
    model, results = train_flow(loaders, flow, checkpoint_path, device,
                                model_name=model_name, config=config)
    print("Results:", results)

if __name__ == "__main__":
    conf = f'../configs/nf_model_config_smart_v0.1.json'
    main(conf)