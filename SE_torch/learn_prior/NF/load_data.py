import os

import torch
from torch import nn
from torch.utils.data import TensorDataset, Subset, DataLoader, Dataset
from torchvision import transforms
import pytorch_lightning as pl
from SE_torch.data_generator import DataGenerator
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System, Branch
from SE_torch.utils import init_start_point

DATA_DIM = 118
HALF_DATA_DIM = 59
CHANNELS = 2

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

class VoltageDataset(Dataset):
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


def load_data(config, n_samples, cart=False, seed=555):
    file = config.get('file')
    if file is None:
        raise(ValueError('Please specify a net file'))

    prefix = 'vri' if cart else 'vam'
    train_dataset = torch.load(f"../../data_parser/data/time_series3/train_dataset_{prefix}.pt", weights_only=False)
    test_dataset = torch.load(f"../../data_parser/data/time_series3/test_dataset_{prefix}.pt", weights_only=False)
    train_dataset = train_dataset.dataset.tensors[0][train_dataset.indices].to(torch.get_default_dtype()).squeeze(2)
    # train_dataset = torch.cat([train_data[:, :, :68], train_data[:, :, 69:]], dim=-1)
    # train_dataset = train_dataset.reshape(len(train_dataset), -1)
    test_dataset = test_dataset.dataset.tensors[0][test_dataset.indices].to(torch.get_default_dtype()).squeeze(2)
    # test_dataset = torch.cat([test_data[:, :, :68], test_data[:, :, 69:]], dim=-1)
    # test_dataset = test_dataset.reshape(len(test_data), -1)
    mean = torch.load(f"../../data_parser/data/time_series3/mean_{prefix}.pt").to(torch.get_default_dtype())
    # mean = torch.cat([mean[:68], mean[69:]], dim=-1)
    std = torch.load(f"../../data_parser/data/time_series3/std_{prefix}.pt").to(torch.get_default_dtype())
    # std = torch.cat([std[:68], std[69:]], dim=-1)
    transform = transforms.Compose([
        Normalize(mean=mean, std=std)
    ])
    train_dataset = VoltageDataset(train_dataset, transform=transform)

    n = len(train_dataset)
    t, v = int(0.9 * n), n - int(0.9 * n)

    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [t, v])


    # We define a set of data loaders that we can use for various purposes later.
    # Note that for actually training a model, we will use different data loaders
    # with a lower batch size.
    # train_loader = data.DataLoader(train_set, batch_size=256, shuffle=False, drop_last=False)
    # val_loader = data.DataLoader(val_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    # test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    test_dataset = VoltageDataset(test_dataset, transform=transform)

    batch_size = config.get('batch_size', 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset, config

def _normalize_nf_data(data, mean, cov, config):
    normalization = config.get("data_normalization", "diag")
    dim = data.shape[1]
    eye = torch.eye(dim, dtype=data.dtype, device=data.device)
    cov_eps = cov + (1e-12 * eye)
    std = torch.diag(cov_eps.diag().sqrt())

    if normalization == "chol":
        L = torch.linalg.cholesky(cov_eps)
        L_inv = torch.linalg.inv(L)
        return (data - mean) @ L_inv.T
    if normalization == "diag":
        std_inv = torch.diag(1.0 / std.diag().clamp_min(1e-6))
        return (data - mean) @ std_inv.T
    if normalization == "minmax":
        data_min = config.get("data_min")
        data_max = config.get("data_max")
        if data_min is None or data_max is None:
            raise ValueError("Min-max normalization requires 'data_min' and 'data_max' in config.")
        center = 0.5 * (data_max + data_min)
        half_range = (0.5 * (data_max - data_min)).clamp_min(1e-6)
        return (data - center) / half_range
    if normalization == "mean":
        return data - mean
    if normalization == "none":
        return data
    raise ValueError(f"Unsupported data_normalization '{normalization}'")

# def load_data(config, n_samples, cart=False, seed=555):
#     file = config.get('file')
#     if file is None:
#         raise(ValueError('Please specify a net file'))
#
#     # data = parse_ieee_mat(file)
#     # system_data = data['data']['system']
#     # sys = System(system_data)
#     # branch = Branch(sys.branch)
#     # edge_index = torch.stack([branch.i, branch.j]).to(torch.long)
#     # config['slk_bus'] = sys.slk_bus
#     # config['edge_index'] = edge_index
#     prefix = 'vri' if cart else 'vam'
#     # slk_idx = sys.slk_bus[0] + sys.nb if cart else sys.slk_bus[0]
#     train_dataset = torch.load(f"../../data_parser/data/time_series3/train_dataset_{prefix}.pt", weights_only=False)
#     test_dataset = torch.load(f"../../data_parser/data/time_series3/test_dataset_{prefix}.pt", weights_only=False)
#     # if os.path.exists(f'../datasets/data{prefix}.pt'):
#     #     data = torch.load(f'../datasets/data{prefix}.pt').to(torch.get_default_dtype())
#     #     T0, V0 = init_start_point(sys, how='flat')
#     #     data = torch.concat([data, torch.concat([T0, V0], dim=0).unsqueeze(0)], dim=0)
#     #     n_samples += 1
#     # else:
#     #     data_generator = DataGenerator(device="cpu")
#     #     kwargs = {
#     #         "sample_save_path": f'../datasets/data{prefix}.pt',
#     #         "sample_save_every": 1000,
#     #     }
#     #     if cart:
#     #         v_real, v_imag = data_generator.sample(sys, num_samples=n_samples, random_flow=True, cart=True, verbose=True)
#     #         data = torch.concat([v_real, v_imag], dim=1)
#     #     else:
#     #         T, V = data_generator.sample(sys, num_samples=n_samples, random_flow=True, verbose=True, seed=seed, **kwargs)
#     #         T0, V0 = init_start_point(sys, how='flat')
#     #         data = torch.concat([T, V] , dim=1)
#     #         data = torch.concat([data, torch.concat([T0, V0], dim=0).unsqueeze(0)], dim=0)
#         #
#         # data = torch.tensor(data)
#         # torch.save(data, f'../datasets/data{prefix}.pt')
#     # train_dataset = torch.cat([train_dataset[:, :slk_idx], train_dataset[:, slk_idx + 1:]], dim=1)
#     # test_dataset = torch.cat([test_dataset[:, :slk_idx], test_dataset[:, slk_idx:]], dim=1)
#     # data = torch.concat([data[:, :slk_idx], data[:, slk_idx + 1:]], dim=1).to(torch.get_default_dtype())
#     mean = torch.load(f"../../data_parser/data/time_series3/mean_{prefix}.pt")
#     std = torch.load(f"../../data_parser/data/time_series3/std_{prefix}.pt")
#     transform = transforms.Compose([
#         transforms.Normalize(mean=mean, std=std)
#     ])
#     train_dataset = VoltageDataset(train_dataset.dataset.tensors[0], transform=transform)
#     test_dataset = VoltageDataset(test_dataset.dataset.tensors[0], transform=transform)
#     # mean = train_dataset.mean(0)
#     # cov = train_dataset.T.cov()
#     # torch.save(mean, f'../../data_parser/data/time_series3/mean_{prefix}.pt')
#     # torch.save(cov, f'../../data_parser/data/time_series3/cov_{prefix}.pt')
#     # cov_for_stats = cov + (torch.eye(cov.shape[0], dtype=data.dtype, device=data.device) * 1e-12)
#     # std = torch.diag(cov_for_stats.diag().sqrt())
#     # torch.save(std, f'SE_torch/data_parser/data/time_series3/std_{prefix}.pt')
#     #
#     # data_min = data.min(dim=0).values
#     # data_max = data.max(dim=0).values
#     # config["data_min"] = data_min
#     # config["data_max"] = data_max
#     # torch.save(data_min, f'../datasets/min_NF{prefix}.pt')
#     # torch.save(data_max, f'../datasets/max_NF{prefix}.pt')
#     # train_dataset = _normalize_nf_data(train_dataset, mean, cov, config)
#     # test_dataset = _normalize_nf_data(test_dataset, mean, cov, config)
#
#     # dataset_size = data.shape[0]
#     # n_train = int(0.8 * dataset_size)
#     # permutation = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed))
#     # train_indices = permutation[:n_train]
#     # test_indices = permutation[n_train:]
#     # dataset = TensorDataset(data)
#     # train_dataset = Subset(dataset, train_indices.tolist())
#     # test_dataset = Subset(dataset, test_indices.tolist())
#     batch_size = config.get('batch_size', 1)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, test_loader, train_dataset, test_dataset, config
