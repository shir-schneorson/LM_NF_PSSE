import os

import torch
from torch.utils.data import TensorDataset, Subset, DataLoader, random_split
from sklearn.model_selection import train_test_split

from SE_torch.data_generator import DataGenerator
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System

DATA_DIM = 118
HALF_DATA_DIM = 59
CHANNELS = 2


def load_data(config, n_samples, cart=False, seed=42):
    file = config.get('file')
    if file is None:
        raise(ValueError('Please specify a net file'))

    data = parse_ieee_mat(file)
    system_data = data['data']['system']
    sys = System(system_data)
    config['slk_bus'] = sys.slk_bus
    prefix = '_cart' if cart else '_polar'
    slk_idx = sys.slk_bus[0] + sys.nb if cart else sys.slk_bus[0]
    if os.path.exists(f'../datasets/data{prefix}.pt'):
        data = torch.load(f'../datasets/data{prefix}.pt')
        mean = torch.load(f'../datasets/mean_NF{prefix}.pt')
        std = torch.load(f'../datasets/std_NF{prefix}.pt')
        cov = torch.load(f'../datasets/cov_NF{prefix}.pt')
    else:
        data_generator = DataGenerator()
        if cart:
            v_real, v_imag = data_generator.sample(sys, num_samples=n_samples, random_flow=True, cart=True, verbose=True)
            data = torch.concat([v_real, v_imag], dim=1)
        else:
            T, V = data_generator.sample(sys, num_samples=n_samples, random_flow=True)
            data = torch.concat([T, V] , dim=1)

        data = torch.tensor(data)
        torch.save(data, f'../datasets/data{prefix}.pt')
        mean, std, cov = data.mean(0), data.std(0), data.T.cov()
        std[std < 1e-10] = 1
        cov += torch.eye(DATA_DIM * CHANNELS - 1)
        torch.save(mean, f'../datasets/mean{prefix}.pt')
        torch.save(std, f'../datasets/std{prefix}.pt')
        torch.save(cov, f'../datasets/cov{prefix}.pt')

    # data = torch.concat([data[:, :slk_idx], data[:, slk_idx + 1:]], dim=1)
    # mean, std, cov = data.mean(0), data.std(0), data.T.cov()
    # std[std < 1e-10] = 1
    # cov += torch.eye(DATA_DIM * CHANNELS)
    # torch.save(mean, f'../datasets/mean_NF{prefix}.pt')
    # torch.save(std, f'../datasets/std_NF{prefix}.pt')
    # torch.save(cov, f'../datasets/cov_NF{prefix}.pt')

    data = (data - mean) / std
    # data = torch.concat([data[:, :slk_idx], torch.zeros(len(data), 1), data[:, slk_idx:]], dim=1)
    dataset = TensorDataset(data)
    n_train = int(0.8 * n_samples)
    n_test = n_samples - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test],
                                               generator=torch.Generator().manual_seed(666))
    batch_size = config.get('batch_size', 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset, config