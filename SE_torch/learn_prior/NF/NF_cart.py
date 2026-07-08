import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from SE_torch.learn_prior.NF.load_data import load_data, DATA_DIM, HALF_DATA_DIM, CHANNELS
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System

# ----------------- Device & dtype -----------------
# DEVICE = (
#     torch.device("mps") if torch.backends.mps.is_available()
#     else torch.device("cuda") if torch.cuda.is_available()
#     else torch.device("cpu")
# )
DEVICE = "cpu"
DTYPE = torch.float32  # MPS is best with float32
# torch.set_default_dtype(DTYPE)

NUM_EPOCHS = 20
BATCH_SIZE = 512
S_MAX = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
BETAS = (0.9, 0.99)
NUM_DATA_POINTS = 250
HIDDEN_DIM = 8
NUM_HIDDEN_LAYERS = 0
NUM_BLOCKS = 2


class AffineCouplingLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        device = kwargs.get('device', DEVICE)
        dtype = kwargs.get('dtype', DTYPE)
        channels = kwargs.get('channels', CHANNELS)
        half_data_dim = kwargs.get('half_data_dim', HALF_DATA_DIM)
        hidden_dim = kwargs.get('hidden_dim',HIDDEN_DIM)
        num_hidden_layers = kwargs.get('num_hidden_layers', NUM_HIDDEN_LAYERS)

        self.mix_channels = nn.Linear(channels, 1).to(device, dtype)
        self.split_channels = nn.Linear(1, channels).to(device, dtype)

        self.log_s = nn.Sequential(nn.Linear(half_data_dim, hidden_dim).to(device, dtype),
                                   nn.LeakyReLU().to(device, dtype),)
        for _ in range(num_hidden_layers):
            self.log_s.append(nn.Linear(hidden_dim, hidden_dim).to(device, dtype))
            self.log_s.append(nn.LeakyReLU().to(device, dtype))
        self.log_s.append(nn.Linear(hidden_dim, half_data_dim - 1).to(device, dtype))

        self.b = nn.Sequential(nn.Linear(half_data_dim, hidden_dim).to(device, dtype),
                               nn.LeakyReLU().to(device, dtype),)
        for _ in range(num_hidden_layers):
            self.b.append(nn.Linear(hidden_dim, hidden_dim).to(device, dtype))
            self.b.append(nn.LeakyReLU().to(device, dtype))
        self.b.append(nn.Linear(hidden_dim, half_data_dim - 1).to(device, dtype))

    def forward(self, z):
        # z: [B, DATA_DIM]
        z_l, z_r = z.chunk(2, dim=1)
        # z_l_mix = self.mix_channels(z_l).squeeze(-1)

        log_s = self.log_s(z_l)
        # log_s = self.split_channels(log_s)
        log_s = S_MAX * torch.tanh(log_s)
        s = torch.exp(log_s)

        b = self.b(z_l)
        # b = self.split_channels(b)

        y_l = z_l
        y_r = s * z_r + b
        return torch.cat([y_l, y_r], dim=1)

    def inverse(self, y):
        y_l, y_r = y.chunk(2, dim=1)
        # y_l_mix = self.mix_channels(y_l).squeeze(-1)

        log_s = self.log_s(y_l)
        # log_s = self.split_channels(log_s)
        log_s = S_MAX * torch.tanh(log_s)
        s = torch.exp(log_s)

        b = self.b(y_l)
        # b = self.split_channels(b)

        z_l = y_l
        z_r = (y_r - b) / s
        return torch.cat([z_l, z_r], dim=1)

    def log_det_inv_jacobian(self, y):
        y_l, _ = y.chunk(2, dim=1)
        # y_l_mix = self.mix_channels(y_l).squeeze(-1)
        log_s = self.log_s(y_l)
        log_s = S_MAX * torch.tanh(log_s)
        # log_s = self.split_channels(log_s)
        # Sum over features; return [B]
        return torch.sum(-log_s, dim=1)


class PermutationalLayer(nn.Module):
    """
    MPS-friendly permutation using index selection (no dense matmul).
    """
    def __init__(self, **kwargs):
        super().__init__()
        device = kwargs.get('device', DEVICE)
        dtype = kwargs.get('dtype', DTYPE)
        data_dim = kwargs.get('data_dim', DATA_DIM)
        perm = torch.randperm(data_dim)
        while torch.equal(perm, torch.arange(data_dim)):
            perm = torch.randperm(data_dim)
        inv_perm = torch.argsort(perm)

        perm = torch.eye(data_dim)[perm].to(device, dtype)
        inv_perm = torch.eye(data_dim)[inv_perm].to(device, dtype)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, z):
        # z = z.transpose(1, 2)
        return (z @ self.perm.T)#.transpose(1, 2)

    def inverse(self, y):
        # y = y.transpose(1, 2)
        return (y @ self.inv_perm.T)#.transpose(1, 2)


class FlowModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.slk_idx = kwargs.get('slk_idx', 68)
        self.device = kwargs.get('device', DEVICE)
        self.dtype = kwargs.get('dtype', DTYPE)
        self.data_dim = kwargs.get('data_dim', DATA_DIM)
        self.half_data_dim = kwargs.get('half_data_dim', HALF_DATA_DIM)
        self.channels = kwargs.get('channels', CHANNELS)
        num_blocks = kwargs.get('num_blocks', NUM_BLOCKS)

        self.ac_layers = nn.ModuleList([AffineCouplingLayer(**kwargs) for _ in range(num_blocks)])
        self.perm_layers = nn.ModuleList([PermutationalLayer(**kwargs) for _ in range(num_blocks)])

    def _remove_slack(self, x):
        return torch.cat([x[:, :self.slk_idx + self.data_dim],
                          x[:, self.slk_idx + self.data_dim + 1:]], dim=1)


    def _insert_zero_at_slack(self, x):
        return torch.cat([x[:, :self.slk_idx + self.data_dim],
                          torch.zeros(x.shape[0], 1).to(self.device, self.dtype),
                          x[:, self.slk_idx + self.data_dim:]], dim=1)

    def forward(self, z):
        # z = self._insert_zero_at_slack(z)
        # z = z.reshape((-1, self.channels, self.data_dim)).transpose(1, 2)
        for ac_layer, perm_layer in zip(self.ac_layers, self.perm_layers):
            z = ac_layer(perm_layer(z))
        # z = z.transpose(1, 2).reshape((-1, self.channels * self.data_dim))
        # z = self._remove_slack(z)
        return z

    def inverse(self, y):
        # y = self._insert_zero_at_slack(y)
        # y = y.reshape((-1, self.channels, self.data_dim)).transpose(1, 2)
        for ac_layer, perm_layer in reversed(list(zip(self.ac_layers, self.perm_layers))):
            y = perm_layer.inverse(ac_layer.inverse(y))
        # y = y.transpose(1, 2).reshape((-1, self.channels * self.data_dim))
        # y = self._remove_slack(y)
        return y

    def log_det_inv_jacobian(self, y):
        # y = self._insert_zero_at_slack(y)
        # y = y.reshape((-1, self.channels, self.data_dim)).transpose(1, 2)
        log_det = torch.zeros(y.shape[0]).to(y.device, y.dtype)
        for ac_layer, perm_layer in reversed(list(zip(self.ac_layers, self.perm_layers))):
            log_det = log_det + ac_layer.log_det_inv_jacobian(y)
            y = perm_layer.inverse(ac_layer.inverse(y))
        return log_det


def plot_val_loss(val_losses, log_dets, log_probs):
    epochs = list(range(1, len(val_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_losses, marker='o', label=f'Validation Loss - final {val_losses[-1]:.3f}')
    plt.plot(epochs, log_dets, marker='s', label=f'LogDet - final {log_dets[-1]:.3f}')
    plt.plot(epochs, log_probs, marker='^', label=f'LogProb - final {log_probs[-1]:.3f}')
    plt.title("Training and Test Loss over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.xticks(epochs); plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('./plots/flow_validation_loss.png')
    plt.close()


class NormalizingFlowTrainer:
    def __init__(self, model, data_loaders, ckpt_path, **kwargs):
        self.lr = kwargs.get('lr', LEARNING_RATE)
        self.num_epochs = kwargs.get('num_epochs', NUM_EPOCHS)
        weight_decay = kwargs.get('weight_decay', WEIGHT_DECAY)
        self.ckpt_path = ckpt_path
        self.device = kwargs.get('device', DEVICE)
        self.dtype = kwargs.get('dtype', DTYPE)
        self.model = model
        self.train_loader, self.test_loader, self.train_dataset, self.data_mean, self.data_cov, self.data_std = data_loaders

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

        mean = torch.zeros(self.model.data_dim).to(device=self.device, dtype=self.dtype)
        cov = torch.eye(self.model.data_dim).to(device=self.device, dtype=self.dtype)
        self.prior_dist = MultivariateNormal(mean, cov)
        self.simple_data_dist = MultivariateNormal(self.data_mean, self.data_cov)


    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        desc = f'{self.model.__class__.__name__} [Epoch {epoch}] 🟢 Training'

        for batch in tqdm(self.train_loader, desc=desc, colour='green'):
            # Assuming batch = (tensor, labels?) -> take first
            y = batch[0].to(device=self.device, dtype=self.dtype)

            self.optimizer.zero_grad()

            z = self.model.inverse(y)
            log_prob = self.prior_dist.log_prob(z)                     # [B]
            log_det_inv_jacobian = self.model.log_det_inv_jacobian(y) # [B]
            loss = torch.mean(- (log_prob + log_det_inv_jacobian))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            running_loss += float(loss.item())

        self.scheduler.step()
        return running_loss / max(1, len(self.train_loader))

    def validate_epoch(self, epoch):
        self.model.eval()
        agg_val_loss = 0.0
        agg_log_det = 0.0
        agg_log_prob = 0.0
        desc = f'{self.model.__class__.__name__} [Epoch {epoch}] 🔵 Validating'

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=desc, colour='blue'):
                y = batch[0].to(device=self.device, dtype=self.dtype)

                z = self.model.inverse(y)

                log_prob = self.prior_dist.log_prob(z)
                log_det_inv_jacobian = self.model.log_det_inv_jacobian(y)
                loss = torch.mean(- (log_prob + log_det_inv_jacobian))
                agg_val_loss += float(loss.item())
                agg_log_det += float(torch.mean(log_det_inv_jacobian).item())
                agg_log_prob += float(torch.mean(log_prob).item())

        n = max(1, len(self.test_loader))
        return agg_val_loss / n, agg_log_det / n, agg_log_prob / n

    def train(self):
        train_losses, val_losses, log_dets, log_probs = [], [], [], []

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, log_det, log_prob = self.validate_epoch(epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            log_dets.append(log_det)
            log_probs.append(log_prob)

            print(f'Epoch {epoch} - Train Loss: {train_loss:.3f}  |  Val Loss: {val_loss:.3f}')

        plot_val_loss(val_losses, log_dets, log_probs)

        if self.ckpt_path:
            os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.ckpt_path)


def train_normalizing_flow(config_path):
    file = '../../../nets/ieee118_186.mat'
    config = json.load(open(config_path))
    device = config.get('device', DEVICE)
    dtype = config.get('dtype', DTYPE)
    num_samples = config.get('num_samples', 100)
    batch_size = config.get('batch_size', BATCH_SIZE)
    cart = config.get('cart', False)
    data = parse_ieee_mat(file)
    system_data = data['data']['system']
    sys = System(system_data)

    train_loader, test_loader, train_dataset, test_dataset, mean, cov, std = load_data(sys, num_samples, batch_size, cart=cart)

    model = FlowModel(**config).to(device=device, dtype=dtype)

    # Load weights if present
    ckpt_path = f"./models/{config.get('ckpt_name')}"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    trainer = NormalizingFlowTrainer(model,
                                     (train_loader, test_loader, train_dataset, mean, cov, std),
                                    ckpt_path, **config)
    trainer.train()


if __name__ == "__main__":
    # for config_name in os.listdir("../configs_cart"):
    #     config_path = f'../configs_cart/{config_name}'
    config_path = f'../configs_cart/NF_4_0_128.json'
    train_normalizing_flow(config_path)