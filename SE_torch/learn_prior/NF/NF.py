import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from SE_torch.learn_prior.NF.load_data import load_data


DEVICE = "mps"
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
DATA_DIM = 235
HALF_DATA_DIM = 118
CHANNELS = 2
NUM_EPOCHS = 20
S_MAX = 1.
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
BETAS = (0.9, 0.99)
NUM_DATA_POINTS = 250000
HIDDEN_DIM = 8
NUM_HIDDEN_LAYERS = 0
NUM_BLOCKS = 2


class AffineCouplingLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        half_data_dim = kwargs.get('half_data_dim', HALF_DATA_DIM)
        hidden_dim = kwargs.get('hidden_dim',HIDDEN_DIM)
        num_hidden_layers = kwargs.get('num_hidden_layers', NUM_HIDDEN_LAYERS)

        self.log_s = nn.Sequential(nn.Linear(half_data_dim, hidden_dim), nn.LeakyReLU())
        for _ in range(num_hidden_layers):
            self.log_s.append(nn.Linear(hidden_dim, hidden_dim))
            self.log_s.append(nn.LeakyReLU())
        self.log_s.append(nn.Linear(hidden_dim, half_data_dim - 1))

        self.b = nn.Sequential(nn.Linear(half_data_dim, hidden_dim), nn.LeakyReLU())
        for _ in range(num_hidden_layers):
            self.b.append(nn.Linear(hidden_dim, hidden_dim))
            self.b.append(nn.LeakyReLU())
        self.b.append(nn.Linear(hidden_dim, half_data_dim - 1))

    def forward(self, z):
        z_l, z_r = z.chunk(2, dim=1)
        log_s = self.log_s(z_l)
        log_s = S_MAX * torch.tanh(log_s)
        s = torch.exp(log_s)
        b = self.b(z_l)
        y_l = z_l
        y_r = s * z_r + b
        return torch.cat([y_l, y_r], dim=1)

    def inverse(self, y):
        y_l, y_r = y.chunk(2, dim=1)
        log_s = self.log_s(y_l)
        log_s = S_MAX * torch.tanh(log_s)
        s = torch.exp(log_s)
        b = self.b(y_l)
        z_l = y_l
        z_r = (y_r - b) / s
        return torch.cat([z_l, z_r], dim=1)

    def log_det_inv_jacobian(self, y):
        y_l, _ = y.chunk(2, dim=1)
        log_s = self.log_s(y_l)
        log_s = S_MAX * torch.tanh(log_s)
        # Sum over features; return [B]
        return torch.sum(-log_s, dim=1)

    def log_det_jacobian(self, z):
        z_l, _ = z.chunk(2, dim=1)
        log_s = self.log_s(z_l)
        log_s = S_MAX * torch.tanh(log_s)
        # Sum over features; return [B]
        return torch.sum(log_s, dim=1)

    def log_det_inv_jacobian_residuals(self, y):
        y_l, _ = y.chunk(2, dim=1)
        log_s = self.log_s(y_l)
        log_s = S_MAX * torch.tanh(log_s)
        return torch.cat([torch.zeros_like(y_l), -log_s], dim=1)


class PermutationalLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        data_dim = kwargs.get('data_dim', DATA_DIM)
        perm = torch.randperm(data_dim)
        while torch.equal(perm, torch.arange(data_dim)):
            perm = torch.randperm(data_dim)
        inv_perm = torch.argsort(perm)

        perm = torch.eye(data_dim)[perm]
        inv_perm = torch.eye(data_dim)[inv_perm]
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, z):
        return z @ self.perm.T

    def inverse(self, y):
        return y @ self.inv_perm.T


class FlowModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        num_blocks = kwargs.get('num_blocks', NUM_BLOCKS)
        self.ac_layers = nn.ModuleList([AffineCouplingLayer(**kwargs) for _ in range(num_blocks)])
        self.perm_layers = nn.ModuleList([PermutationalLayer(**kwargs) for _ in range(num_blocks)])

    def forward(self, z):
        for ac_layer, perm_layer in zip(self.ac_layers, self.perm_layers):
            z = ac_layer(perm_layer(z))
        return z

    def inverse(self, y):
        for ac_layer, perm_layer in reversed(list(zip(self.ac_layers, self.perm_layers))):
            y = perm_layer.inverse(ac_layer.inverse(y))
        return y

    def log_det_inv_jacobian(self, y):
        log_det = torch.zeros(y.shape[0])
        for ac_layer, perm_layer in reversed(list(zip(self.ac_layers, self.perm_layers))):
            log_det = log_det + ac_layer.log_det_inv_jacobian(y)
            y = perm_layer.inverse(ac_layer.inverse(y))
        return log_det

    def log_det_jacobian(self, z):
        log_det = torch.zeros(z.shape[0])
        for ac_layer, perm_layer in zip(self.ac_layers, self.perm_layers):
            z = perm_layer(z)
            log_det = log_det + ac_layer.log_det_jacobian(z)
            z = ac_layer(z)
        return log_det

    def log_det_inv_jacobian_residuals(self, y):
        log_det = torch.zeros_like(y)
        for ac_layer, perm_layer in reversed(list(zip(self.ac_layers, self.perm_layers))):
            log_det = perm_layer.inverse(log_det) + ac_layer.log_det_inv_jacobian_residuals(y)
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
    plt.legend(); plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('./plots/flow_validation_loss.png')
    plt.close()


class NormalizingFlowTrainer:
    def __init__(self, model, data_loaders, num_epochs=NUM_EPOCHS,
                 learning_rate=LEARNING_RATE, ckpt_path='', device='cpu', dtype=torch.get_default_dtype(), weight_decay=WEIGHT_DECAY, betas=BETAS):
        self.model = model
        self.train_loader, self.test_loader, self.train_dataset, self.test_dataset, self.config = data_loaders

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        mean = torch.zeros(DATA_DIM).to(device=device, dtype=dtype)
        cov = torch.eye(DATA_DIM).to(device=device, dtype=dtype)
        self.prior_dist = MultivariateNormal(mean, cov)
        self.num_epochs = num_epochs
        self.ckpt_path = ckpt_path
        self.device = device
        self.dtype = dtype

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        desc = f'{self.model.__class__.__name__} [Epoch {epoch}] 🟢 Training'

        for batch in tqdm(self.train_loader, desc=desc, colour='green'):
            y = batch[0].to(device=self.device, dtype=self.dtype, non_blocking=False)
            self.optimizer.zero_grad(set_to_none=True)
            z = self.model.inverse(y)
            log_prob = self.prior_dist.log_prob(z)
            log_det_inv_jacobian = self.model.log_det_inv_jacobian(y)
            loss = torch.mean(- (log_prob + log_det_inv_jacobian))
            loss.backward()
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
                y = batch[0].to(device=self.device, dtype=self.dtype, non_blocking=False)
                z = self.model.inverse(y)
                log_prob = self.prior_dist.log_prob(z)
                log_det_inv_jacobian = self.model.log_det_inv_jacobian(y)
                loss = torch.mean(- (log_prob + log_det_inv_jacobian))
                agg_val_loss += float(loss.item())
                agg_log_det  += float(torch.mean(log_det_inv_jacobian).item())
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
    config = json.load(open(config_path))
    device = config.get('device', DEVICE)
    dtype = config.get('dtype', DTYPE)
    num_samples = config.get('num_samples', 10000)

    data = load_data(config, num_samples)

    model = FlowModel(**config).to(device=device, dtype=dtype)

    ckpt_path = f"./models/{config.get('ckpt_name')}"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    trainer = NormalizingFlowTrainer(model, data, ckpt_path=ckpt_path, device=device, dtype=dtype)
    trainer.train()


def main():
    config_name = "NF_4_0_235.json"
    config_path = f'../configs/{config_name}'
    train_normalizing_flow(config_path)


if __name__ == "__main__":
    main()