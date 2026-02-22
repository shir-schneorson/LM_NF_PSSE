import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from SE_torch.learn_prior.NF.load_data import load_data

# -----------------
# Globals / defaults
# -----------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

NUM_EPOCHS = 30
LR_G = 1e-4
LR_D = 1e-4
BETAS = (0.0, 0.9)     # common for WGAN-GP
N_CRITIC = 5
LAMBDA_GP = 10.0

IN_FEATURES = 235
LATENT_DIM = 200
NUM_DATA_POINTS = 25000


# -----------------
# Simple MLP blocks
# -----------------
def make_mlp(in_dim, hidden_dims, out_dim, act="leaky_relu", last_act=None, dropout=0.0):
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        if act == "relu":
            layers.append(nn.ReLU())
        elif act == "tanh":
            layers.append(nn.Tanh())
        else:
            layers.append(nn.LeakyReLU(1e-2))
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    if last_act == "tanh":
        layers.append(nn.Tanh())
    elif last_act == "relu":
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# -----------------
# GAN: Generator/Critic
# -----------------
class Generator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", DEVICE)
        self.dtype = kwargs.get("dtype", torch.get_default_dtype())
        self.z_dim = kwargs.get("latent_dim", LATENT_DIM)
        self.x_dim = kwargs.get("in_features", IN_FEATURES)

        hidden_dims = kwargs.get("g_hidden_dims", [512, 512, 256])
        dropout = kwargs.get("g_dropout", 0.0)
        act = kwargs.get("g_activation", "leaky_relu")

        # Note: no final nonlinearity by default (data may be unbounded after standardization)
        self.net = make_mlp(self.z_dim, hidden_dims, self.x_dim, act=act, last_act=None, dropout=dropout)

        if self.device is not None:
            self.to(self.device, dtype=self.dtype)

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", DEVICE)
        self.dtype = kwargs.get("dtype", torch.get_default_dtype())
        self.x_dim = kwargs.get("in_features", IN_FEATURES)

        hidden_dims = kwargs.get("d_hidden_dims", [512, 512, 256])
        dropout = kwargs.get("d_dropout", 0.0)
        act = kwargs.get("d_activation", "leaky_relu")

        self.net = make_mlp(self.x_dim, hidden_dims, 1, act=act, last_act=None, dropout=dropout)

        if self.device is not None:
            self.to(self.device, dtype=self.dtype)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)


# -----------------
# WGAN-GP utilities
# -----------------
@torch.no_grad()
def sample_z(batch_size, z_dim, device, dtype):
    return torch.randn(batch_size, z_dim, device=device, dtype=dtype)

def gradient_penalty(critic, real, fake, device, dtype, lambda_gp=LAMBDA_GP):
    # real,fake: (B, x_dim)
    B = real.size(0)
    alpha = torch.rand(B, 1, device=device, dtype=dtype)
    x_hat = alpha * real + (1 - alpha) * fake
    x_hat.requires_grad_(True)

    d_hat = critic(x_hat)  # (B,)
    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # (B, x_dim)

    grad_norm = grads.view(B, -1).norm(2, dim=1)  # (B,)
    gp = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
    return gp


# -----------------
# Training config
# -----------------
class TrainingConfig:
    def __init__(self, **kwargs):
        self.num_epochs = kwargs.get("num_epochs", NUM_EPOCHS)
        self.lr_g = kwargs.get("lr_g", LR_G)
        self.lr_d = kwargs.get("lr_d", LR_D)
        self.betas = kwargs.get("betas", BETAS)
        self.n_critic = kwargs.get("n_critic", N_CRITIC)
        self.lambda_gp = kwargs.get("lambda_gp", LAMBDA_GP)

        self.ckpt_name = kwargs.get("ckpt_name", "gan_ckpt")
        self.device = kwargs.get("device", DEVICE)
        self.dtype = kwargs.get("dtype", DTYPE)

        self.latent_dim = kwargs.get("latent_dim", LATENT_DIM)
        self.in_features = kwargs.get("in_features", IN_FEATURES)


# -----------------
# Trainer
# -----------------
class GANTrainer:
    def __init__(self, G, D, config, data_loaders):
        self.G = G
        self.D = D
        self.cfg = config
        self.train_loader, self.test_loader, self.train_dataset, self.test_dataset, _ = data_loaders

        self.opt_g = optim.Adam(self.G.parameters(), lr=self.cfg.lr_g, betas=self.cfg.betas)
        self.opt_d = optim.Adam(self.D.parameters(), lr=self.cfg.lr_d, betas=self.cfg.betas)

    def train(self):
        os.makedirs("./models", exist_ok=True)

        for epoch in range(1, self.cfg.num_epochs + 1):
            self.G.train()
            self.D.train()

            d_loss_running = 0.0
            g_loss_running = 0.0

            desc = f"WGAN-GP [Epoch {epoch}/{self.cfg.num_epochs}]"
            for batch in tqdm(self.train_loader, desc=desc):
                real = batch[0].to(self.cfg.device, dtype=self.cfg.dtype)
                B = real.size(0)

                # -------------------------
                # 1) Train Critic n_critic
                # -------------------------
                for _ in range(self.cfg.n_critic):
                    z = torch.randn(B, self.cfg.latent_dim, device=self.cfg.device, dtype=self.cfg.dtype)
                    fake = self.G(z).detach()

                    d_real = self.D(real).mean()
                    d_fake = self.D(fake).mean()
                    gp = gradient_penalty(self.D, real, fake, self.cfg.device, self.cfg.dtype, self.cfg.lambda_gp)

                    d_loss = (d_fake - d_real) + gp

                    self.opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), 10.0)
                    self.opt_d.step()

                # -------------------------
                # 2) Train Generator
                # -------------------------
                z = torch.randn(B, self.cfg.latent_dim, device=self.cfg.device, dtype=self.cfg.dtype)
                fake = self.G(z)
                g_loss = -self.D(fake).mean()

                self.opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10.0)
                self.opt_g.step()

                d_loss_running += float(d_loss.item())
                g_loss_running += float(g_loss.item())

            d_loss_epoch = d_loss_running / max(1, len(self.train_loader))
            g_loss_epoch = g_loss_running / max(1, len(self.train_loader))

            mean_err, std_err, cov_err = eval_moments(self.G, self.test_loader, self.cfg.latent_dim, self.cfg.device,
                                                      self.cfg.dtype)
            print(f"Epoch {epoch}: mean_err={mean_err:.4f} std_err={std_err:.4f} cov_err={cov_err:.4f}")
            print(f"Epoch {epoch}: D_loss={d_loss_epoch:.6f} | G_loss={g_loss_epoch:.6f}")

            # Save each epoch (or change to save best)
            self.save(epoch)

    def save(self, epoch=None):
        base = self.cfg.ckpt_name
        if epoch is None:
            g_path = f"./models/{base}_G.pt"
            d_path = f"./models/{base}_D.pt"
        else:
            g_path = f"./models/{base}_G_e{epoch}.pt"
            d_path = f"./models/{base}_D_e{epoch}.pt"

        torch.save(self.G.state_dict(), g_path)
        torch.save(self.D.state_dict(), d_path)


# -----------------
# Main
# -----------------
def main(config_path):
    config = json.load(open(config_path))

    num_samples = config.get("num_samples", NUM_DATA_POINTS)
    device = config.get("device", DEVICE)
    dtype = config.get("dtype", DTYPE)

    # Load data (kept as-is)
    data = load_data(config, num_samples)

    # Build GAN
    G = Generator(**config).to(device, dtype)
    D = Critic(**config).to(device, dtype)

    train_cfg = TrainingConfig(**config)

    # Load checkpoints if exist
    base = config.get("ckpt_name", "gan_ckpt")
    g_path = f"./models/{base}_G_e30.pt"
    d_path = f"./models/{base}_D_e30.pt"
    if os.path.exists(g_path) and os.path.exists(d_path):
        G.load_state_dict(torch.load(g_path, map_location=device))
        D.load_state_dict(torch.load(d_path, map_location=device))
        print(f"Loaded checkpoints: {g_path}, {d_path}")

    trainer = GANTrainer(G, D, train_cfg, data)
    trainer.train()


@torch.no_grad()
def eval_moments(G, loader, z_dim, device, dtype, num_batches=20):
    G.eval()
    real_list, fake_list = [], []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        real = batch[0].to(device, dtype=dtype)
        z = torch.randn(real.size(0), z_dim, device=device, dtype=dtype)
        fake = G(z)
        real_list.append(real)
        fake_list.append(fake)

    real = torch.cat(real_list, dim=0)
    fake = torch.cat(fake_list, dim=0)

    m_r, s_r = real.mean(0), real.std(0)
    m_f, s_f = fake.mean(0), fake.std(0)

    mean_err = (m_f - m_r).pow(2).mean().sqrt().item()
    std_err  = (s_f - s_r).pow(2).mean().sqrt().item()

    # covariance (Frobenius RMS)
    real_c = real - m_r
    fake_c = fake - m_f
    Cov_r = (real_c.T @ real_c) / (real.size(0) - 1)
    Cov_f = (fake_c.T @ fake_c) / (fake.size(0) - 1)
    cov_err = (Cov_f - Cov_r).pow(2).mean().sqrt().item()

    return mean_err, std_err, cov_err

if __name__ == "__main__":
    for config_name in os.listdir("../configs"):
        if config_name.startswith("GAN_v0.3"):
            print(config_name)
            config_pth = f"../configs/{config_name}"
            main(config_pth)