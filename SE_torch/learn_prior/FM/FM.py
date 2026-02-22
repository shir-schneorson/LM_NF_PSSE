import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from tqdm.auto import tqdm

from SE_torch.learn_prior.NF.load_data import load_data

DEVICE = 'mps'
DTYPE = torch.get_default_dtype()
DATA_DIM = 236
NUM_LABELS = 5
EMBEDDING_DIM = 2
HIDDEN_DIM = 64

NUM_EPOCHS = 20
BATCH_SIZE = 128
NUM_DATA_POINTS = 250000
LEARNING_RATE = 1e-5
DELTA_T = 1e-2


class FlowMatching(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_features = kwargs.get("input_dim", DATA_DIM)
        self.hidden_dim  = kwargs.get("hidden_dim", HIDDEN_DIM)
        self.delta_t     = kwargs.get("delta_t", DELTA_T)

        # ----- time embedding -----
        self.t_embed_dim = kwargs.get("t_embed_dim", 32)   # even
        if self.t_embed_dim % 2 != 0:
            raise ValueError(f"t_embed_dim must be even. Got {self.t_embed_dim}.")

        half = self.t_embed_dim // 2
        freqs = torch.exp(
            torch.linspace(torch.log(torch.tensor(1.0)), torch.log(torch.tensor(1000.0)), half)
        )
        self.register_buffer("t_freqs", freqs)

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # ----- MLP with configurable depth -----
        self.num_layers = int(kwargs.get("num_layers", 16*8))  # total linear layers incl. output
        if self.num_layers < 2:
            raise ValueError(f"num_layers must be >= 2 (1 hidden + 1 output). Got {self.num_layers}.")

        layers = []
        in_dim0 = self.in_features + self.hidden_dim

        # first layer
        layers.append(nn.Linear(in_dim0, self.hidden_dim))
        # middle hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        # output layer
        layers.append(nn.Linear(self.hidden_dim, self.in_features))

        self.net = nn.ModuleList(layers)

    def _format_t(self, y, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=y.device, dtype=y.dtype)
        else:
            t = t.to(device=y.device, dtype=y.dtype)

        if t.ndim == 0:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t.view(-1, 1)

        if t.shape[0] == 1 and y.shape[0] > 1:
            t = t.expand(y.shape[0], 1)
        elif t.shape[0] != y.shape[0]:
            raise ValueError(f"t batch size {t.shape[0]} != y batch size {y.shape[0]}")
        return t  # (B,1)

    def _embed_t(self, t):
        angles = t * self.t_freqs.view(1, -1) * (2.0 * torch.pi)  # (B, half)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (B, t_embed_dim)
        return self.t_proj(emb)  # (B, hidden_dim)

    def forward(self, y, t):
        t = self._format_t(y, t)
        t_emb = self._embed_t(t)
        x = torch.cat([y, t_emb], dim=1)

        # leaky_relu on all but last layer
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i != len(self.net) - 1:
                x = F.leaky_relu(x)
        return x

    def run_flow_matching(self, y, inverse=False):
        dt = -abs(self.delta_t) if inverse else abs(self.delta_t)
        t = 1.0 if inverse else 0.0

        # avoid rounding; use integer steps for stability
        n_steps = int(round(1.0 / abs(dt)))
        for k in range(n_steps + 1):
            t_k = t + k * dt
            if t_k < 0.0 - 1e-9 or t_k > 1.0 + 1e-9:
                continue
            y = y + dt * self.forward(y, t_k)

        return y


class FlowMatchingTrainer:
    def __init__(self, model, data_loaders, config):
        self.model = model
        self.train_loader, self.test_loader, self.train_dataset, self.test_dataset, _ = data_loaders

        self.num_epochs = config.get("num_epochs", NUM_EPOCHS)
        self.learning_rate = config.get("learning_rate", LEARNING_RATE)
        self.data_dim = config.get("in_features", DATA_DIM)
        self.device = config.get('device', DEVICE)
        self.dtype = config.get('dtype', DTYPE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        self.prior_dist = MultivariateNormal(torch.zeros(self.data_dim), torch.eye(self.data_dim))
        self.criterion = nn.MSELoss()
        self.config = config

    def train_eval_batch(self, batch):
        y1 = batch[0].to(self.device, self.dtype)
        y0 = torch.randn_like(y1).to(self.device, self.dtype)
        t = torch.rand(y1.shape[0], 1).to(self.device, self.dtype)
        yt = t * y1 + (1 - t) * y0
        v_t = y1 - y0
        v_t_hat = self.model(yt, t)
        loss = self.criterion(v_t_hat, v_t)
        return loss

    def validate_train(self):
        self.model.eval()
        batch_losses = []
        desc = f"FlowMatching 🔵 Train Batch Validation"

        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc=desc, colour='blue'):
                loss = self.train_eval_batch(batch)
                batch_losses.append(loss.item())

        return batch_losses

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        desc = f"FlowMatching [Epoch {epoch}] 🟢 Training"

        for batch in tqdm(self.train_loader, desc=desc, colour='green'):
            self.optimizer.zero_grad()
            loss = self.train_eval_batch(batch)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        self.scheduler.step()

        return running_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        self.model.eval()
        agg_val_loss = 0.0
        desc = f"FlowMatching [Epoch {epoch}] 🔵 Validating"

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=desc, colour='blue'):
                loss = self.train_eval_batch(batch)
                agg_val_loss += loss.item()

        return agg_val_loss / len(self.test_loader)

    def train(self):

        train_losses = []
        val_losses = []

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)

            val_loss = self.validate_epoch(epoch)
            val_losses.append(val_loss)

            print(f'Epoch {epoch} - Train Loss: {train_loss:.3f} - Val Loss: {val_loss:.3f}')

        train_batchs_loss = self.validate_train()

        os.makedirs('./models', exist_ok=True)
        ckpt_path = f"./models/{self.config.get('ckpt_name')}"

        torch.save(self.model.state_dict(), ckpt_path)


def train_flow_matching(config_path):
    config = json.load(open(config_path))
    device = config.get('device', DEVICE)
    dtype = config.get('dtype', DTYPE)
    num_samples = config.get('num_samples', 10000)

    data = load_data(config, num_samples)

    model = FlowMatching(**config).to(device=device, dtype=dtype)

    ckpt_path = f"./models/{config.get('ckpt_name')}"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    trainer = FlowMatchingTrainer(model, data, config)
    trainer.train()


def main():
    config_name = "FM_v0.2_config.json"
    config_path = f'../configs/{config_name}'
    train_flow_matching(config_path)

if __name__ == '__main__':
    main()
