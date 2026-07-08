"""
Flow-matching prior network.

State convention (v0.6+)
------------------------
The FM prior operates on the **reduced** Cartesian state: the slack bus'
real and imaginary coordinates are removed before the network ever sees
the data, so samples have shape ``(2, nb-1)`` -- for IEEE-118 that is
``(2, 117)``, i.e. ``in_features = 234``. The slack voltage is fully
determined (fixed angle and magnitude), so it is re-inserted at decode
time by the caller (see ``optimizers/Flower_se.py``); modelling it would
only waste capacity on a degenerate coordinate. Training data is reduced
by ``load_data(..)`` when the config sets ``drop_slack: true``.

Network architecture
--------------------
Residual MLP with AdaLN-zero time conditioning and SwiGLU blocks:

    x_in   -> Linear -> hidden h_0
    t      -> sinusoidal Fourier embed -> 2-layer MLP -> t_emb
    h_k+1  =  h_k + gate_k * Block_k(h_k, t_emb)   (k = 0..N-1)
    v_hat  = Linear(mod(LayerNorm(h_N), t_emb))    (zero-init)

Each ``AdaLNZeroBlock`` is

    s, b, g <- Linear(t_emb).chunk(3)   (zero-init  =>  s = b = g = 0 at init)
    h       <- LayerNorm(x) * (1 + s) + b          (adaptive LayerNorm)
    a, c    <- Linear(h).chunk(2)
    h       <- SiLU(a) * c                          (SwiGLU)
    h       <- Dropout -> Linear(h)
    out     <- x + g * h                            (identity at init)

Zero-initialising the modulation (scale/shift/gate) and the final
projection is the AdaLN-zero recipe from DiT (Peebles & Xie 2023): the
network is exactly the identity flow at init, which stabilises training
of deep stacks. The SwiGLU inner MLP (Shazeer 2020) replaces the previous
plain Linear-SiLU-Linear block.

Public API (unchanged)
----------------------
* ``forward(y, t)``               -> velocity prediction, shape (B, in_features)
* ``run_flow_matching(y, inverse=False)`` -> integrate the ODE 0->1 (default)
  or 1->0 (``inverse=True``) with step ``self.delta_t``; the integrator is
  selected by ``ode_method`` ("euler" | "midpoint" | "heun").

Constructor kwargs (read from a config JSON or passed directly)
---------------------------------------------------------------
* ``input_dim``     int   data dimension                   (default DATA_DIM)
* ``hidden_dim``    int   block width                      (default HIDDEN_DIM)
* ``num_blocks`` /
  ``num_layers``    int   number of residual blocks        (default 8)
* ``t_embed_dim``   int   sinusoidal embed dim, must be even (default 128)
* ``mlp_ratio``     int   inner width = mlp_ratio * hidden  (default 4)
* ``activation``    str   "silu" | "gelu"                   (default "silu")
* ``dropout``       float in-block dropout                  (default 0.0)
* ``delta_t``       float integration step for run_flow_matching (default DELTA_T)
* ``ode_method``    str   "euler" | "midpoint" | "heun"     (default "euler")
"""
import os
import sys

# When run directly (`python FM.py -c ...`), make sure THIS bundled copy of
# `SE_torch` is imported, not another one on PYTHONPATH (same shim as
# analysis/run_experiments.py). Harmless when imported as a package module.
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PKG_ROOT in sys.path:
    sys.path.remove(_PKG_ROOT)
sys.path.insert(0, _PKG_ROOT)

import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm

from SE_torch.learn_prior.FM.load_data import load_data

DEVICE = 'mps'
DTYPE = torch.get_default_dtype()
DATA_DIM = 234   # reduced Cartesian state: 2 * (118 - 1), slack bus removed
NUM_LABELS = 5
EMBEDDING_DIM = 2
HIDDEN_DIM = 256

NUM_EPOCHS = 20
BATCH_SIZE = 128
NUM_DATA_POINTS = 250000
LEARNING_RATE = 1e-4
DELTA_T = 1e-2


_ACTIVATIONS = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


class AdaLNZeroBlock(nn.Module):
    """Residual SwiGLU block with AdaLN-zero time conditioning.

    The time embedding produces three per-sample vectors: ``scale`` and
    ``shift`` modulate the (affine-free) LayerNorm output, and ``gate``
    scales the residual branch. All three come from a zero-initialised
    Linear, so ``forward(x, t_emb) == x`` exactly at init (AdaLN-zero,
    Peebles & Xie 2023) -- which keeps deep stacks trainable. The inner
    MLP is a SwiGLU (Shazeer 2020): two parallel projections, one gated
    through the activation, which consistently outperforms the plain
    Linear-act-Linear block at equal parameter count.

    Note: unlike the previous block, the *output* Linear is NOT
    zero-initialised -- zeroing both the gate and the output would kill
    the gradient of each (their grads are mutually proportional).
    """

    def __init__(self, dim, t_emb_dim, mlp_ratio=4, dropout=0.0, activation="silu"):
        super().__init__()
        try:
            act_cls = _ACTIVATIONS[activation.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown activation {activation!r}. "
                f"Choose from {sorted(_ACTIVATIONS)}."
            )

        # LayerNorm without learnable affine: the affine is supplied by the
        # AdaLN (scale, shift) computed from t_emb.
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

        # AdaLN-zero conditioning: produces (scale, shift, gate) per sample.
        self.ada = nn.Linear(t_emb_dim, 3 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

        inner = int(mlp_ratio) * dim
        # SwiGLU: one projection for the activation branch, one for the
        # multiplicative branch, fused in a single Linear.
        self.fc_in = nn.Linear(dim, 2 * inner)
        self.act = act_cls()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc_out = nn.Linear(inner, dim)

    def forward(self, x, t_emb):
        scale, shift, gate = self.ada(t_emb).chunk(3, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        a, b = self.fc_in(h).chunk(2, dim=-1)
        h = self.act(a) * b
        h = self.drop(h)
        h = self.fc_out(h)
        return x + gate * h


class FlowMatching(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_features = int(kwargs.get("input_dim", DATA_DIM))
        self.hidden_dim  = int(kwargs.get("hidden_dim", HIDDEN_DIM))
        self.delta_t     = float(kwargs.get("delta_t", DELTA_T))

        # Either ``num_blocks`` (preferred) or the legacy ``num_layers`` key.
        n_blocks = kwargs.get("num_blocks", kwargs.get("num_layers", 8))
        self.num_blocks = int(n_blocks)
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1. Got {self.num_blocks}.")

        self.t_embed_dim = int(kwargs.get("t_embed_dim", 128))
        if self.t_embed_dim % 2 != 0:
            raise ValueError(f"t_embed_dim must be even. Got {self.t_embed_dim}.")

        self.mlp_ratio  = int(kwargs.get("mlp_ratio", 4))
        self.activation = str(kwargs.get("activation", "silu"))
        self.dropout    = float(kwargs.get("dropout", 0.0))
        self.t_max_freq = float(kwargs.get("t_max_freq", 1000.0))
        self.ode_method = str(kwargs.get("ode_method", "euler")).lower()
        if self.ode_method not in ("euler", "midpoint", "heun"):
            raise ValueError(
                f"ode_method must be 'euler', 'midpoint' or 'heun'. "
                f"Got {self.ode_method!r}."
            )

        # ---------- time embedding ----------
        half = self.t_embed_dim // 2
        # log-spaced frequencies: 1 .. t_max_freq.
        freqs = torch.exp(torch.linspace(0.0, math.log(self.t_max_freq), half))
        self.register_buffer("t_freqs", freqs)

        # 2-layer MLP that turns the Fourier features into a t_cond vector
        # of width ``hidden_dim``, which is what the FiLM blocks consume.
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_embed_dim, self.hidden_dim),
            _ACTIVATIONS[self.activation.lower()](),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # ---------- input / output projections ----------
        self.in_proj  = nn.Linear(self.in_features, self.hidden_dim)
        # Final layer follows DiT: time-modulated LayerNorm (zero-init
        # AdaLN, no gate needed here) followed by a zero-init projection,
        # so v_hat == 0 at initialisation (identity flow).
        self.out_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.out_ada = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        nn.init.zeros_(self.out_ada.weight)
        nn.init.zeros_(self.out_ada.bias)
        self.out_proj = nn.Linear(self.hidden_dim, self.in_features)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # ---------- residual stack ----------
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(
                dim=self.hidden_dim,
                t_emb_dim=self.hidden_dim,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                activation=self.activation,
            )
            for _ in range(self.num_blocks)
        ])

    # --------------------------------------------------------------- helpers

    def _format_t(self, y, t):
        """Coerce ``t`` to shape (B, 1) on the same device/dtype as ``y``,
        regardless of whether the caller passed a python scalar, a 0-D
        tensor, a (1,) tensor, a (B,) tensor, a (B, 1) tensor, or a
        (B, 1, 1) tensor (as in the trainer)."""
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=y.device, dtype=y.dtype)
        else:
            t = t.to(device=y.device, dtype=y.dtype)

        # Flatten any rank into a 1-D vector, then broadcast/expand to (B,)
        # and reshape to (B, 1).
        t = t.reshape(-1)
        if t.numel() == 1:
            t = t.expand(y.shape[0])
        elif t.numel() != y.shape[0]:
            raise ValueError(
                f"t has {t.numel()} elements, expected 1 or {y.shape[0]} "
                f"(=y.shape[0])."
            )
        return t.reshape(-1, 1)  # (B, 1)

    def _embed_t(self, t):
        """Sinusoidal Fourier embedding -> 2-layer MLP -> (B, hidden_dim)."""
        angles = t * self.t_freqs.view(1, -1) * (2.0 * math.pi)        # (B, half)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, t_embed_dim)
        return self.t_proj(emb)                                         # (B, hidden_dim)

    # ----------------------------------------------------------- public API

    def forward(self, y, t):
        """Velocity prediction v_theta(y, t).

        Parameters
        ----------
        y : Tensor of shape (B, ...), where the trailing dims flatten to
            ``in_features``. Typical shapes: (B, in_features) or
            (B, 2, F) with 2*F == in_features.
        t : python number, or tensor of any rank, broadcastable / expandable
            to (B,). Anything goes through ``_format_t``.

        Returns
        -------
        Tensor with the same shape as ``y``: v_hat preserves the input
        layout (so ``run_flow_matching`` can safely add ``dt * v_hat`` to
        ``y`` regardless of whether the caller works in flat or
        (channels, features) form).
        """
        B = y.shape[0]
        y_flat = y.reshape(B, -1)

        t_flat = self._format_t(y_flat, t)         # (B, 1)
        t_cond = self._embed_t(t_flat)             # (B, hidden_dim)

        h = self.in_proj(y_flat)
        for block in self.blocks:
            h = block(h, t_cond)
        scale, shift = self.out_ada(t_cond).chunk(2, dim=-1)
        h = self.out_norm(h) * (1.0 + scale) + shift
        return self.out_proj(h).reshape_as(y)

    def run_flow_matching(self, y, inverse=False):
        """Numerical integration of the learned ODE.

        ``inverse=False`` integrates from t=0 to t=1 (sample from prior),
        ``inverse=True`` integrates from t=1 to t=0 (push a data point
        back to noise space). Step size is ``self.delta_t``; the scheme is
        ``self.ode_method``:

          * "euler"    -- 1 model call/step (legacy behaviour).
          * "midpoint" -- 2 calls/step, 2nd order; markedly straighter
                          trajectories at equal step count.
          * "heun"     -- 2 calls/step, 2nd order (trapezoidal).
        """
        dt = -abs(self.delta_t) if inverse else abs(self.delta_t)
        t0 = 1.0 if inverse else 0.0

        def _clamp(t):
            # stay inside [0, 1]: the model was never trained outside it
            return max(0.0, min(1.0, t))

        # integer step count for floating-point stability
        n_steps = int(round(1.0 / abs(dt)))
        for k in range(n_steps):
            t_k = _clamp(t0 + k * dt)
            v1 = self.forward(y, t_k)
            if self.ode_method == "euler":
                y = y + dt * v1
            elif self.ode_method == "midpoint":
                y_mid = y + 0.5 * dt * v1
                y = y + dt * self.forward(y_mid, _clamp(t_k + 0.5 * dt))
            else:  # heun
                y_pred = y + dt * v1
                v2 = self.forward(y_pred, _clamp(t_k + dt))
                y = y + 0.5 * dt * (v1 + v2)
        return y


# --------------------------------------------------------------------- trainer


class FlowMatchingTrainer:
    def __init__(self, model, data_loaders, config):
        self.model = model
        self.train_loader, self.test_loader, self.train_dataset, self.test_dataset, _ = data_loaders

        self.num_epochs = config.get("num_epochs", NUM_EPOCHS)
        self.learning_rate = config.get("learning_rate", LEARNING_RATE)
        self.data_dim = config.get("in_features", DATA_DIM)
        self.device = torch.device(config.get('device', DEVICE))
        self.dtype = torch.float32 if config.get('dtype', DTYPE) == 'float32' else torch.float64

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        self.criterion = nn.MSELoss()
        self.config = config

        # ---- Checkpoint layout ----------------------------------------
        # `ckpt_dir` is the directory the trainer writes to (and reads
        # from on resume). Falls back to ./models so existing configs
        # that only set `ckpt_name` keep working.
        # `ckpt_prefix` is the basename used for the persistent
        # `_best.pth` / `_last.pth` / `_history.json` files; defaults
        # to the basename of `ckpt_name` without extension, or
        # "FlowMatching" if neither is set.
        ckpt_dir = config.get("ckpt_dir", "./models")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        ckpt_name = config.get("ckpt_name")
        if config.get("ckpt_prefix"):
            self.ckpt_prefix = str(config["ckpt_prefix"])
        elif ckpt_name:
            self.ckpt_prefix = os.path.splitext(os.path.basename(str(ckpt_name)))[0]
        else:
            self.ckpt_prefix = "FlowMatching"

        # ---- Resume bookkeeping ---------------------------------------
        # `resume_from`: either `None` (start fresh), `"auto"` (look for
        # `<ckpt_dir>/<ckpt_prefix>_last.pth` then `_best.pth`), or an
        # explicit checkpoint path.
        self.resume_from = config.get("resume_from", None)
        self.resume_strict = bool(config.get("resume_strict", True))
        self.resume_optim = bool(config.get("resume_optim", True))
        self.start_epoch = 1
        self.best_val = float("inf")
        self.resumed_history = {"train": [], "val": []}
        self._maybe_resume()

    # ------------------------------------------------------------------
    # Resume helpers (mirror the PIGNN / ProxLinear trainers so configs
    # behave the same across priors).
    # ------------------------------------------------------------------
    def _resolve_resume_path(self):
        rf = self.resume_from
        if rf is None:
            return None
        if rf == "auto":
            for tag in ("last", "best"):
                candidate = os.path.join(
                    self.ckpt_dir, f"{self.ckpt_prefix}_{tag}.pth"
                )
                if os.path.isfile(candidate):
                    return candidate
            return None
        return rf if os.path.isabs(rf) else os.path.abspath(rf)

    def _maybe_resume(self):
        path = self._resolve_resume_path()
        if path is None:
            return
        if not os.path.isfile(path):
            print(f"[FlowMatchingTrainer] resume_from={self.resume_from!r} "
                  f"-> {path} not found, starting fresh.")
            return

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        print(os.path.abspath(path))
        # Tolerate two formats: a bare state_dict (legacy), or the rich
        # dict written by `_save_ckpt()`.
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            self.model.load_state_dict(ckpt, strict=self.resume_strict)
            print(f"[FlowMatchingTrainer] resumed model weights from "
                  f"{path} (no optimizer/scheduler state available)")
            return

        self.model.load_state_dict(
            ckpt["state_dict"], strict=self.resume_strict
        )
        if self.resume_optim and "optim" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optim"])
            except Exception as e:  # noqa: BLE001
                print(f"[FlowMatchingTrainer] could not restore optim "
                      f"({e}); continuing with a fresh one.")
        if self.resume_optim and "scheduler" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:  # noqa: BLE001
                print(f"[FlowMatchingTrainer] could not restore "
                      f"scheduler state ({e}); continuing with a fresh "
                      f"one.")

        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.best_val = float(ckpt.get("best_val", float("inf")))
        saved_history = ckpt.get("history")
        if isinstance(saved_history, dict):
            self.resumed_history = {
                "train": list(saved_history.get("train", [])),
                "val":   list(saved_history.get("val",   [])),
            }
        print(
            f"[FlowMatchingTrainer] resumed from {path} | "
            f"start_epoch={self.start_epoch} | "
            f"best_val={self.best_val:.4e}"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_ckpt(self, epoch, val_loss, history, best_val, tag):
        path = os.path.join(
            self.ckpt_dir, f"{self.ckpt_prefix}_{tag}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_val": best_val,
                "val_loss": val_loss,
                "history": history,
                "config": self.config,
            },
            path,
        )

    def _save_history_json(self, history, best_val):
        path = os.path.join(
            self.ckpt_dir, f"{self.ckpt_prefix}_history.json"
        )
        with open(path, "w") as f:
            json.dump(
                {"best_val": best_val, **history},
                f, indent=2, default=str,
            )

    def train_eval_batch(self, batch):
        y1 = batch.to(self.device, self.dtype)
        y0 = torch.randn_like(y1).to(self.device, self.dtype)
        t = torch.rand((y1.shape[0], 1, 1)).to(self.device, self.dtype)
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
        pbar = tqdm(self.train_loader, desc=desc, colour='green')
        for batch in pbar:
            self.optimizer.zero_grad()
            loss = self.train_eval_batch(batch)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        self.scheduler.step()

        return running_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        self.model.eval()
        agg_val_loss = 0.0
        desc = f"FlowMatching [Epoch {epoch}] 🔵 Validating"

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=desc, colour='blue')
            for batch in pbar:
                loss = self.train_eval_batch(batch)
                agg_val_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        return agg_val_loss / len(self.test_loader)

    def train(self):
        # Continue any history loaded by `_maybe_resume()` so the
        # JSON / checkpoint records carry across resumes.
        history = {
            "train": list(self.resumed_history.get("train", [])),
            "val":   list(self.resumed_history.get("val",   [])),
        }
        best_val = self.best_val

        start = self.start_epoch
        end = self.num_epochs
        if start > end:
            print(
                f"[FlowMatchingTrainer] resumed start_epoch={start} is "
                f"past num_epochs={end}; nothing to do. Bump "
                f"num_epochs in the config to continue training."
            )
            return

        for epoch in range(start, end + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            history["train"].append({"epoch": epoch, "loss": train_loss})
            history["val"].append({"epoch": epoch, "loss": val_loss})

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                self._save_ckpt(epoch, val_loss, history, best_val,
                                 tag="best")

            # Always write a `_last.pth` so a crash at any point is
            # recoverable, and refresh the history JSON.
            self._save_ckpt(epoch, val_loss, history, best_val, tag="last")
            self._save_history_json(history, best_val)

            tag = "  *new best*" if is_best else ""
            print(
                f"[Epoch {epoch:>3}] | Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.4f} - best={best_val:.5f}{tag} | "
                f"lr: {self.scheduler.get_last_lr()[0]:.3e}"
            )

        # Final train-batch validation (useful for diagnostics) and
        # legacy-format checkpoint if the caller still wants one at
        # the original `./models/<ckpt_name>` location.
        self.validate_train()
        ckpt_name = self.config.get("ckpt_name")
        if ckpt_name:
            legacy_path = f"./models/{ckpt_name}"
            os.makedirs(os.path.dirname(legacy_path) or ".", exist_ok=True)
            torch.save(self.model.state_dict(), legacy_path)

        self.best_val = best_val
        return history


def train_flow_matching(config_path, **overrides):
    """Build the FM model, wrap it in a trainer, and run training.

    `overrides` are merged into the config (keys take precedence over
    the JSON values) so the CLI can forward `--resume`, `--epochs`,
    `--device`, etc. without mutating the original JSON.
    """
    config = json.load(open(config_path))
    if overrides:
        # Drop None-valued overrides so the JSON defaults still apply
        # when the caller didn't set a flag.
        for k, v in overrides.items():
            if v is not None:
                config[k] = v

    device = torch.device(config.get('device', DEVICE))
    dtype = config.get('dtype', DTYPE)
    dtype = torch.float32 if dtype == "float32" else torch.float64
    num_samples = config.get('num_samples', None)
    cart = config.get('cart', False)
    data = load_data(config, num_samples, cart=cart)

    model = FlowMatching(**config).to(device=device, dtype=dtype)
    # NOTE: the trainer now owns checkpoint loading. The previous
    # "always-load `<ckpt_dir>/<ckpt_name>` if it exists" behaviour
    # silently turned every run into a continue-from-weights run with
    # a reset epoch counter; if you want that behaviour now, pass
    # `resume_from="auto"` in the config (or `--resume` on the CLI).

    trainer = FlowMatchingTrainer(model, data, config)
    return trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c",
        default=os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "configs", "FM_v0.3_config.json",
        )),
        help="Path to the ProxLinear config JSON.",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.num_epochs.")
    parser.add_argument("--n_train_samples", type=int, default=None,
                        help="Override training.n_train_samples.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override training.device.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick sanity run: 2 epochs on 64 training samples, "
             "batch size 16 -- ignores the config's epoch / sample / "
             "batch settings.",
    )
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help="Resume from a checkpoint; `--resume` alone auto-picks "
             "`<ckpt_dir>/<ckpt_prefix>_last.pth`.",
    )
    parser.add_argument(
        "--resume-fresh-optim", action="store_true",
        help="Resume only model weights; start with a fresh optimizer.",
    )
    parser.add_argument(
        "--resume-nonstrict", action="store_true",
        help="Allow resumed `state_dict` keys to mismatch the current model.",
    )
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)

    # Translate CLI flags into config overrides forwarded by
    # `train_flow_matching`. The trainer's resume hooks consume
    # `resume_from` (None | "auto" | <path>) and the
    # `resume_strict` / `resume_optim` booleans.
    overrides = {}
    if args.epochs is not None:
        overrides["num_epochs"] = int(args.epochs)
    if args.n_train_samples is not None:
        overrides["num_samples"] = int(args.n_train_samples)
    if args.device is not None:
        overrides["device"] = args.device
    if args.resume is not None:
        overrides["resume_from"] = args.resume
    if args.resume_fresh_optim:
        overrides["resume_optim"] = False
    if args.resume_nonstrict:
        overrides["resume_strict"] = False

    train_flow_matching(config_path, **overrides)

if __name__ == '__main__':
    main()
