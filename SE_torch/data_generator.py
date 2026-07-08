"""Synthetic PSSE data generation.

`DataGenerator` produces (a) ground-truth voltage states -- either sampled from a
precomputed operating-point pool or solved from load/generation via
Newton-Raphson power flow -- and (b) noisy AC measurements at a chosen
observability level. `generate_measurements` returns the measurement vector `z`,
its per-entry variances, the measurement index masks, the cartesian/polar
measurement models, and the ground-truth state. Observability is controlled by
per-type Bernoulli subsampling (`sample`); noise variances are type-dependent.
"""

import os

import torch
import pandas as pd
from tqdm import tqdm

from SE_torch.utils import init_start_point
from SE_torch.optimizers.NR_acpf import NR_PF
from SE_torch.PF_equations.PF_polar import H_AC as H_AC_polar
from SE_torch.PF_equations.PF_cartesian import H_AC as H_AC_cartesian


def aggregate_meas_idx(meas_idx: dict, meas_types: list[str], device=None):
    agg_meas_idx = {}
    last_idx = 0

    for k in meas_types:
        mask = meas_idx.get(f"{k}_idx", None)
        if mask is None:
            v = torch.tensor([], dtype=torch.long, device=device)
        else:
            mask_t = (
                mask.to(device=device)
                if isinstance(mask, torch.Tensor)
                else torch.as_tensor(mask, dtype=torch.bool, device=device)
            )
            v = torch.nonzero(mask_t, as_tuple=False).view(-1).to(torch.long)
        agg_meas_idx[k] = torch.arange(
            last_idx, last_idx + v.numel(), dtype=torch.long, device=device
        )
        last_idx += v.numel()

    return agg_meas_idx

def regenerate_PQ_numeric_types(
    sys,
    seed: int | None = None,
    load_sigma_frac: float = 0.12,
    pf_mean_by_type_id: dict | None = None,
    pf_sigma: float = 0.02,
    gen_noise_frac: float = 0.08,
    device: str | torch.device | None = None,
    random_generator: torch.Generator = None
):

    bus = sys.bus.copy()
    nb = sys.nb
    assert len(bus) == nb, "sys.nb and sys.bus length mismatch"

    Pbase_L = torch.as_tensor(bus['Pl'].fillna(0.0).values, device=device ,dtype=torch.get_default_dtype())
    if not torch.any(Pbase_L > 0):
        Pbase_L = torch.ones(nb, device=device)

    dP = torch.normal(mean=0.0, std=load_sigma_frac, size=(nb,), device=device, dtype=torch.get_default_dtype(), generator=random_generator)
    P_L = torch.clamp(Pbase_L * (1.0 + dP), min=0.0)
    if pf_mean_by_type_id is None:
        pf_mean_by_type_id = {
            1: 0.95,  # PQ
            2: 0.98,  # PV
            3: 0.99   # Slack
        }
    types_id = torch.as_tensor(bus['bus_type'].to_numpy(int), device=device)
    pf_nom = torch.empty(nb, device=device)
    for tval, mean_pf in pf_mean_by_type_id.items():
        pf_nom[types_id == int(tval)] = float(mean_pf)
    pf_nom[(pf_nom != pf_nom)] = 0.95
    min_pf = min(list(pf_mean_by_type_id.values()))
    max_pf = max(list(pf_mean_by_type_id.values()))
    pf_L = torch.normal(mean=pf_nom, std=pf_sigma, generator=random_generator)
    pf_L = torch.clamp(pf_L, min=min_pf, max=max_pf)

    Q_L = P_L * torch.tan(torch.arccos(pf_L))

    Qmin = torch.as_tensor(bus['Qmin'].to_numpy(float), device=device, dtype=torch.get_default_dtype())
    Qmax = torch.as_tensor(bus['Qmax'].to_numpy(float), device=device, dtype=torch.get_default_dtype())
    prev_Pg = torch.as_tensor(bus['Pg'].fillna(0.0).to_numpy(float), device=device, dtype=torch.get_default_dtype())

    finite_min = torch.isfinite(Qmin)
    finite_max = torch.isfinite(Qmax)
    gen_mask = (finite_min & finite_max) | (prev_Pg > 0)
    gen_idx = torch.nonzero(gen_mask, as_tuple=True)[0]

    Pg = torch.zeros(nb, device=device, dtype=torch.get_default_dtype())
    if gen_idx.numel() > 0:
        w = torch.clamp(prev_Pg[gen_idx], min=0.0)
        if w.sum() <= 1e-9:
            w = torch.ones_like(w)

        P_target = P_L.sum()
        Pg_raw = P_target * (w / w.sum())

        Pg_noise = torch.normal(mean=0.0, std=gen_noise_frac, size=Pg_raw.shape, device=device, generator=random_generator)
        Pg_gen = torch.clamp(Pg_raw + Pg_raw * Pg_noise, min=0.0)

        if Pg_gen.sum() > 1e-9:
            Pg_gen = Pg_gen * (P_target / Pg_gen.sum())

        Pg[gen_idx] = Pg_gen

    Qmin_eff = torch.where(gen_mask, torch.nan_to_num(Qmin, nan=0.0), torch.zeros_like(Qmin))
    Qmax_eff = torch.where(gen_mask, torch.nan_to_num(Qmax, nan=0.0), torch.zeros_like(Qmax))

    Qg = torch.zeros(nb, device=device)
    has_range = (Qmax_eff > Qmin_eff)
    mid = 0.5 * (Qmin_eff + Qmax_eff)
    Qg[has_range] = mid[has_range]

    Q_def = Q_L.sum() - Qg.sum()
    if torch.abs(Q_def) > 1e-9 and gen_idx.numel() > 0:
        room_up = torch.clamp(Qmax_eff - Qg, min=0.0)
        room_dn = torch.clamp(Qg - Qmin_eff, min=0.0)

        if Q_def > 0 and room_up.sum() > 1e-12:
            w = room_up / room_up.sum()
            Qg = torch.minimum(Qg + Q_def * w, Qmax_eff)
        elif Q_def < 0 and room_dn.sum() > 1e-12:
            w = room_dn / room_dn.sum()
            Qg = torch.maximum(Qg + Q_def * w, Qmin_eff)

    Qg[~gen_mask] = 0.0

    return P_L, Q_L, Pg, Qg


def _to_1d_tensor(x, *, device=None, dtype=None):
    if x is None:
        return None
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if t.ndim == 0:
        t = t.view(1)
    elif t.ndim == 2 and 1 in t.shape:
        t = t.reshape(-1)
    elif t.ndim > 1:
        raise ValueError(f"Expected 1D tensor-like input, got shape {tuple(t.shape)}")
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


def _get_field(sample, name: str):
    if sample is None:
        return None
    if hasattr(sample, name):
        return getattr(sample, name)
    if isinstance(sample, dict):
        return sample.get(name, None)

    keys_attr = getattr(sample, "keys", None)
    keys = None
    if callable(keys_attr):
        try:
            keys = keys_attr()
        except TypeError:
            keys = None
    elif keys_attr is not None:
        keys = keys_attr

    if keys is not None:
        try:
            if name in keys:
                return sample[name]
        except Exception:
            pass
    return None


def _first_existing(sample, names):
    for name in names:
        v = _get_field(sample, name)
        if v is not None:
            return v
    return None


def _dict_get_nested(d: dict, path):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key, None)
    return cur


def _extract_raw_bus_solution_from_dict(sample_dict, *, device=None):
    if not isinstance(sample_dict, dict):
        return None

    bus_data = None
    for path in (
        ("solution", "solution", "bus"),
        ("solution", "bus"),
        ("bus",),
        ("network", "bus"),
    ):
        v = _dict_get_nested(sample_dict, path)
        if isinstance(v, dict):
            bus_data = v
            break
    if not isinstance(bus_data, dict):
        return None

    bus_ids, va, vm = [], [], []
    for bus_key, bus_entry in bus_data.items():
        if not isinstance(bus_entry, dict):
            continue
        va_k = "va" if "va" in bus_entry else ("Va" if "Va" in bus_entry else None)
        vm_k = "vm" if "vm" in bus_entry else ("Vm" if "Vm" in bus_entry else None)
        if va_k is None or vm_k is None:
            continue
        try:
            bus_id = int(bus_key)
        except Exception:
            if "bus_i" in bus_entry:
                bus_id = int(bus_entry["bus_i"])
            elif "index" in bus_entry:
                bus_id = int(bus_entry["index"])
            elif "id" in bus_entry:
                bus_id = int(bus_entry["id"])
            elif isinstance(bus_entry.get("source_id", None), (list, tuple)) and len(bus_entry["source_id"]) >= 2:
                bus_id = int(bus_entry["source_id"][1])
            else:
                continue
        bus_ids.append(bus_id)
        va.append(float(bus_entry[va_k]))
        vm.append(float(bus_entry[vm_k]))

    if len(bus_ids) == 0:
        return None

    return (
        torch.as_tensor(va, dtype=torch.get_default_dtype(), device=device),
        torch.as_tensor(vm, dtype=torch.get_default_dtype(), device=device),
        torch.as_tensor(bus_ids, dtype=torch.long, device=device),
    )


def _extract_raw_bus_solution(sample, *, device=None):
    # Works for raw dict samples and for HeteroData/global-store metadata that keeps raw JSON.
    candidates = []
    if isinstance(sample, dict):
        candidates.append(sample)

    for key in ("raw", "raw_data", "json", "sample", "metadata", "meta", "attrs", "solution", "network"):
        v = _get_field(sample, key)
        if isinstance(v, dict):
            candidates.append(v)

    node_types = getattr(sample, "node_types", None)
    if node_types is not None:
        preferred = [nt for nt in node_types if str(nt).lower() in {"bus", "buses", "node", "nodes"}]
        search_order = preferred + [nt for nt in node_types if nt not in preferred]
        for nt in search_order:
            try:
                node_store = sample[nt]
            except Exception:
                continue
            for key in ("raw", "raw_data", "json", "sample", "metadata", "meta", "attrs", "solution", "network"):
                v = _get_field(node_store, key)
                if isinstance(v, dict):
                    candidates.append(v)

    seen = set()
    for cand in candidates:
        cid = id(cand)
        if cid in seen:
            continue
        seen.add(cid)
        extracted = _extract_raw_bus_solution_from_dict(cand, device=device)
        if extracted is not None:
            return extracted
    return None


def _extract_from_node_store(node_store, *, state_order="va_vm", device=None):
    bus_voltages = _get_field(node_store, "bus_voltages")
    if bus_voltages is not None:
        bv = torch.as_tensor(bus_voltages)
        if bv.ndim == 2 and bv.shape[1] >= 2:
            # Explicit schema from user: col0=va, col1=vm
            va = bv[:, 0]
            vm = bv[:, 1]
            bus_ids = _first_existing(
                node_store,
                ("bus_id", "bus_ids", "bus_idx", "idx_bus", "bus_i", "node_id", "node_ids"),
            )
            va = _to_1d_tensor(va, device=device, dtype=torch.get_default_dtype())
            vm = _to_1d_tensor(vm, device=device, dtype=torch.get_default_dtype())
            if bus_ids is None:
                bus_ids = torch.arange(va.numel(), device=device, dtype=torch.long)
            else:
                bus_ids = _to_1d_tensor(bus_ids, device=device, dtype=torch.long)
                if bus_ids.numel() != va.numel():
                    return None
            return va, vm, bus_ids
        if bv.ndim == 2 and bv.shape[0] >= 2:
            # Secondary layout fallback: first row=va, second row=vm
            va = _to_1d_tensor(bv[0, :], device=device, dtype=torch.get_default_dtype())
            vm = _to_1d_tensor(bv[1, :], device=device, dtype=torch.get_default_dtype())
            bus_ids = _first_existing(
                node_store,
                ("bus_id", "bus_ids", "bus_idx", "idx_bus", "bus_i", "node_id", "node_ids"),
            )
            if bus_ids is None:
                bus_ids = torch.arange(va.numel(), device=device, dtype=torch.long)
            else:
                bus_ids = _to_1d_tensor(bus_ids, device=device, dtype=torch.long)
                if bus_ids.numel() != va.numel():
                    return None
            return va, vm, bus_ids

    va = _first_existing(node_store, ("va", "Va", "theta", "angle", "voltage_angle"))
    vm = _first_existing(node_store, ("vm", "Vm", "v", "voltage_magnitude"))
    bus_ids = _first_existing(
        node_store,
        ("bus_id", "bus_ids", "bus_idx", "idx_bus", "bus_i", "node_id", "node_ids"),
    )

    if va is None or vm is None:
        y = _first_existing(node_store, ("y", "target", "targets", "state", "states"))
        if y is not None:
            y_t = torch.as_tensor(y)
            if y_t.ndim == 2 and y_t.shape[1] >= 2:
                if state_order == "vm_va":
                    vm, va = y_t[:, 0], y_t[:, 1]
                else:
                    va, vm = y_t[:, 0], y_t[:, 1]
            elif y_t.ndim == 2 and y_t.shape[0] >= 2:
                if state_order == "vm_va":
                    vm, va = y_t[0, :], y_t[1, :]
                else:
                    va, vm = y_t[0, :], y_t[1, :]

    if va is None or vm is None:
        x = _first_existing(node_store, ("x", "node_features", "features"))
        feature_names = _first_existing(node_store, ("feature_names", "x_names", "node_feature_names"))
        if x is not None and feature_names is not None:
            x_t = torch.as_tensor(x)
            if x_t.ndim == 2:
                names = [str(n).strip().lower() for n in list(feature_names)]
                va_candidates = ("va", "theta", "angle", "voltage_angle")
                vm_candidates = ("vm", "voltage_magnitude", "v")
                va_idx = next((i for i, n in enumerate(names) if n in va_candidates), None)
                vm_idx = next((i for i, n in enumerate(names) if n in vm_candidates), None)
                if va_idx is not None and vm_idx is not None:
                    va = x_t[:, va_idx]
                    vm = x_t[:, vm_idx]

    if va is None or vm is None:
        return None

    va = _to_1d_tensor(va, device=device, dtype=torch.get_default_dtype())
    vm = _to_1d_tensor(vm, device=device, dtype=torch.get_default_dtype())
    if va.numel() != vm.numel():
        return None

    if bus_ids is None:
        bus_ids = torch.arange(va.numel(), device=device, dtype=torch.long)
    else:
        bus_ids = _to_1d_tensor(bus_ids, device=device, dtype=torch.long)
        if bus_ids.numel() != va.numel():
            return None

    return va, vm, bus_ids


def _extract_from_heterodata(sample, *, state_order="va_vm", device=None):
    node_types = getattr(sample, "node_types", None)
    if node_types is None:
        return None

    preferred = [nt for nt in node_types if str(nt).lower() in {"bus", "buses", "node", "nodes"}]
    search_order = preferred + [nt for nt in node_types if nt not in preferred]

    for nt in search_order:
        try:
            node_store = sample[nt]
        except Exception:
            continue
        extracted = _extract_from_node_store(node_store, state_order=state_order, device=device)
        if extracted is not None:
            return extracted
    return None


def _extract_va_vm_bus_ids(sample, *, state_order="va_vm", device=None):
    raw_extracted = _extract_raw_bus_solution(sample, device=device)
    if raw_extracted is not None:
        return raw_extracted

    hetero_extracted = _extract_from_heterodata(sample, state_order=state_order, device=device)
    if hetero_extracted is not None:
        return hetero_extracted

    va = _first_existing(sample, ("va", "Va", "theta", "angle", "voltage_angle"))
    vm = _first_existing(sample, ("vm", "Vm", "v", "voltage_magnitude"))
    bus_ids = _first_existing(
        sample,
        ("bus_id", "bus_ids", "bus_idx", "idx_bus", "bus_i", "node_id", "node_ids"),
    )

    if va is None or vm is None:
        y = _first_existing(sample, ("y", "target", "targets", "state", "states"))
        if y is not None:
            y_t = torch.as_tensor(y)
            if y_t.ndim == 2 and y_t.shape[1] >= 2:
                if state_order == "vm_va":
                    vm, va = y_t[:, 0], y_t[:, 1]
                else:
                    va, vm = y_t[:, 0], y_t[:, 1]
            elif y_t.ndim == 2 and y_t.shape[0] >= 2:
                if state_order == "vm_va":
                    vm, va = y_t[0, :], y_t[1, :]
                else:
                    va, vm = y_t[0, :], y_t[1, :]

    if va is None or vm is None:
        x = _first_existing(sample, ("x", "node_features", "features"))
        feature_names = _first_existing(sample, ("feature_names", "x_names", "node_feature_names"))
        if x is not None and feature_names is not None:
            x_t = torch.as_tensor(x)
            if x_t.ndim == 2:
                names = [str(n).strip().lower() for n in list(feature_names)]
                va_candidates = ("va", "theta", "angle", "voltage_angle")
                vm_candidates = ("vm", "voltage_magnitude", "v")
                va_idx = next((i for i, n in enumerate(names) if n in va_candidates), None)
                vm_idx = next((i for i, n in enumerate(names) if n in vm_candidates), None)
                if va_idx is not None and vm_idx is not None:
                    va = x_t[:, va_idx]
                    vm = x_t[:, vm_idx]

    if va is None or vm is None:
        raise KeyError(
            "Could not extract voltage angle/magnitude from PFDelta sample. "
            "Expected fields like ('va','vm') or a 2-column target tensor."
        )

    va = _to_1d_tensor(va, device=device, dtype=torch.get_default_dtype())
    vm = _to_1d_tensor(vm, device=device, dtype=torch.get_default_dtype())
    if va.numel() != vm.numel():
        raise ValueError(
            f"Mismatched va/vm lengths: va has {va.numel()} entries, vm has {vm.numel()} entries."
        )

    if bus_ids is None:
        bus_ids = torch.arange(va.numel(), device=device, dtype=torch.long)
    else:
        bus_ids = _to_1d_tensor(bus_ids, device=device, dtype=torch.long)
        if bus_ids.numel() != va.numel():
            raise ValueError(
                f"Bus-id length ({bus_ids.numel()}) does not match va/vm length ({va.numel()})."
            )

    return va, vm, bus_ids


def _sort_and_align_bus_order(va, vm, bus_ids, sys, *, device=None):
    perm = torch.argsort(bus_ids)
    va = va.index_select(0, perm)
    vm = vm.index_select(0, perm)
    bus_ids = bus_ids.index_select(0, perm)

    if sys is None or not hasattr(sys, "bus") or sys.bus is None:
        return va, vm

    ref_bus_ids = None
    for col in ("idx_bus", "bus_i", "index"):
        if col in sys.bus.columns:
            ref_bus_ids = torch.as_tensor(sys.bus[col].to_numpy(), dtype=torch.long, device=device)
            break
    if ref_bus_ids is None:
        try:
            ref_bus_ids = torch.as_tensor(sys.bus.index.to_numpy(), dtype=torch.long, device=device)
        except Exception:
            return va, vm

    if ref_bus_ids.numel() != bus_ids.numel():
        return va, vm

    ref_list = [int(x) for x in ref_bus_ids.tolist()]
    ref_set = set(ref_list)

    candidates = (bus_ids, bus_ids - 1, bus_ids + 1)
    for cand in candidates:
        cand_list = [int(x) for x in cand.tolist()]
        if len(set(cand_list)) != len(cand_list):
            continue
        if set(cand_list) != ref_set:
            continue
        pos = {bid: i for i, bid in enumerate(cand_list)}
        order = torch.as_tensor([pos[bid] for bid in ref_list], dtype=torch.long, device=device)
        return va.index_select(0, order), vm.index_select(0, order)

    return va, vm


class DataGenerator:
    def __init__(self, device: str | torch.device | None = None, seed: None | int = None):
        self.device = self._normalize_device(device)
        self.Pl = None
        self.Ql = None
        self.Pg = None
        self.Qg = None
        self.timeseries = None
        self.m = None
        self.cov = None
        self.val_data = None
        self.it = 0.
        self.used_idx = []
        self.random_generator = self._create_generator(seed, device=self.device)

    @staticmethod
    def _normalize_device(device: str | torch.device | None) -> torch.device | None:
        if device is None:
            return torch.device("cpu")
        d = torch.device(device)
        if d.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if d.type == "mps":
            mps_ok = bool(torch.backends.mps.is_built()) and bool(torch.backends.mps.is_available())
            if not mps_ok:
                return torch.device("cpu")
        return d

    @staticmethod
    def _create_generator(seed: int, device: str | torch.device | None = None) -> torch.Generator:
        if device is None:
            gen = torch.Generator()
        else:
            gen = torch.Generator(device=torch.device(device))
        if seed is not None:
            gen.manual_seed(int(seed))
        return gen

    def reseed(self, seed: int):
        self.random_generator = self._create_generator(seed, device=self.device)

    def load_val_data(self, data_dir="../../SE_torch/data_parser/data/time_series4", ood=False):
        if ood:
            val_dataset = torch.load(os.path.join(data_dir, "vam_ood.pt"), weights_only=False,
                                     map_location="cpu")
        else:
            val_dataset = torch.load(os.path.join(data_dir, "val_dataset_vam.pt"), weights_only=False,
                                     map_location="cpu")
        # val_data = val_dataset.dataset.tensors[0][val_dataset.indices]
        self.val_data = val_dataset

    def load_flow_from_dir(self, data_dir="../../SE_torch/data_parser/data/time_series2"):
        # Load CSVs with pandas, convert to torch tensors
        Pg_df = pd.read_csv(f'{data_dir}/ieee118_186_Pg_timeseries.csv', index_col=0)
        Pl_df = pd.read_csv(f'{data_dir}/ieee118_186_Pl_timeseries.csv', index_col=0)
        Qg_df = pd.read_csv(f'{data_dir}/ieee118_186_Qg_timeseries.csv', index_col=0)
        Ql_df = pd.read_csv(f'{data_dir}/ieee118_186_Ql_timeseries.csv', index_col=0)

        self.timeseries = Pg_df.index.to_numpy()

        self.Pg = torch.as_tensor(Pg_df.values, device=self.device, dtype=torch.get_default_dtype())
        self.Pl = torch.as_tensor(Pl_df.values, device=self.device, dtype=torch.get_default_dtype())
        self.Qg = torch.as_tensor(Qg_df.values, device=self.device, dtype=torch.get_default_dtype())
        self.Ql = torch.as_tensor(Ql_df.values, device=self.device, dtype=torch.get_default_dtype())

        mean_path = f'{data_dir}/mean.npy'
        cov_path  = f'{data_dir}/covariance.npy'
        if os.path.exists(mean_path):
            self.m = torch.from_numpy(__import__("numpy").load(mean_path)).to(self.device)
        if os.path.exists(cov_path):
            self.cov = torch.from_numpy(__import__("numpy").load(cov_path)).to(self.device)

    def _resolve_generator(self, seed: int | None = None, device: str | torch.device | None = None) -> torch.Generator:
        target_device = self.device if device is None else self._normalize_device(device)
        if seed is not None:
            return self._create_generator(seed, device=target_device)
        if target_device is not None and self.random_generator.device != target_device:
            src_device = self.random_generator.device
            seed_token = torch.randint(
                low=0,
                high=2_147_483_647,
                size=(1,),
                generator=self.random_generator,
                device=src_device,
            ).item()
            return self._create_generator(int(seed_token), device=target_device)
        return self.random_generator

    @staticmethod
    def _save_sample_snapshot(path: str, vcs: list[torch.Tensor], completed: int, total: int,
                              random_flow: bool, seed: int | None = None, error: str | None = None):
        if path is None:
            return
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        payload = {
            "completed": int(completed),
            "total": int(total),
            "random_flow": bool(random_flow),
            "seed": None if seed is None else int(seed),
            "error": error,
        }
        if len(vcs) > 0:
            vcs_t = torch.stack(vcs, dim=0).detach().cpu()
            payload["Vcs"] = vcs_t
            payload["T"] = torch.angle(vcs_t).to(torch.get_default_dtype())
            payload["V"] = torch.abs(vcs_t).to(torch.get_default_dtype())
        else:
            payload["Vcs"] = torch.empty((0,), dtype=torch.complex64)
            payload["T"] = torch.empty((0,), dtype=torch.get_default_dtype())
            payload["V"] = torch.empty((0,), dtype=torch.get_default_dtype())
        torch.save(payload, path)

    def sample(self, sys, num_samples=1, random_flow='generate', cart=False, seed=None, verbose=False, **kwargs):
        g = self._resolve_generator(seed=seed)
        pf_mean_by_type_id = kwargs.get('pf_mean_by_type_id', None)
        checkpoint_base = kwargs.get("sample_save_path", None)
        checkpoint_every = kwargs.get("sample_save_every", None)
        checkpoint_every = int(checkpoint_every) if checkpoint_every is not None else None
        save_midpoint = bool(kwargs.get("sample_save_midpoint", checkpoint_base is not None))
        save_on_exception = bool(kwargs.get("sample_save_on_exception", checkpoint_base is not None))
        return_partial_on_exception = bool(kwargs.get("sample_return_partial_on_exception", True))

        def _mk_save_path(kind: str):
            if checkpoint_base is None:
                return None
            root, ext = os.path.splitext(checkpoint_base)
            ext = ext if ext else ".pt"
            if kind == "progress":
                return f"{root}_progress{ext}"
            if kind == "midpoint":
                return f"{root}_midpoint{ext}"
            if kind == "exception":
                return f"{root}_exception{ext}"
            return checkpoint_base

        if random_flow == 'generate':
            T0, V0 = init_start_point(sys)
            x_init = torch.stack([T0, V0], dim=1).to(self.device, dtype=torch.get_default_dtype())

            user = {'list': ['voltage'], 'stop': 1e-8, 'maxIter': 500}

            Vcs = []
            midpoint_target = max(1, num_samples // 2)
            midpoint_saved = False
            try:
                for i in tqdm(range(num_samples), desc="Generating data", colour='MAGENTA', disable=not verbose, leave=False):
                    Pli, Qli, Pgi, Qgi = regenerate_PQ_numeric_types(sys, device=self.device, seed=seed,
                                                                     pf_mean_by_type_id=pf_mean_by_type_id,
                                                                     random_generator=g)
                    curr_sys = sys.copy(self.device)
                    loads_i = torch.stack([Pli, Qli], dim=1)
                    gens_i = torch.stack([Pgi, Qgi], dim=1)
                    pf = NR_PF(curr_sys, loads_i, gens_i, x_init, user, device=self.device.type)
                    Vcs.append(pf['Vc'])

                    completed = len(Vcs)
                    if save_midpoint and (not midpoint_saved) and completed >= midpoint_target:
                        self._save_sample_snapshot(
                            _mk_save_path("midpoint"), Vcs, completed, num_samples, random_flow=True, seed=seed
                        )
                        midpoint_saved = True
                    if checkpoint_every is not None and checkpoint_every > 0 and (completed % checkpoint_every == 0):
                        self._save_sample_snapshot(
                            _mk_save_path("progress"), Vcs, completed, num_samples, random_flow=True, seed=seed
                        )
            except Exception as exc:
                if save_on_exception:
                    self._save_sample_snapshot(
                        _mk_save_path("exception"), Vcs, len(Vcs), num_samples, random_flow=True,
                        seed=seed, error=repr(exc)
                    )
                if return_partial_on_exception and len(Vcs) > 0:
                    Vcs_p = torch.stack(Vcs, dim=0)
                    if cart:
                        return Vcs_p.real, Vcs_p.imag
                    return torch.angle(Vcs_p).to(torch.get_default_dtype()), torch.abs(Vcs_p).to(torch.get_default_dtype())
                raise


            Vcs = torch.stack(Vcs, dim=0)
            if num_samples == 1:
                Vcs = Vcs.view(-1)

            if cart:
                return Vcs.real, Vcs.imag

            T = torch.angle(Vcs).to(torch.get_default_dtype())
            V = torch.abs(Vcs).to(torch.get_default_dtype())

        elif random_flow == 'val' or random_flow == 'val_ood':
            if self.val_data is None:
                self.load_val_data(ood=random_flow == 'val_ood')
            n_rows = self.val_data.shape[0]
            row_idx = torch.randint(low=0, high=n_rows, size=(num_samples,), device=self.device)
            while row_idx in self.used_idx:
                row_idx = torch.randint(low=0, high=n_rows, size=(num_samples,), device=self.device)
            self.used_idx.append(row_idx)
            vam = self.val_data[row_idx].squeeze(0)
            T, V = vam[0].view(-1), vam[1].view(-1)

        else:
            if self.Pl is None:
                self.load_flow_from_dir()
            n_rows = self.Pl.shape[0]
            row_idx = torch.randint(low=0, high=n_rows, size=(num_samples,), generator=g, device=self.device)
            Pl = self.Pl.index_select(0, row_idx)
            Pg = self.Pg.index_select(0, row_idx)
            Ql = self.Ql.index_select(0, row_idx)
            Qg = self.Qg.index_select(0, row_idx)

            T0, V0 = init_start_point(sys)
            x_init = torch.stack([T0, V0], dim=1)

            user = {'list': ['voltage'], 'stop': 1e-8, 'maxIter': 500}

            Vcs = []
            total = int(Pl.shape[0])
            midpoint_target = max(1, total // 2)
            midpoint_saved = False
            try:
                for i in tqdm(range(total), desc="Generating data", colour='MAGENTA', disable=not verbose, leave=False):
                    curr_sys = sys.copy(self.device)
                    loads_i = torch.stack([Pl[i], Ql[i]], dim=1)
                    gens_i  = torch.stack([Pg[i], Qg[i]], dim=1)
                    pf = NR_PF(curr_sys, loads_i, gens_i, x_init, user, device=self.device)
                    Vcs.append(pf['Vc'])

                    completed = len(Vcs)
                    if save_midpoint and (not midpoint_saved) and completed >= midpoint_target:
                        self._save_sample_snapshot(
                            _mk_save_path("midpoint"), Vcs, completed, total, random_flow=False, seed=seed
                        )
                        midpoint_saved = True
                    if checkpoint_every is not None and checkpoint_every > 0 and (completed % checkpoint_every == 0):
                        self._save_sample_snapshot(
                            _mk_save_path("progress"), Vcs, completed, total, random_flow=False, seed=seed
                        )
            except Exception as exc:
                if save_on_exception:
                    self._save_sample_snapshot(
                        _mk_save_path("exception"), Vcs, len(Vcs), total, random_flow=False,
                        seed=seed, error=repr(exc)
                    )
                if return_partial_on_exception and len(Vcs) > 0:
                    Vcs_p = torch.stack(Vcs, dim=0)
                    if cart:
                        return Vcs_p.real, Vcs_p.imag
                    return torch.angle(Vcs_p).to(torch.get_default_dtype()), torch.abs(Vcs_p).to(torch.get_default_dtype())
                raise

            Vcs = torch.stack(Vcs, dim=0)
            if num_samples == 1:
                Vcs = Vcs.view(-1)

            if cart:
                return Vcs.real, Vcs.imag

            T = torch.angle(Vcs).to(torch.get_default_dtype())
            V = torch.abs(Vcs).to(torch.get_default_dtype())

        return T, V

    def generate_measurements(self, sys, branch, random_flow='generate', T_true=None, V_true=None, device=torch.device("cpu"), **kwargs):
        dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        kwargs["verbose"] = False
        if T_true is None or V_true is None:
            T_true, V_true = self.sample(sys, random_flow=random_flow, **kwargs)
            sys.slk_bus = (sys.slk_bus[0], T_true[sys.slk_bus[0]], V_true[sys.slk_bus[0]])
        eff_device = self._normalize_device(device if device is not None else self.device)
        if eff_device is None and isinstance(T_true, torch.Tensor):
            eff_device = T_true.device
        if eff_device is None:
            eff_device = torch.device("cpu")

        T_true = T_true.to(device=eff_device, dtype=torch.get_default_dtype())
        V_true = V_true.to(device=eff_device, dtype=torch.get_default_dtype())
        vc = torch.polar(V_true, T_true)
        Vc_true = torch.cat([vc.real, vc.imag], dim=-1)

        keep_nans = bool(kwargs.get("keep_nans", False))
        sample_cfg = kwargs.get("sample", 1.0)

        def _p_for_type(mtype: str) -> float:
            if isinstance(sample_cfg, dict):
                p = float(sample_cfg.get(mtype, 1.0))
            else:
                p = float(sample_cfg)
            return max(0.0, min(1.0, p))

        def _bernoulli_mask(n: int, p: float) -> torch.Tensor:
            if n <= 0:
                return torch.empty((0,), device=eff_device, dtype=torch.bool)
            indexes = torch.randperm(n)[:int(n * p) + 1]
            ber_mask = torch.isin(torch.arange(n), indexes)
            return ber_mask.to(torch.bool)

        meas_idx = {}
        bus_mask_all = torch.ones(len(sys.bus), device=eff_device, dtype=torch.bool)
        nbr = len(branch.i)
        half = nbr // 2
        meas_masks = {}
        if kwargs.get("flow"):
            PQf_mask = torch.cat(
                [torch.ones(half, device=eff_device), torch.zeros(half, device=eff_device)],
                dim=0
            ).to(torch.bool)
            meas_masks["Pf"] = PQf_mask
            meas_masks["Qf"] = PQf_mask
            if not keep_nans:
                p = _p_for_type("Pf")
                samp = _bernoulli_mask(int(PQf_mask.sum().item()), p)
                PQf_mask_s = PQf_mask.clone()
                PQf_mask_s[PQf_mask] = samp
                meas_idx["Pf_idx"] = PQf_mask_s
                meas_idx["Qf_idx"] = PQf_mask_s
            else:
                meas_idx["Pf_idx"] = PQf_mask
                meas_idx["Qf_idx"] = PQf_mask

        if kwargs.get("injection"):
            meas_masks["Pi"] = bus_mask_all
            meas_masks["Qi"] = bus_mask_all
            if not keep_nans:
                p = _p_for_type("Pi")
                samp = _bernoulli_mask(int(bus_mask_all.sum().item()), p)
                inj_mask = bus_mask_all.clone()
                inj_mask[bus_mask_all] = samp
                meas_idx["Pi_idx"] = inj_mask
                meas_idx["Qi_idx"] = inj_mask
            else:
                meas_idx["Pi_idx"] = bus_mask_all
                meas_idx["Qi_idx"] = bus_mask_all

        if kwargs.get("voltage"):
            meas_masks["Vm"] = bus_mask_all
            if not keep_nans:
                p = _p_for_type("Vm")
                samp = _bernoulli_mask(int(bus_mask_all.sum().item()), p)
                vm_mask = bus_mask_all.clone()
                vm_mask[bus_mask_all] = samp
                meas_idx["Vm_idx"] = vm_mask
            else:
                meas_idx["Vm_idx"] = bus_mask_all

        if kwargs.get("current"):
            cm_mask_all = torch.ones(len(branch.i), device=eff_device, dtype=torch.bool)
            meas_masks["Cm"] = cm_mask_all
            if not keep_nans:
                p = _p_for_type("Cm")
                samp = _bernoulli_mask(int(cm_mask_all.sum().item()), p)
                cm_mask = cm_mask_all.clone()
                cm_mask[cm_mask_all] = samp
                meas_idx["Cm_idx"] = cm_mask
            else:
                meas_idx["Cm_idx"] = cm_mask_all

        meas_types = ["Pf", "Qf", "Cm", "Pi", "Qi", "Vm"]
        agg_meas_idx = aggregate_meas_idx(meas_idx, meas_types)

        if kwargs.get("noise"):
            pieces = []
            for mtype in meas_types:
                count = len(agg_meas_idx[mtype])
                sigma2 = float(kwargs.get(f"{mtype}_noise", 1.0))
                pieces.append(torch.full((count,), sigma2, device=eff_device))
            var = torch.cat(pieces, dim=0)

            if keep_nans:
                sample_pieces = []
                for mtype in meas_types:
                    if mtype == "Qf" or mtype == "Qi":
                        continue
                    mask = meas_masks.get(mtype)
                    if mask is None:
                        continue
                    mask_s = mask.clone()
                    count = int(mask.sum().item())
                    if count == 0:
                        continue
                    p = _p_for_type(mtype)
                    samp = _bernoulli_mask(count, p)
                    mask_s[mask] = samp
                    sample_pieces.append(samp)
                    if mtype == "Pf" or mtype == "Pi":
                        sample_pieces.append(samp)

                sample = torch.cat(sample_pieces, dim=0).to(torch.int) if len(sample_pieces) else torch.empty((0,), device=eff_device)
                var = (var * sample) + (1 - sample) * 10.
            noise = torch.randn(*var.shape, dtype=torch.get_default_dtype()) * torch.sqrt(var)

        torch.set_default_dtype(dtype)
        h_ac_cart = H_AC_cartesian(sys, branch, meas_idx)
        h_ac_polar = H_AC_polar(sys, branch, meas_idx)

        z = torch.as_tensor(h_ac_cart.estimate(Vc_true), device=eff_device)
        if kwargs.get("noise"):
            var = var.to(dtype=torch.get_default_dtype())
            noise = noise.to(dtype=torch.get_default_dtype())
            z = z + noise
        else:
            var = torch.ones(z.numel(), device=eff_device, dtype=torch.get_default_dtype())
        self.it += 1
        return z, var, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true
