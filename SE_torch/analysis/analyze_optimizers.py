import json
import os
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch

from SE_torch.PF_equations.PF_cartesian import H_AC as H_AC_cartesian
from SE_torch.data_generator import DataGenerator
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System, Branch
from SE_torch.utils import square_mag, normalize_measurements, RMSE, init_start_point
from SE_torch.optimizers.GN_se import GN_se
from SE_torch.optimizers.LM_opt import LMOpt
from SE_torch.optimizers.FO_se import LBFGS_se, LBFGS_se_latent, GDBT_se
from SE_torch.learn_prior.GNU_GNN.models import GNU_Model
from SE_torch.optimizers.se_loss import (SELoss, SELossGN, SELossNF, SELossVAE, SELossNFLat, SELossVAELat, SELossFM, SELossNF2, SELossNF2Lat, SELossFMLat)


def _rmse_curve_from_all_x(all_x, T_true, V_true, nb):
    if all_x is None:
        return None

    if isinstance(all_x, (list, tuple)):
        xs = all_x
    else:
        xs = list(all_x)

    vals = []
    with torch.no_grad():
        for x in xs:
            T = x[:nb]
            V = x[nb:nb + nb]
            vals.append(RMSE(T_true, V_true, T.detach(), V.detach()))
    return torch.as_tensor(vals, dtype=torch.float32)


def _pad_nan_stack_1d(curves):
    if len(curves) == 0:
        return None, None

    lengths = [int(c.numel()) for c in curves]
    Kmax = max(lengths)

    stacked = torch.full((len(curves), Kmax), float("nan"), dtype=torch.float32)
    for i, c in enumerate(curves):
        k = int(c.numel())
        stacked[i, :k] = c

    counts = torch.sum(~torch.isnan(stacked), dim=0).to(torch.int32)
    return stacked, counts


def _nanmean_curve(curves):
    stacked, counts = _pad_nan_stack_1d(curves)
    if stacked is None:
        return None, None
    mean_curve = torch.nanmean(stacked, dim=0)
    return mean_curve, counts


def _plot_rmse_vs_step_for_obs(obs, mean_curve_by_opt, out_dir, opt_name_map=None):
    _ensure_dir(out_dir)

    opt_names = list(mean_curve_by_opt.keys())
    if opt_name_map is None:
        opt_name_map = {k: k.upper() for k in opt_names}

    plt.figure(figsize=(7.0, 4.0), dpi=300)
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    eps = 1e-12

    for i, opt in enumerate(opt_names):
        y = mean_curve_by_opt[opt]
        if y is None:
            continue
        y = y.detach().cpu().numpy()[1:]
        y = np.maximum(y, eps)
        x = np.arange(len(y))  # step index

        plt.plot(
            x,
            y,
            label=opt_name_map.get(opt, opt),
            color=colors[i % len(colors)],
            linewidth=1.4,
            marker=markers[i % len(markers)],
            markersize=4.0,
            markeredgewidth=0.8,
        )

    plt.xlabel("Step")
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs Step (observability={obs:.2f})")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.4)
    plt.tick_params(direction="in", length=4, width=0.8, which="both", top=True, right=True)
    plt.legend(loc="best", frameon=True, framealpha=0.9, ncol=2)
    plt.tight_layout()

    png = os.path.join(out_dir, f"rmse_vs_step_obs{obs:.2f}.png")
    pdf = os.path.join(out_dir, f"rmse_vs_step_obs{obs:.2f}.pdf")
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.show()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _plot_metric_vs_obs(obs_levels, stats_by_obs, metric_key, ylabel, title, save_path_png, save_path_pdf=None,
    opt_name_map=None, ymin=None,ymax=None,):

    obs_levels = np.array(obs_levels)
    first_obs = obs_levels[0]
    # opt_names = list(stats_by_obs[first_obs].keys())
    # opt_names = ['lm', 'lm_gs', 'lm_nf_small', 'lbfgs_nf', 'gdbt_nf', 'lm_nf_lat', 'lm_nf_nld', 'lm_nf']
    opt_names = ['lm_gs', 'lm_nf_lat', 'lm_nf_nld', 'lm_nf']
    # opt_names.reverse()
    opt_name_map = {
        'lm_nf': 'lmnf',
        'lm_nf_small': 'wlmnf',
        'lm_nf_nld': 'lmis',
        'lm_nf_lat': 'lmil',
        'lm_gs': 'lmgs',
        'lbfgs_nf': 'lbfgsnf',
        'gdbt_nf': 'gdnf',
        'lm': 'lm'
    }
    opt_name_map = {k: v.upper() for k, v in opt_name_map.items()}
    if opt_name_map is None:
        opt_name_map = {k: k.upper() for k in opt_names}

    # ---- Figure style ----
    plt.figure(figsize=(7.0, 4.0), dpi=300)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    eps = 1e-12  # safety for log-scale

    for i, opt in enumerate(opt_names):
        y = np.array([stats_by_obs[obs][opt][metric_key] for obs in obs_levels])
        y = np.maximum(y, eps)  # avoid log(0)

        plt.plot(
            obs_levels,
            y,
            label=opt_name_map.get(opt, opt),
            color=colors[i % len(colors)],
            linewidth=1.4,
            marker=markers[i % len(markers)],
            markersize=4.5,
            markeredgewidth=0.8,
        )

    plt.xlabel("Observability")
    plt.ylabel(ylabel)
    plt.title(title)

    # ---- log scale ----
    plt.yscale("log")

    if ymin is not None or ymax is not None:
        plt.ylim(bottom=ymin, top=ymax)

    # ---- grid & ticks ----
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.4)
    plt.tick_params(
        direction="in",
        length=4,
        width=0.8,
        which="both",
        top=True,
        right=True,
    )

    plt.legend(
        loc="best",
        frameon=True,
        framealpha=0.9,
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(save_path_png, bbox_inches="tight")
    if save_path_pdf is not None:
        plt.savefig(save_path_pdf, bbox_inches="tight")
    plt.show()


def gen_pf_ood_plots(config, x_as_distance=True):
    out_dir = config.get("out_dir", "./results/observability_pf_sweep")
    results_json_path = os.path.join(out_dir, "rmse_vs_pf_by_observability.json")

    if not os.path.exists(results_json_path):
        print("PF-OOD results not found.")
        return

    results = json.load(open(results_json_path))
    pf_id = float(config.get("pf_id", 0.95))

    for obs_str, obs_data in results.items():
        obs = float(obs_str)

        pf_grid = sorted(float(pf) for pf in obs_data.keys())
        opt_names = list(next(iter(obs_data.values())).keys())

        x = np.array(pf_grid)
        if x_as_distance:
            x = np.abs(x - pf_id)

        plt.figure(figsize=(7.0, 4.0), dpi=300)
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        })

        colors = plt.cm.tab10.colors
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        eps = 1e-12

        for i, opt in enumerate(opt_names):
            y = np.array([obs_data[f"{pf:.3f}"][opt]["rmse_mean"] for pf in pf_grid])
            y = np.maximum(y, eps)

            plt.plot(
                x, y,
                label=opt.upper(),
                color=colors[i % len(colors)],
                linewidth=1.4,
                marker=markers[i % len(markers)],
                markersize=4.5,
                markeredgewidth=0.8,
            )

        plt.xlabel(r"$|pf - pf_{\mathrm{ID}}|$" if x_as_distance else "Load-bus PF mean")
        plt.ylabel("RMSE")
        plt.title(f"RMSE vs OOD (observability={obs:.2f})")
        plt.yscale("log")
        plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.4)
        plt.legend(loc="best", frameon=True, framealpha=0.9, ncol=2)
        plt.tight_layout()

        suffix = "dist" if x_as_distance else "pf"
        png = os.path.join(out_dir, f"rmse_vs_{suffix}_obs{obs:.2f}.png")
        pdf = os.path.join(out_dir, f"rmse_vs_{suffix}_obs{obs:.2f}.pdf")
        plt.savefig(png, bbox_inches="tight")
        plt.savefig(pdf, bbox_inches="tight")
        plt.show()


def run_experiment(data, sys, branch, data_generator, optimizers, **kwargs):
    T0, V0 = init_start_point(sys, data, how='flat')

    gen_data = data_generator.generate_measurements(sys, branch, device=None, **kwargs)
    z, v, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true = gen_data
    # norm_H = torch.linalg.norm(h_ac_cart.H, dim=(1, 2)) + 1e-12
    final_lm = LMOpt(loss_func=SELoss())

    x0 = torch.concatenate([T0, V0])

    rmse, k, converged, loss, timing = {}, {}, {}, {}, {}
    rmse_curve = {}

    for opt_name, optimizer in optimizers.items():
        # if 'gdbt' in opt_name:
        #     norm_H = torch.linalg.norm(h_ac_cart.H, dim=(1, 2)) + 1e-12
        t0 = time.perf_counter()

        x_opt, T_opt, V_opt, opt_converged, k_opt, loss_opt, all_x_opt = optimizer(
            x0, z, v, sys.slk_bus, h_ac_polar, sys.nb
        )
        dt = time.perf_counter() - t0
        # x_lm, T_lm, V_lm, lm_converged, k_lm, loss_lm, all_x_lm = optimizer(x_opt, z, v, sys.slk_bus, h_ac_polar, sys.nb)

        rmse[opt_name] = RMSE(T_true, V_true, T_opt.detach(), V_opt.detach())
        k[opt_name] = k_opt
        converged[opt_name] = opt_converged
        loss[opt_name] = loss_opt
        timing[opt_name] = dt  # seconds
        rmse_curve[opt_name] = _rmse_curve_from_all_x(all_x_opt, T_true, V_true, sys.nb)

    return rmse, rmse_curve, k, converged, loss, timing


def run_all_experiments(config):
    net_file = config["net_file"]
    verbose = config.get("verbose", False)
    max_iter = config.get("max_iter", 1000)
    obs_levels = config.get("observability_levels", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ood_lvl = config.get("ood_lvl", .95)
    num_experiments = int(config.get("num_experiments_per_obs", 100))

    out_dir = config.get("out_dir", "./results/observability_sweep")
    _ensure_dir(out_dir)
    results_json_path = os.path.join(out_dir, "mean_results_by_observability.json")

    data = parse_ieee_mat(net_file)
    system_data = data["data"]["system"]
    sys = System(system_data)
    branch = Branch(sys.branch)
    edge_index = torch.stack([branch.i, branch.j]).to(torch.long)
    data_generator = DataGenerator()

    kwargs_base = deepcopy(config["generator_kwargs"])

    m = torch.load("../learn_prior/datasets/mean_polar.pt")
    cov = torch.load("../learn_prior/datasets/cov_polar.pt")

    NF_prior_config_path = "../learn_prior/configs/NF_4_0_128.json"
    NF_prior_weak_config_path = "../learn_prior/configs/NF_2_0_128.json"

    pf_mean_by_type_id = {1: ood_lvl, 2: 0.98, 3: 0.99}
    optimizers = {
        "lm": LMOpt(loss_func=SELoss()),
        "lm_gs": LMOpt(loss_func=SELossGN(m=m, Q=cov)),
        "lm_nf": LMOpt(loss_func=SELossNF(NF_config_path=NF_prior_config_path, with_log_det=True)),
        "lm_nf_nld": LMOpt(loss_func=SELossNF(NF_config_path=NF_prior_config_path, with_log_det=False)),
        "lm_nf_nld_lat": LMOpt(loss_func=SELossNFLat(NF_config_path=NF_prior_config_path, with_log_det=False), latent='dss'),
        "lm_nf_weak": LMOpt(loss_func=SELossNF(NF_config_path=NF_prior_weak_config_path, with_log_det=True)),
        "gdbt_nf": GDBT_se(verbose=False, use_prior=True, prior_config_path=NF_prior_config_path),
        "lbfgs_nf": LBFGS_se(verbose=False, use_prior=True, prior_config_path=NF_prior_config_path),
    }
    kwargs_base["pf_mean_by_type_id"] = pf_mean_by_type_id

    stats_by_obs = {}

    for obs in obs_levels:
        GNU_GNN_config_path = config.get("GNU_config", f"../learn_prior/configs/GNU_GNN_obs{obs}_config.json")
        GNU_GNN_config = json.load(open(GNU_GNN_config_path))
        ckpt_path = f"../learn_prior/GNU_GNN/models/{GNU_GNN_config.get('ckpt_name')}"

        GNU_GNN_model = GNU_Model(**GNU_GNN_config)
        edge_index = torch.stack([branch.i, branch.j]).to(torch.long)
        GNU_GNN_model.load_state_dict(torch.load(ckpt_path))
        GNU_GNN_model.edge_index = edge_index
        GNU_GNN_model.slk_bus = sys.slk_bus

        optimizers["gnu_gnn"] = GNU_GNN_model.optimize
        kwargs = deepcopy(kwargs_base)
        kwargs["sample"] = float(obs)

        all_rmse = {opt: [] for opt in optimizers.keys()}
        all_k = {opt: [] for opt in optimizers.keys()}
        all_conv = {opt: [] for opt in optimizers.keys()}
        all_loss = {opt: [] for opt in optimizers.keys()}
        all_time = {opt: [] for opt in optimizers.keys()}
        all_rmse_curve = {opt: [] for opt in optimizers.keys()}  # NEW

        if verbose:
            print(f"\n=== Observability = {obs:.2f} | Experiments = {num_experiments} ===")

        for i in range(num_experiments):
            rmse, rmse_curve, k, conv, loss, timing = run_experiment(
                data, sys, branch, data_generator, optimizers, **kwargs
            )

            for opt in optimizers.keys():
                all_rmse[opt].append(rmse[opt])
                all_k[opt].append(k[opt])
                all_conv[opt].append(conv[opt])
                all_loss[opt].append(loss[opt])
                all_time[opt].append(timing[opt])
                all_rmse_curve[opt].append(rmse_curve[opt])

            if verbose:
                print(f"\n[obs={obs:.2f}] Trial {i + 1:03d}/{num_experiments}")
                for opt in optimizers.keys():
                    print(
                        f"  {opt.upper():<15} | "
                        f"RMSE={rmse[opt]:.6f} | "
                        f"steps={k[opt]:4d} | "
                        f"conv={int(conv[opt])} | "
                        f"time={timing[opt]:.3f}s"
                    )

        stats_by_obs[obs] = {}
        mean_curve_by_opt = {}
        counts_by_opt = {}
        curves_path = os.path.join(out_dir, f"rmse_curves_obs{obs:.2f}.npz")
        savez_dict = {}
        for opt in optimizers.keys():
            all_rmse_opt = torch.as_tensor(all_rmse[opt], dtype=torch.float32)
            all_k_opt = torch.as_tensor(all_k[opt], dtype=torch.float32)
            all_conv_opt = torch.as_tensor(all_conv[opt], dtype=torch.float32)
            all_loss_opt = torch.as_tensor(all_loss[opt], dtype=torch.float32)
            all_time_opt = torch.as_tensor(all_time[opt], dtype=torch.float32)

            stats_by_obs[obs][opt] = {
                "rmse_mean": torch.nanmean(all_rmse_opt).item(),
                "k_mean": torch.nanmean(all_k_opt).item(),
                "conv_mean": torch.nanmean(all_conv_opt).item(),
                "loss_mean": torch.nanmean(all_loss_opt).item(),
                "time_mean": torch.nanmean(all_time_opt).item(),  # seconds
            }
            mean_curve, counts = _nanmean_curve(all_rmse_curve[opt])
            mean_curve_by_opt[opt] = mean_curve
            counts_by_opt[opt] = counts
            savez_dict[f"{opt}__mean"] = (mean_curve.cpu().numpy() if mean_curve is not None else np.array([]))
            savez_dict[f"{opt}__count"] = (counts.cpu().numpy() if counts is not None else np.array([]))
            np.savez(curves_path, **savez_dict)

        if verbose:
            print(f"\n--- Summary @ obs={obs:.2f} ---")
            for opt in optimizers.keys():
                s = stats_by_obs[obs][opt]
                space = " " * (25 - len(opt))
                print(
                    f"[{opt.upper()}]{space} "
                    f"Mean RMSE: {s['rmse_mean']:.8f}, "
                    f"Mean loss: {s['loss_mean']:.4f}, "
                    f"Mean steps: {s['k_mean']:.2f}, "
                    f"Mean conv: {s['conv_mean']:.2f}"
                )

    json_dump = {
        f"{obs:.2f}": {opt: {k: float(v) for k, v in d.items()} for opt, d in stats_by_obs[obs].items()}
        for obs in obs_levels
    }
    with open(results_json_path, "w") as f:
        json.dump(json_dump, f, indent=2)

    print(f"\nSaved mean results to: {results_json_path}")


def run_pf_ood_sweep(config):
    net_file = config["net_file"]
    verbose = config.get("verbose", False)

    obs_levels = config.get("observability_levels", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    num_experiments = int(config.get("num_experiments_per_obs", 100))

    pf_grid = config.get("pf_grid", [0.85, 0.75, 0.65, 0.55, 0.45, 0.35])

    out_dir = config.get("out_dir", "./results/observability_pf_sweep")
    _ensure_dir(out_dir)
    results_json_path = os.path.join(out_dir, "rmse_vs_pf_by_observability.json")

    data = parse_ieee_mat(net_file)
    sys = System(data["data"]["system"])
    branch = Branch(sys.branch)
    data_generator = DataGenerator()

    kwargs_base = deepcopy(config["generator_kwargs"])

    m = torch.load("../learn_prior/datasets/mean_polar.pt")
    cov = torch.load("../learn_prior/datasets/cov_polar.pt")
    NF_prior_config_path = "../learn_prior/configs/NF_4_0_128.json"
    VAE_config_path = "../learn_prior/configs/VAE_v0.4_config.json"

    optimizers = {
        "gn": GN_se(),
        "lm": LMOpt(loss_func=SELoss()),
        "lm_gs": LMOpt(loss_func=SELossGN(m=m, Q=cov)),
        "lm_nf": LMOpt(loss_func=SELossNF(NF_config_path=NF_prior_config_path, with_log_det=True)),
        "lm_nf_nld": LMOpt(loss_func=SELossNF(NF_config_path=NF_prior_config_path, with_log_det=False)),
        "lm_nf_lat": LMOpt(loss_func=SELossNFLat(NF_config_path=NF_prior_config_path, with_log_det=True), latent='dss'),
        # "lm_vae": LMOpt(loss_func=SELossVAE(VAE_config_path=VAE_config_path)),
        "lm_vae_lat": LMOpt(loss_func=SELossVAELat(VAE_config_path=VAE_config_path), latent='dss'),
        "lbfgs_nf": LBFGS_se(verbose=False, use_prior=True, prior_config_path=NF_prior_config_path),
    }

    pf_fixed = {2: 0.98, 3: 0.99}

    results = {}

    for obs in obs_levels:
        GNU_GNN_config_path = config.get("GNU_config", f"../learn_prior/configs/GNU_GNN_obs{obs}_config.json")
        GNU_GNN_config = json.load(open(GNU_GNN_config_path))
        ckpt_path = f"../learn_prior/GNU_GNN/models/{GNU_GNN_config.get('ckpt_name')}"

        GNU_GNN_model = GNU_Model(**GNU_GNN_config)
        edge_index = torch.stack([branch.i, branch.j]).to(torch.long)
        GNU_GNN_model.load_state_dict(torch.load(ckpt_path))
        GNU_GNN_model.edge_index = edge_index
        GNU_GNN_model.slk_bus = sys.slk_bus
        optimizers["gnu_gnn"] = GNU_GNN_model.optimize

        results[obs] = {}

        for pf in pf_grid:
            kwargs = deepcopy(kwargs_base)
            kwargs["sample"] = float(obs)

            kwargs["pf_mean_by_type_id"] = {1: float(pf), **pf_fixed}

            all_rmse = {opt: [] for opt in optimizers.keys()}

            if verbose:
                print(f"\n=== obs={obs:.2f} | pf_load_mean={pf:.3f} | N={num_experiments} ===")

            for _ in range(num_experiments):
                rmse, _, _, _, _, _ = run_experiment(data, sys, branch, data_generator, optimizers, **kwargs)
                for opt in optimizers.keys():
                    all_rmse[opt].append(rmse[opt])

            results[obs][pf] = {}
            for opt in optimizers.keys():
                x = torch.as_tensor(all_rmse[opt], dtype=torch.float32)
                results[obs][pf][opt] = {
                    "rmse_mean": torch.nanmean(x).item(),
                    "rmse_std": torch.std(x).item(),
                }

    json_dump = {
        f"{obs:.2f}": {
            f"{pf:.3f}": {opt: {k: float(v) for k, v in results[obs][pf][opt].items()} for opt in results[obs][pf].keys()}
            for pf in results[obs].keys()
        }
        for obs in results.keys()
    }
    with open(results_json_path, "w") as f:
        json.dump(json_dump, f, indent=2)

    print(f"\nSaved PF-OOD sweep results to: {results_json_path}")


def gen_step_curves_plots(config):
    out_dir = config.get("out_dir", "./results/observability_id_logscale")
    obs_levels = config.get("observability_levels", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    for obs in obs_levels:
        npz_path = os.path.join(out_dir, f"rmse_curves_obs{obs:.2f}.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path, allow_pickle=False)

        mean_curve_by_opt = {}
        for k in data.keys():
            if k.endswith("__mean"):
                opt = k.replace("__mean", "")
                arr = data[k]
                if arr.size == 0:
                    mean_curve_by_opt[opt] = None
                else:
                    mean_curve_by_opt[opt] = torch.from_numpy(arr)

        _plot_rmse_vs_step_for_obs(float(obs), mean_curve_by_opt, out_dir=out_dir)


def gen_plots(config):
    out_dir = config.get("out_dir", "./results/observability_sweep")
    _ensure_dir(out_dir)
    prefix = 'ID' if '_id_' in out_dir else 'OOD'
    results_json_path = os.path.join(out_dir, "mean_results_by_observability.json")
    stats_by_obs = json.load(open(results_json_path))
    obs_levels = list(stats_by_obs.keys())
    rmse_png = os.path.join(out_dir, "rmse_vs_observability.png")
    rmse_pdf = os.path.join(out_dir, "rmse_vs_observability.pdf")
    _plot_metric_vs_obs(
        obs_levels=obs_levels,
        stats_by_obs=stats_by_obs,
        metric_key="rmse_mean",
        ylabel="RMSE",
        title=f"RMSE vs Observability - {prefix}",
        save_path_png=rmse_png,
        save_path_pdf=rmse_pdf,
    )

    k_png = os.path.join(out_dir, "steps_vs_observability.png")
    k_pdf = os.path.join(out_dir, "steps_vs_observability.pdf")
    _plot_metric_vs_obs(
        obs_levels=obs_levels,
        stats_by_obs=stats_by_obs,
        metric_key="k_mean",
        ylabel="Mean number of steps",
        title=f"Steps vs Observability - {prefix}",
        save_path_png=k_png,
        save_path_pdf=k_pdf,
    )

    time_png = os.path.join(out_dir, "time_vs_observability.png")
    time_pdf = os.path.join(out_dir, "time_vs_observability.pdf")
    _plot_metric_vs_obs(
        obs_levels=obs_levels,
        stats_by_obs=stats_by_obs,
        metric_key="time_mean",
        ylabel="Mean runtime (sec)",
        title=f"Runtime vs Observability - {prefix}",
        save_path_png=time_png,
        save_path_pdf=time_pdf,
    )
    # gen_step_curves_plots(config)
    # gen_pf_ood_plots(config, x_as_distance=True)

def main():
    torch.set_default_dtype(torch.float32)
    config = json.load(open("./configs/analyze_optimizer_ood_config.json"))
    config.setdefault("observability_levels", [0.2])
    config.setdefault("num_experiments_per_obs", 10)
    config.setdefault("out_dir", "./results/test")
    run_all_experiments(config)
    gen_plots(config)

    # config = json.load(open("./configs/analyze_optimizer_id_config.json"))
    # config.setdefault("observability_levels", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # config.setdefault("num_experiments_per_obs", 100)
    # config.setdefault("out_dir", "./results/observability_id_logscale")
    # # run_all_experiments(config)
    # gen_plots(config)
    #
    # config = json.load(open("./configs/analyze_optimizer_ood_config.json"))
    # config.setdefault("observability_levels", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # config.setdefault("num_experiments_per_obs", 100)
    # config.setdefault("out_dir", "./results/observability_ood_logscale")
    # # run_all_experiments(config)
    # gen_plots(config)


if __name__ == "__main__":
    main()