# LMNF: Levenberg–Marquardt with Normalizing-Flow Priors for Power System State Estimation

Reproducible code for the paper *"Levenberg–Marquardt with Normalizing-Flow
Priors for Power System State Estimation"* (LMNF). It reproduces the paper's
experiments on the IEEE 118-bus system:

* **Figure 1 / Figure 2** — RMSE vs. observability, in-distribution (ID) and
  out-of-distribution (OOD), comparing the generative-prior estimators
  **LMGS**, **LMNF-NLD**, and **LMNF**.
* **Table I** — RMSE and time-to-convergence at observability {0.3, 0.5, 0.7},
  ID and OOD, comparing **ProxNet**, **FLOWER**, **LM**, **LMNF**, **LBFGSNF**.

## Install
```bash
pip install -r requirements.txt
```
Python 3.10+ recommended. A GPU is optional; the default config runs on CPU.

## Download the model checkpoints and datasets
Large artifacts are not committed. Before running, download them into place:

* Model checkpoints — see `SE_torch/learn_prior/models/DOWNLOAD.md` and
  `SE_torch/learn_prior/FM/models/DOWNLOAD.md`.
* The in-distribution operating-point set `val_dataset_vam.pt` (needed for the
  ID figure/table) and, if retraining, the full state datasets — see
  `SE_torch/data_parser/README.md`.

The OOD operating points and all normalization/prior tensors are already
bundled, so the **OOD figure is runnable as soon as the NF checkpoint is
downloaded**.

## Run the experiments
```bash
cd SE_torch/analysis
python run_experiments.py --experiment fig-id     # Figure 1  (ID)
python run_experiments.py --experiment fig-ood    # Figure 2  (OOD)
python run_experiments.py --experiment table      # Table I   (ID + OOD)
python run_experiments.py --experiment all        # everything
```
Useful flags: `--num-experiments N` (trials per observability level),
`--seed S` (override the base seed), `--no-show` (save figures without opening a
window), `--verbose`, `--config PATH`.

Outputs are written under `SE_torch/analysis/results/`:
`observability_id/` and `observability_ood/` hold the RMSE-vs-observability
figures (`.png`/`.pdf`) and JSON results; `table/` holds the LaTeX table and its
JSON.

**Reproducibility.** Every trial is seeded (`base_seed + trial_index * seed_step`,
configured in `configs/experiments.json`), so RMSE curves and table numbers are
identical on every run. Wall-clock times in Table I depend on your hardware.

**Efficiency note.** ProxNet and FLOWER are used only in Table I, so they are
built and evaluated only at the table observability levels — never during the
figure sweep. ProxNet uses a separate checkpoint per observability level; FLOWER
uses a single observability-agnostic model.

## Repository layout
```
SE_torch/
  data_generator.py            # noisy AC measurements from ground-truth states
  utils.py                     # init_start_point, normalize_measurements, RMSE
  PF_equations/                # AC measurement model (cartesian / polar)
  net_preprocess/              # IEEE .mat parsing, measurement composition
  optimizers/
    LM_opt.py                  # Levenberg-Marquardt (LM, LMGS, LMNF, LMNF-NLD)
    FO_se.py                   # L-BFGS estimator (LBFGSNF)
    Flower_se.py               # flow-matching sampler (FLOWER)
    se_loss.py                 # MAP loss functions (data-fit + prior variants)
    NR_acpf.py                 # Newton-Raphson AC power flow
  learn_prior/
    NF/                        # normalizing-flow prior + training (nf_model.py)
    FM/                        # flow-matching (FLOWER) model + training
    ProxLinear/                # unrolled prox-linear net (ProxNet) + training
    PIGNN/                     # data/estimator helpers reused by ProxNet (not a baseline)
    pca_gaussian_prior.py      # empirical Gaussian prior (LMGS)
    configs/                   # model configs referenced by the experiments
    models/                    # checkpoints (download; see DOWNLOAD.md)
  data_parser/data/            # bundled small tensors; large datasets download
  analysis/
    run_experiments.py         # <-- entry point for all experiments
    plotting.py                # RMSE-vs-observability figures
    generate_table.py          # Table I (LaTeX + text)
    configs/experiments.json   # single experiment config
```

## Retrain the priors (optional)
All training scripts read state datasets from `SE_torch/data_parser/data/`
(download the `*_dataset_*.pt` files first — see `data_parser/README.md`).

* **Normalizing-flow prior (LMNF):**
  `python -m SE_torch.learn_prior.NF.nf_model` — see `main(config_path)`; config
  `learn_prior/configs/nf_model_config_smart_v0.1.json`.
* **ProxNet (per observability level):**
  `python -m SE_torch.learn_prior.ProxLinear.train_prox_linear` with the
  `learn_prior/configs/ProxLinear_paper_obs0_{3,5,7}_config.json` configs.
* **FLOWER / flow matching (FM_v0.6):**
  `cd SE_torch/learn_prior/FM && python FM.py -c ../configs/FM_v0.6_config.json`.
  The v0.6 prior is trained on the slack-reduced state (`input_dim = 234`);
  older 236-dim FM checkpoints are not compatible.
* **Gaussian prior (LMGS):** `SE_torch/learn_prior/pca_gaussian_prior.py`.

## Citation
If you use this code, please cite the paper (see the accompanying PDF / arXiv
entry).
