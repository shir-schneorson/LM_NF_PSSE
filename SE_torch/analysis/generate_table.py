"""Table I rendering: RMSE / time-to-convergence, ID vs OOD.

Consumes the two result JSONs written by `run_experiments.py --experiment table`
(one for ID, one for OOD), each mapping an observability level to
``{"optimizers": {method: {"rmse_mean", "time_mean", ...}}}``, and emits both a
LaTeX table and a plain-text table for the five methods benchmarked in the
paper's Table I.
"""

import json
from pathlib import Path

import pandas as pd


# Paper method order and display names for Table I.
METHOD_ORDER = ["proxnet", "flower", "lm", "lm_nf", "lbfgs_nf"]
METHOD_LABELS = {
    "proxnet": "ProxNet",
    "flower": "FLOWER",
    "lm": "LM",
    "lm_nf": "LMNF",
    "lbfgs_nf": "LBFGSNF",
}


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def _long_df(results, setting):
    """Flatten a `stats_by_obs` dict into rows of (setting, ol, method, rmse, time)."""
    rows = []
    for ol_str, entry in results.items():
        ol = float(ol_str)
        opt = entry.get("optimizers", entry)
        for method, m in opt.items():
            if method not in METHOD_LABELS:
                continue
            rows.append({
                "setting": setting,
                "ol": ol,
                "method": METHOD_LABELS[method],
                "rmse": float(m["rmse_mean"]),
                "time": float(m["time_mean"]),
            })
    return pd.DataFrame(rows)


def make_table(id_results, ood_results):
    """Return a pivot table: rows = (OL, setting), cols = method, cells = 'rmse / time'."""
    df = pd.concat([_long_df(id_results, "ID"), _long_df(ood_results, "OOD")],
                   ignore_index=True)
    df["cell"] = df.apply(lambda r: f"{r['rmse']:.4f} / {r['time']:.2f}s", axis=1)

    piv = df.pivot_table(index=["ol", "setting"], columns="method",
                         values="cell", aggfunc="first")
    ols = sorted(df["ol"].unique())
    piv = piv.reindex(index=pd.MultiIndex.from_product([ols, ["ID", "OOD"]],
                                                       names=["ol", "setting"]))
    ordered_cols = [METHOD_LABELS[m] for m in METHOD_ORDER
                    if METHOD_LABELS[m] in piv.columns]
    piv = piv.reindex(columns=ordered_cols)
    piv.index = [f"OL={ol:.2f} ({setting})" for (ol, setting) in piv.index]
    piv.index.name = ""
    return piv


def to_latex(df, caption, label):
    tabular = df.to_latex(escape=False, na_rep="—",
                          column_format="l" + "c" * df.shape[1])
    tabular = ("\\normalsize\n"
               "\\setlength{\\tabcolsep}{4pt}\n"
               "\\renewcommand{\\arraystretch}{1.2}\n" + tabular)
    return ("\\begin{table}[t]\n\\centering\n" + tabular
            + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


def build_table(id_json, ood_json, out_tex):
    """Load the ID/OOD result JSONs, print the table, and write the LaTeX file."""
    id_results = load_json(id_json)
    ood_results = load_json(ood_json)
    table = make_table(id_results, ood_results)
    print("\n=== Table I: RMSE / time-to-convergence (ID vs OOD) ===")
    print(table.to_string())
    Path(out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(out_tex).write_text(to_latex(
        table,
        caption="ID and OOD results (RMSE / time-to-convergence) across observability levels.",
        label="tab:id_ood_rmse_time",
    ))
    print(f"\nSaved LaTeX table to: {out_tex}")
    return table


if __name__ == "__main__":
    build_table(
        "./results/table_id/mean_results_by_observability.json",
        "./results/table_ood/mean_results_by_observability.json",
        "./results/table/table_id_ood.tex",
    )
