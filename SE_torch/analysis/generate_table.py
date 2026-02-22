import json
import math
from pathlib import Path
import pandas as pd

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

# METHOD_ORDER = ['lm', 'lmgs', 'lmnf', 'wlmnf', 'lmis', 'lmil', 'lbfgsnf', 'gdnf']
METHOD_ORDER = ['lm', 'lmnf', 'wlmnf', 'lbfgsnf', 'gdnf']


def load_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def to_tabular_booktabs_multiindex(df: pd.DataFrame) -> str:
    return df.to_latex(
        escape=False,
        na_rep="—",
        multicolumn=True,
        multicolumn_format="c",
        column_format="l" + "c" * df.shape[1],
    )

# def json_to_long_df(results: dict, setting: str) -> pd.DataFrame:
#     rows = []
#     for ol_str, methods in results.items():
#         ol = float(ol_str)
#         for method_key, m in methods.items():
#             rows.append({
#                 "setting": setting,              # "ID" / "OOD"
#                 "ol": ol,
#                 "method": opt_name_map.get(method_key, method_key),
#                 "rmse": float(m["rmse_mean"]),
#                 "time": float(m["time_mean"]),
#             })
#     df = pd.DataFrame(rows)
#     df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
#     return df


def make_merged_table(id_results: dict, ood_results: dict) -> pd.DataFrame:
    df = pd.concat([
        json_to_long_df(id_results, "ID"),
        json_to_long_df(ood_results, "OOD"),
    ], ignore_index=True)

    df["cell"] = df.apply(lambda r: f"{r['rmse']:.4g} / {r['time']:.3g}s", axis=1)

    piv = df.pivot_table(index="method", columns=["ol", "setting"], values="cell", aggfunc="first")

    # order columns: OL ascending, then ID, OOD
    ols = sorted(df["ol"].unique())
    piv = piv.reindex(
        columns=pd.MultiIndex.from_product([ols, ["ID", "OOD"]], names=["ol", "setting"])
    )

    # pretty header names
    piv.columns = pd.MultiIndex.from_tuples(
        [(f"OL={ol:.2f}", setting) for (ol, setting) in piv.columns],
        names=["", ""],   # IMPORTANT: avoids "Observability" printing
    )

    piv = piv.reindex(METHOD_ORDER)
    piv.index.name = ""  # IMPORTANT: avoids extra "method" header row
    return piv


# def df_to_latex_table(df: pd.DataFrame, caption: str, label: str, resize: bool = True) -> str:
#     latex_tabular = df.to_latex(
#         escape=False,
#         na_rep="—",
#         multicolumn=True,
#         multicolumn_format="c",
#         column_format="l" + "c" * df.shape[1],
#     )
#
#     # Make it fit and look tighter
#     latex_tabular = (
#         "\\scriptsize\n"
#         "\\setlength{\\tabcolsep}{2pt}\n"
#         "\\renewcommand{\\arraystretch}{1.15}\n"
#         + latex_tabular
#     )
#
#     if resize:
#         latex_tabular = latex_tabular.replace(
#             "\\begin{tabular}",
#             "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}"
#         ).replace(
#             "\\end{tabular}",
#             "\\end{tabular}\n}"
#         )
#
#     return (
#         "\\begin{table}[t]\n"
#         "\\centering\n"
#         + latex_tabular +
#         f"\\caption{{{caption}}}\n"
#         f"\\label{{{label}}}\n"
#         "\\end{table}\n"
#     )


def json_to_long_df(results: dict, setting: str) -> pd.DataFrame:
    rows = []
    for ol_str, methods in results.items():
        ol = float(ol_str)
        for method_key, m in methods.items():
            rows.append({
                "setting": setting,              # "ID" / "OOD"
                "ol": ol,
                "method": opt_name_map.get(method_key, method_key),
                "rmse": float(m["rmse_mean"]),
                "time": float(m["time_mean"]),
            })
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df


def make_transposed_table(id_results: dict, ood_results: dict) -> pd.DataFrame:
    df = pd.concat([
        json_to_long_df(id_results, "ID"),
        json_to_long_df(ood_results, "OOD"),
    ], ignore_index=True)

    # cell content
    df["cell"] = df.apply(lambda r: f"{r['rmse']:.3f} / {r['time']:.2f}s", axis=1)

    # rows: (OL, setting), cols: method
    piv = df.pivot_table(index=["ol", "setting"], columns="method", values="cell", aggfunc="first")

    # order rows and cols
    ols = sorted(df["ol"].unique())
    piv = piv.reindex(index=pd.MultiIndex.from_product([ols, ["ID", "OOD"]], names=["ol", "setting"]))
    piv = piv.reindex(columns=METHOD_ORDER)

    # pretty row labels
    piv.index = [f"OL={ol:.2f} ({setting})" for (ol, setting) in piv.index]
    piv.index.name = ""

    return piv


def df_to_latex_table(df: pd.DataFrame, caption: str, label: str, resize: bool = False) -> str:
    latex_tabular = df.to_latex(
        escape=False,
        na_rep="—",
        column_format="l" + "c" * df.shape[1],
    )

    # Bigger font + comfy spacing
    latex_tabular = (
        "\\normalsize\n"                 # <- enlarge font (use \\large if it still fits)
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        + latex_tabular
    )

    if resize:
        latex_tabular = latex_tabular.replace(
            "\\begin{tabular}",
            "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}"
        ).replace(
            "\\end{tabular}",
            "\\end{tabular}\n}"
        )

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        + latex_tabular +
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
# def df_to_latex_table(df: pd.DataFrame, caption: str, label: str, resize: bool = True) -> str:
#     latex_tabular = df.to_latex(
#         escape=False,
#         na_rep="—",
#         column_format="l" + "c" * df.shape[1],
#     )
#
#     latex_tabular = (
#         "\\scriptsize\n"
#         "\\setlength{\\tabcolsep}{3pt}\n"
#         "\\renewcommand{\\arraystretch}{1.15}\n"
#         + latex_tabular
#     )
#
#     if resize:
#         latex_tabular = latex_tabular.replace(
#             "\\begin{tabular}",
#             "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}"
#         ).replace(
#             "\\end{tabular}",
#             "\\end{tabular}\n}"
#         )
#
#     return (
#         "\\begin{table}[t]\n"
#         "\\centering\n"
#         + latex_tabular +
#         f"\\caption{{{caption}}}\n"
#         f"\\label{{{label}}}\n"
#         "\\end{table}\n"
#     )


# -------- Usage --------
# transposed = make_transposed_table(id_results, ood_results)
# Path("table_id_ood_transposed.tex").write_text(
#     df_to_latex_table(
#         transposed,
#         caption="ID and OOD results (RMSE / time-to-convergence) across observability levels.",
#         label="tab:id_ood_rmse_time",
#         resize=True,
#     )
# )

# ---------------- Example usage ----------------
id_results = load_json("./results/observability_id_logscale/mean_results_by_observability.json")
ood_results = load_json("./results/observability_ood_logscale/mean_results_by_observability.json")
# merged = make_merged_table(id_results, ood_results)
# Path("table_id_ood.tex").write_text(
#     df_to_latex_table(
#         merged,
#         caption="ID and OOD results (RMSE / time-to-convergence) across observability levels.",
#         label="tab:id_ood_rmse_time",
#         resize=True,
#     )
# )
transposed = make_transposed_table(id_results, ood_results)
Path("table_id_ood_transposed.tex").write_text(
    df_to_latex_table(
        transposed,
        caption="ID and OOD results (RMSE / time-to-convergence) across observability levels.",
        label="tab:id_ood_rmse_time",
        resize=True,
    )
)
