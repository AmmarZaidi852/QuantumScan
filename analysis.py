"""
analysis.py - Turns results/results.csv into plots, tables, tests and a summary.

Reads (written by run_all.py):
    results/results.csv
    results/geometric_difference.csv

Writes:
    results/learning_curves.png - AUROC vs N, one panel per number of qubits
    results/summary_table.csv   - mean ± std of AUROC, sensitivity, specificity, F1, train time
    results/wilcoxon.csv        - paired Wilcoxon signed-rank tests (paired by seed)
    results/summary.txt         - plain-English summary of the results

Usage:
    python analysis.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = "results"
REFERENCE = "SVM-RBF"
QUANTUM = ["QK-Fidelity", "QK-Projected", "QK-NoEntangle"]
MODEL_ORDER = ["SVM-RBF", "LogReg"] + QUANTUM

# Fixed colour per model (colourblind-checked categorical palette), plus a
# distinct marker and line style so models are not identified by colour alone.
STYLE = {
    "SVM-RBF":       dict(color="#2a78d6", marker="o", linestyle="--"),
    "LogReg":        dict(color="#eb6834", marker="s", linestyle="--"),
    "QK-Fidelity":   dict(color="#1baf7a", marker="^", linestyle="-"),
    "QK-Projected":  dict(color="#eda100", marker="D", linestyle="-"),
    "QK-NoEntangle": dict(color="#e87ba4", marker="v", linestyle="-"),
}
METRICS = {"auroc": "AUROC", "sensitivity": "Sensitivity", "specificity": "Specificity",
           "f1": "F1", "train_time_s": "Train time (s)"}


# ---------------------------------------------------------------------------
# 1. Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(df, path):
    """AUROC vs N (mean ± std over seeds), one panel per number of qubits."""
    n_list = sorted(df["N"].unique())
    pca_list = sorted(df["n_components"].unique())
    fig, axes = plt.subplots(1, len(pca_list), figsize=(6 * len(pca_list), 4.5), sharey=True)

    for ax, n_comp in zip(np.atleast_1d(axes), pca_list):
        sub = df[df["n_components"] == n_comp]
        for k, model in enumerate(MODEL_ORDER):
            stats = sub[sub["model"] == model].groupby("N")["auroc"].agg(["mean", "std"])
            # small horizontal offset per model so the error bars don't overlap
            x = stats.index.values * (1 + 0.035 * (k - 2))
            ax.errorbar(x, stats["mean"], yerr=stats["std"], label=model, linewidth=2,
                        markersize=7, capsize=3, elinewidth=1.2, **STYLE[model])
        ax.set_xscale("log")
        ax.set_xticks(n_list)
        ax.set_xticklabels([str(n) for n in n_list])
        ax.minorticks_off()
        ax.set_title(f"{n_comp} qubits ({n_comp} PCA features)")
        ax.set_xlabel("Training patients (N)")
        ax.grid(True, color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    np.atleast_1d(axes)[0].set_ylabel("Test AUROC (mean ± std, 5 seeds)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODEL_ORDER), frameon=False)
    fig.suptitle("Pneumonia detection with few training patients: classical vs quantum kernels")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Summary table
# ---------------------------------------------------------------------------

def summary_table(df):
    """'mean ± std' strings per model, n_components and N."""
    g = df.groupby(["n_components", "N", "model"])
    table = pd.DataFrame({
        label: g[col].mean().map(lambda v: f"{v:.3f}") + " ± " + g[col].std().map(lambda v: f"{v:.3f}")
        for col, label in METRICS.items()
    }).reset_index()
    table["model"] = pd.Categorical(table["model"], MODEL_ORDER)
    return table.sort_values(["n_components", "N", "model"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Wilcoxon signed-rank tests
# ---------------------------------------------------------------------------

def paired_test(df, model_a, model_b, pair_keys):
    """Wilcoxon signed-rank test on AUROC of model_a vs model_b, paired on pair_keys.

    Returns (mean difference a - b, p-value, number of pairs).
    """
    a = df[df["model"] == model_a].set_index(pair_keys)["auroc"]
    b = df[df["model"] == model_b].set_index(pair_keys)["auroc"]
    a, b = a.align(b, join="inner")
    diff = a - b
    if np.allclose(diff, 0):
        return 0.0, np.nan, len(diff)  # identical results: test undefined
    return diff.mean(), wilcoxon(a, b).pvalue, len(diff)


def wilcoxon_tests(df):
    """Each quantum model vs SVM-RBF, plus the entanglement ablation
    (QK-Fidelity vs QK-NoEntangle).

    Per (n_components, N): paired by seed (5 pairs).
    'all' rows: pooled over N, paired by (N, seed) (20 pairs) - more power.
    """
    comparisons = [(q, REFERENCE) for q in QUANTUM] + [("QK-Fidelity", "QK-NoEntangle")]
    rows = []
    for model_a, model_b in comparisons:
        for n_comp in sorted(df["n_components"].unique()):
            sub = df[df["n_components"] == n_comp]
            groups = [(n, sub[sub["N"] == n], ["seed"]) for n in sorted(sub["N"].unique())]
            groups.append(("all", sub, ["N", "seed"]))
            for n, data, keys in groups:
                diff, p, n_pairs = paired_test(data, model_a, model_b, keys)
                rows.append({"model": model_a, "vs": model_b, "n_components": n_comp, "N": n,
                             "mean_auroc_diff": diff, "p_value": p, "n_pairs": n_pairs})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Plain-English summary
# ---------------------------------------------------------------------------

def describe(diff, p, alpha=0.05):
    """Turn an AUROC difference and p-value into words."""
    direction = "higher" if diff > 0 else "lower"
    if np.isnan(p):
        return "identical"
    if p < alpha:
        return f"significantly {direction} (diff {diff:+.3f}, p = {p:.3f})"
    return f"not significantly different (diff {diff:+.3f}, p = {p:.3f})"


def write_summary(df, tests, gdf):
    lines = []
    mean_auroc = df.groupby(["n_components", "N", "model"])["auroc"].mean()

    lines.append("WHO WON WHERE (highest mean test AUROC over 5 seeds)")
    for (n_comp, n), group in mean_auroc.groupby(level=["n_components", "N"]):
        group = group.droplevel(["n_components", "N"]).sort_values(ascending=False)
        best_q = group[QUANTUM].idxmax()
        lines.append(f"  {n_comp} qubits, N={n:<3}: winner {group.index[0]} ({group.iloc[0]:.3f}); "
                     f"best quantum {best_q} ({group[best_q]:.3f}) vs SVM-RBF ({group[REFERENCE]:.3f})")
    n_cells = len(mean_auroc.groupby(level=["n_components", "N"]))
    q_wins = sum(g.droplevel(["n_components", "N"]).idxmax() in QUANTUM
                 for _, g in mean_auroc.groupby(level=["n_components", "N"]))
    lines.append(f"  -> A quantum model had the best mean AUROC in {q_wins} of {n_cells} settings.")

    lines.append("")
    lines.append("QUANTUM vs SVM-RBF (Wilcoxon signed-rank, pooled over N, 20 pairs)")
    pooled = tests[tests["N"] == "all"]
    for _, r in pooled[pooled["vs"] == REFERENCE].iterrows():
        lines.append(f"  {r['n_components']} qubits: {r['model']} is "
                     f"{describe(r['mean_auroc_diff'], r['p_value'])} than SVM-RBF")

    lines.append("")
    lines.append("DID ENTANGLEMENT HELP? (QK-Fidelity vs QK-NoEntangle, pooled over N)")
    for _, r in pooled[pooled["vs"] == "QK-NoEntangle"].iterrows():
        lines.append(f"  {r['n_components']} qubits: with entanglement, AUROC is "
                     f"{describe(r['mean_auroc_diff'], r['p_value'])}")

    lines.append("")
    lines.append("GEOMETRIC DIFFERENCE g(K_classical || K_quantum) (Huang et al. 2021)")
    lines.append("  g ranges from ~1 (classical kernel can learn everything the quantum kernel can)")
    lines.append("  to order sqrt(N) (room for a quantum advantage). Median [min-max] over seeds:")
    g_stats = gdf.groupby(["n_components", "N"]).agg(
        g=("g", "median"), g_min=("g", "min"), g_max=("g", "max"), sqrt_N=("sqrt_N", "first"))
    for (n_comp, n), r in g_stats.iterrows():
        lines.append(f"  {n_comp} qubits, N={n:<3}: g = {r['g']:.2f} [{r['g_min']:.2f}-{r['g_max']:.2f}]"
                     f"  (sqrt(N) = {r['sqrt_N']:.1f}, median is {100 * r['g'] / r['sqrt_N']:.0f}% of it)")

    lines.append("")
    lines.append("NOTE: with 5 seeds the smallest possible two-sided Wilcoxon p-value is 0.0625,")
    lines.append("so per-N tests in wilcoxon.csv can never reach p < 0.05; use the pooled rows.")
    return "\n".join(lines)


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))
    gdf = pd.read_csv(os.path.join(RESULTS_DIR, "geometric_difference.csv"))

    plot_learning_curves(df, os.path.join(RESULTS_DIR, "learning_curves.png"))

    table = summary_table(df)
    table.to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"), index=False)

    tests = wilcoxon_tests(df)
    tests.to_csv(os.path.join(RESULTS_DIR, "wilcoxon.csv"), index=False)

    summary = write_summary(df, tests, gdf)
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(table.to_string(index=False))
    print()
    print(summary)
    print(f"\nSaved learning_curves.png, summary_table.csv, wilcoxon.csv and summary.txt to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
