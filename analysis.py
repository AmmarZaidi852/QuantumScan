"""
analysis.py - Turns results/results.csv into plots, tables, tests and a summary.

Reads (written by run_all.py, and ablation.py if it was run):
    results/results.csv
    results/geometric_difference.csv
    results/ablation.csv        - optional: QK-Hybrid / QK-Aligned without entanglement
    results/confirm/*.csv       - optional: the same files for the confirmation seeds 5-9
                                  (confirm.py). If present, every table, test and the plot
                                  use all 10 seeds, plus the pre-registered primary test.

Writes:
    results/learning_curves.png - AUROC vs N, one panel per number of qubits
    results/summary_table.csv   - mean ± std of AUROC, sensitivity, specificity, F1, train time
    results/stability.csv       - spread of AUROC over seeds at N=25 and N=50
    results/wilcoxon.csv        - paired Wilcoxon signed-rank tests (paired by seed), Holm-corrected
    results/primary_test.csv    - the pre-registered confirmation test (if confirm.py was run)
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
CONFIRM_DIR = os.path.join(RESULTS_DIR, "confirm")
# Pre-registered primary test of the confirmation run (see confirm.py): one
# two-sided Wilcoxon test, pooled over N, on the new seeds only, no correction.
PRIMARY = {"model": "QK-NoEntangle", "vs": "SVM-RBF", "n_components": 8}
EXPLORATORY_SEEDS = [0, 1, 2, 3, 4]
CONFIRM_SEEDS = [5, 6, 7, 8, 9]
REFERENCE = "SVM-RBF"
CLASSICAL = ["SVM-RBF", "LogReg"]
QUANTUM = ["QK-Fidelity", "QK-Projected", "QK-NoEntangle", "QK-Hybrid", "QK-Aligned"]
MODEL_ORDER = CLASSICAL + QUANTUM
STABILITY_N = [25, 50]   # small-N settings where the spread over seeds matters most
RIVAL_MARGIN = 0.01      # "rivals" SVM-RBF = mean AUROC within +-this; "beats" = more than this higher
ALPHA = 0.05
# Entanglement ablations: model with ZZ gates vs the same model without them.
# The -NoEnt models come from ablation.py and are only used in these tests.
ENTANGLEMENT_PAIRS = [("QK-Fidelity", "QK-NoEntangle"),
                      ("QK-Hybrid", "QK-Hybrid-NoEnt"),
                      ("QK-Aligned", "QK-Aligned-NoEnt")]

# Fixed colour per model (colourblind-checked categorical palette), plus a
# distinct marker and line style so models are not identified by colour alone.
STYLE = {
    "SVM-RBF":       dict(color="#2a78d6", marker="o", linestyle="--"),
    "LogReg":        dict(color="#eb6834", marker="s", linestyle="--"),
    "QK-Fidelity":   dict(color="#1baf7a", marker="^", linestyle="-"),
    "QK-Projected":  dict(color="#eda100", marker="D", linestyle="-"),
    "QK-NoEntangle": dict(color="#e87ba4", marker="v", linestyle="-"),
    "QK-Hybrid":     dict(color="#008300", marker="P", linestyle="-"),
    "QK-Aligned":    dict(color="#6250d6", marker="X", linestyle="-"),
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
    fig, axes = plt.subplots(1, len(pca_list), figsize=(6.5 * len(pca_list), 5), sharey=True)

    for ax, n_comp in zip(np.atleast_1d(axes), pca_list):
        sub = df[df["n_components"] == n_comp]
        for k, model in enumerate(MODEL_ORDER):
            stats = sub[sub["model"] == model].groupby("N")["auroc"].agg(["mean", "std"])
            # small horizontal offset per model so the error bars don't overlap
            x = stats.index.values * (1 + 0.025 * (k - (len(MODEL_ORDER) - 1) / 2))
            ax.errorbar(x, stats["mean"], yerr=stats["std"], label=model, linewidth=2,
                        markersize=7, capsize=2, elinewidth=1, **STYLE[model])
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

    np.atleast_1d(axes)[0].set_ylabel(f"Test AUROC (mean ± std, {df['seed'].nunique()} seeds)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Pneumonia detection with few training patients: classical vs quantum kernels")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
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
# 3. Stability: spread of AUROC over seeds
# ---------------------------------------------------------------------------

def stability_table(df):
    """Std and range (max - min) of test AUROC over the seeds, at small N.

    A model whose AUROC depends a lot on which 25 patients it was given is
    less trustworthy, even if its mean is good. std_vs_svm < 1 means the
    model is more stable than SVM-RBF in that setting.
    """
    sub = df[df["N"].isin(STABILITY_N)]
    g = sub.groupby(["n_components", "N", "model"])["auroc"]
    table = pd.DataFrame({"auroc_mean": g.mean(), "auroc_std": g.std(),
                          "auroc_range": g.max() - g.min()}).reset_index()
    svm_std = table[table["model"] == REFERENCE].set_index(["n_components", "N"])["auroc_std"]
    table["std_vs_svm"] = table["auroc_std"] / svm_std.reindex(
        pd.MultiIndex.from_frame(table[["n_components", "N"]])).values
    table["type"] = np.where(table["model"].isin(QUANTUM), "quantum", "classical")
    table["model"] = pd.Categorical(table["model"], MODEL_ORDER)
    return table.sort_values(["n_components", "N", "model"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Wilcoxon signed-rank tests
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


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (NaN entries are ignored).

    Sort the m p-values; the k-th smallest (k = 0, 1, ...) is multiplied by
    (m - k), and a running maximum keeps the adjusted values in order.
    Controls the chance of ANY false positive in the family at alpha.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def wilcoxon_tests(df):
    """Each quantum model vs SVM-RBF, plus the entanglement ablations
    (each entangled model vs the same model without ZZ gates, when available).

    Per (n_components, N): paired by seed (one pair per seed).
    'all' rows: pooled over N, paired by (N, seed) (4 pairs per seed) - more power.

    Holm correction is applied separately to two families: the pooled tests
    (the ones to report) and the per-N tests.
    """
    available = set(df["model"])
    comparisons = [(q, REFERENCE) for q in QUANTUM] + [
        (a, b) for a, b in ENTANGLEMENT_PAIRS if a in available and b in available]
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
    tests = pd.DataFrame(rows)
    pooled = tests["N"] == "all"
    tests["p_holm"] = np.nan
    for family in (pooled, ~pooled):
        tests.loc[family, "p_holm"] = holm(tests.loc[family, "p_value"])
    return tests


def primary_test(df):
    """The pre-registered confirmation test, on the confirmation seeds only.
    The same contrast on the exploratory seeds is reported for reference."""
    rows = []
    for label, seeds in (("confirmation (primary)", CONFIRM_SEEDS), ("exploratory", EXPLORATORY_SEEDS)):
        data = df[(df["n_components"] == PRIMARY["n_components"]) & df["seed"].isin(seeds)]
        diff, p, n_pairs = paired_test(data, PRIMARY["model"], PRIMARY["vs"], ["N", "seed"])
        wins = (data[data["model"] == PRIMARY["model"]].set_index(["N", "seed"])["auroc"]
                - data[data["model"] == PRIMARY["vs"]].set_index(["N", "seed"])["auroc"])
        rows.append({"seeds": label, "seed_list": " ".join(map(str, seeds)), **PRIMARY,
                     "mean_auroc_diff": diff, "p_value": p, "n_pairs": n_pairs,
                     "n_higher": int((wins > 0).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Plain-English summary
# ---------------------------------------------------------------------------

def describe(r):
    """Turn a test row (AUROC difference, raw and Holm p-value) into words."""
    diff, p, p_holm = r["mean_auroc_diff"], r["p_value"], r["p_holm"]
    if np.isnan(p):
        return "identical"
    direction = "higher" if diff > 0 else "lower"
    stats = f"diff {diff:+.3f}, p = {p:.3f}, Holm p = {p_holm:.3f}"
    if p_holm < ALPHA:
        return f"significantly {direction} ({stats})"
    if p < ALPHA:
        return f"{direction}, but not significant after Holm correction ({stats})"
    return f"not significantly different ({stats})"


def write_summary(df, tests, stability, gdf, primary=None):
    lines = []
    n_seeds = df["seed"].nunique()
    mean_auroc = df.groupby(["n_components", "N", "model"])["auroc"].mean()
    settings = list(mean_auroc.groupby(level=["n_components", "N"]).groups)
    pooled = tests[tests["N"] == "all"]
    vs_svm = pooled[pooled["vs"] == REFERENCE]
    ent = pooled[pooled["vs"] != REFERENCE]

    if primary is not None and len(primary):
        r = primary.iloc[0]
        confirmed = r["p_value"] < ALPHA and r["mean_auroc_diff"] > 0
        lines.append("PRE-REGISTERED CONFIRMATION (new seeds 5-9, decided before running)")
        lines.append(f"  {r['model']} vs {r['vs']}, {r['n_components']} qubits, two-sided Wilcoxon signed-rank,")
        lines.append(f"  paired by (N, seed), pooled over N, seeds {r['seed_list']} only; one test, no correction.")
        for _, row in primary.iterrows():
            lines.append(f"  {row['seeds']:<22}: diff {row['mean_auroc_diff']:+.3f}, p = {row['p_value']:.4f}, "
                         f"{r['model']} higher in {row['n_higher']} of {row['n_pairs']} pairs")
        lines.append("  -> " + ("CONFIRMED: the edge replicates on new seeds." if confirmed else
                                "NOT CONFIRMED: the edge does not replicate on new seeds at p < 0.05."))
        lines.append("")

    lines.append(f"BOTTOM LINE (all {n_seeds} seeds)")
    for n_comp, grp in vs_svm.groupby("n_components"):
        best = grp.loc[grp["mean_auroc_diff"].idxmax()]
        lines.append(f"  {n_comp} qubits: best quantum model vs SVM-RBF (pooled over N) is {best['model']}, "
                     f"{best['mean_auroc_diff']:+.3f} AUROC, Holm p = {best['p_holm']:.3f}.")
    n_sig = int(((vs_svm["p_holm"] < ALPHA) & (vs_svm["mean_auroc_diff"] > 0)).sum())
    lines.append(f"  {n_sig} of {len(vs_svm)} quantum-vs-SVM-RBF comparisons are significantly better "
                 f"after Holm correction.")
    if len(ent):
        ent_better = int(((ent["p_holm"] < ALPHA) & (ent["mean_auroc_diff"] > 0)).sum())
        ent_worse = int(((ent["p_holm"] < ALPHA) & (ent["mean_auroc_diff"] < 0)).sum())
        lines.append(f"  Entanglement (ZZ gates) significantly helped in {ent_better} and significantly hurt "
                     f"in {ent_worse} of {len(ent)} with/without-ZZ comparisons.")
        no_ent = {b for _, b in ENTANGLEMENT_PAIRS}
        best_models = set(vs_svm.loc[vs_svm.groupby("n_components")["mean_auroc_diff"].idxmax(), "model"])
        if best_models <= no_ent:
            lines.append("  The best quantum model uses no entanglement at all, so any edge it has is not")
            lines.append("  a quantum effect: its kernel can be computed classically.")
    lines.append("")

    lines.append(f"WHO WON WHERE (highest mean test AUROC over {n_seeds} seeds)")
    q_wins = 0
    for n_comp, n in settings:
        group = mean_auroc.loc[(n_comp, n)].sort_values(ascending=False)
        best_q = group[QUANTUM].idxmax()
        q_wins += group.index[0] in QUANTUM
        lines.append(f"  {n_comp} qubits, N={n:<3}: winner {group.index[0]} ({group.iloc[0]:.3f}); "
                     f"best quantum {best_q} ({group[best_q]:.3f}) vs SVM-RBF ({group[REFERENCE]:.3f})")
    lines.append(f"  -> A quantum model had the best mean AUROC in {q_wins} of {len(settings)} settings.")

    lines.append("")
    lines.append(f"WHICH QUANTUM MODEL RIVALS OR BEATS SVM-RBF? (mean AUROC difference; "
                 f"'beats' = more than {RIVAL_MARGIN} higher, 'rivals' = within +-{RIVAL_MARGIN})")
    for q in QUANTUM:
        groups = {"beats": [], "rivals": [], "behind": []}
        for n_comp, n in settings:
            d = mean_auroc[(n_comp, n, q)] - mean_auroc[(n_comp, n, REFERENCE)]
            word = "beats" if d > RIVAL_MARGIN else "rivals" if d >= -RIVAL_MARGIN else "behind"
            groups[word].append(f"{n_comp}q N={n} ({d:+.3f})")
        lines.append(f"  {q}:")
        for word, items in groups.items():
            lines.append(f"    {word:<7} {len(items)}/{len(settings)}: {', '.join(items) or '-'}")

    lines.append("")
    lines.append("IS IT SIGNIFICANT? Quantum vs SVM-RBF (Wilcoxon signed-rank, pooled over N,")
    lines.append(f"{4 * n_seeds} pairs per qubit count; Holm correction over all {len(pooled)} pooled tests)")
    for _, r in vs_svm.iterrows():
        lines.append(f"  {r['n_components']} qubits: {r['model']:<13} is {describe(r)} than SVM-RBF")
    sig_better = vs_svm[(vs_svm["p_holm"] < ALPHA) & (vs_svm["mean_auroc_diff"] > 0)]
    if len(sig_better):
        names = ", ".join(f"{r['model']} ({r['n_components']} qubits)" for _, r in sig_better.iterrows())
        lines.append(f"  -> Significantly better than SVM-RBF after Holm correction: {names}.")
    else:
        lines.append("  -> No quantum model is significantly better than SVM-RBF after Holm correction.")

    lines.append("")
    lines.append(f"STABILITY: spread (std) of test AUROC over the {n_seeds} seeds at N = "
                 f"{', '.join(map(str, STABILITY_N))}")
    lines.append("  In brackets: ratio to SVM-RBF's std (< 1 = more stable than SVM-RBF).")
    for (n_comp, n), grp in stability.groupby(["n_components", "N"], observed=True):
        cells = [f"{r['model']} {r['auroc_std']:.3f}"
                 + ("" if r["model"] == REFERENCE else f" [{r['std_vs_svm']:.2f}]")
                 for _, r in grp.iterrows()]
        lines.append(f"  {n_comp} qubits, N={n:<3}: " + ", ".join(cells))
    median_std = stability.groupby("type")["auroc_std"].median()
    lines.append(f"  Median std over these settings: classical {median_std['classical']:.3f}, "
                 f"quantum {median_std['quantum']:.3f}.")
    for q in QUANTUM:
        ratios = stability[stability["model"] == q]["std_vs_svm"]
        lines.append(f"  {q:<13}: spread smaller than SVM-RBF in {int((ratios < 1).sum())} of "
                     f"{len(ratios)} settings (median ratio {ratios.median():.2f})")
    rel_err = 100 / np.sqrt(2 * (n_seeds - 1))  # approx. relative standard error of a sample std
    lines.append(f"  Caution: a std from {n_seeds} seeds is itself uncertain by roughly +-{rel_err:.0f}%, so only")
    lines.append("  ratios well away from 1 mean much.")

    lines.append("")
    lines.append("DID ENTANGLEMENT HELP? (same model with vs without ZZ gates, pooled over N)")
    for _, r in ent.iterrows():
        lines.append(f"  {r['n_components']} qubits: {r['model']} vs {r['vs']}: with entanglement, "
                     f"AUROC is {describe(r)}")
    missing = [a for a, _ in ENTANGLEMENT_PAIRS if a not in set(ent["model"])]
    if missing:
        lines.append(f"  (no ablation for {', '.join(missing)}: run ablation.py)")
    if len(ent):
        helped = ent[(ent["p_holm"] < ALPHA) & (ent["mean_auroc_diff"] > 0)]
        hurt = ent[(ent["p_holm"] < ALPHA) & (ent["mean_auroc_diff"] < 0)]
        lines.append(f"  -> Entanglement significantly helped in {len(helped)} and significantly hurt "
                     f"in {len(hurt)} of {len(ent)} comparisons.")
    lines.append("  Without ZZ gates every patient is a product state, and the fidelity kernel")
    lines.append("  is a product of one-qubit cos^2 terms that a classical computer evaluates directly.")

    if "bandwidth" in df.columns:
        lines.append("")
        lines.append("BANDWIDTH CHOSEN (calibrated on training data only; median [min-max] over N and seeds)")
        bw = df[df["model"].isin(QUANTUM)].groupby(["n_components", "model"])["bandwidth"]
        for (n_comp, model), values in bw:
            lines.append(f"  {n_comp} qubits, {model:<13}: {values.median():.3f} "
                         f"[{values.min():.3f}-{values.max():.3f}]")

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
    lines.append(f"NOTE: with {n_seeds} seeds the smallest possible two-sided Wilcoxon p-value per N is "
                 f"{2 / 2 ** n_seeds:.4f};")
    lines.append("the pooled rows (all N together) are the ones to report.")
    return "\n".join(lines)


def main():
    def load(name, required=True):
        """One results file, with the confirmation seeds appended when they exist."""
        parts = [os.path.join(d, name) for d in (RESULTS_DIR, CONFIRM_DIR)]
        frames = [pd.read_csv(path) for path in parts if os.path.exists(path)]
        if not frames and not required:
            return None
        return pd.concat(frames, ignore_index=True)

    df = load("results.csv")
    gdf = load("geometric_difference.csv")
    ablation = load("ablation.csv", required=False)
    if ablation is None:
        ablation = df.iloc[:0]

    primary = None
    if df["seed"].isin(CONFIRM_SEEDS).any():
        primary = primary_test(df)
        primary.to_csv(os.path.join(RESULTS_DIR, "primary_test.csv"), index=False)

    plot_learning_curves(df, os.path.join(RESULTS_DIR, "learning_curves.png"))

    table = summary_table(df)
    table.to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"), index=False)

    stability = stability_table(df)
    stability.to_csv(os.path.join(RESULTS_DIR, "stability.csv"), index=False)

    tests = wilcoxon_tests(pd.concat([df, ablation], ignore_index=True))
    tests.to_csv(os.path.join(RESULTS_DIR, "wilcoxon.csv"), index=False)

    summary = write_summary(df, tests, stability, gdf, primary)
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(table.to_string(index=False))
    print()
    print(summary)
    print(f"\nSaved learning_curves.png, summary_table.csv, stability.csv, wilcoxon.csv and "
          f"summary.txt to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
