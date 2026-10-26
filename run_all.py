"""
run_all.py - Runs every QuantumScan experiment and saves the results.

Experiment grid:
    n_components (PCA features) : 4, 8
    N (training patients)       : 25, 50, 100, 200
    seeds                       : 0, 1, 2, 3, 4

For every (n_components, N, seed) combination, ALL models get exactly the
same training sample, the same PCA features and the same fixed test set.

Output:
    results/results.csv              - one row per (model, n_components, N, seed)
    results/summary.csv              - mean and std of every metric per (model, n_components, N)
    results/geometric_difference.csv - g(K_classical || K_quantum) per (n_components, N, seed)

Usage:
    python run_all.py
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel

from classical import CLASSICAL_MODELS
from data import load_data, make_experiment
from quantum import QUANTUM_MODELS, geometric_difference

N_COMPONENTS_LIST = [4, 8]
N_LIST = [25, 50, 100, 200]
SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Models to evaluate: display name -> factory(seed) -> unfitted estimator.
# ---------------------------------------------------------------------------
MODELS = dict(CLASSICAL_MODELS)

# Quantum models (quantum.py). They are run on the same PCA features,
# training samples, seeds and test set as the classical models.
MODELS.update(QUANTUM_MODELS)


def get_scores(model, X):
    """Continuous score for the positive class (pneumonia), used for AUROC."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass  # e.g. SVC without probability=True
    return model.decision_function(X)


def evaluate(model, X_test, y_test):
    """Compute all test metrics for a fitted model."""
    y_pred = model.predict(X_test)
    scores = get_scores(model, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y_test, scores),
        "accuracy": accuracy_score(y_test, y_pred),
        "sensitivity": tp / (tp + fn),   # pneumonia cases correctly detected
        "specificity": tn / (tn + fp),   # normal cases correctly cleared
        "f1": f1_score(y_test, y_pred),
    }


def run_geometric_difference(fitted, Z_train):
    """g between the tuned classical RBF kernel (SVM-RBF) and the tuned
    quantum fidelity kernel (QK-Fidelity) on this run's training sample."""
    gamma = fitted["SVM-RBF"].best_params_["gamma"]
    if gamma == "scale":  # same rule scikit-learn uses for SVC(gamma="scale")
        gamma = 1.0 / (Z_train.shape[1] * Z_train.var())
    K_classical = rbf_kernel(Z_train, gamma=gamma)
    K_quantum = fitted["QK-Fidelity"].best_estimator_.K_train_
    return geometric_difference(K_classical, K_quantum)


def summarize(df):
    """Mean and std of each metric per model, n_components and N."""
    metrics = ["auroc", "accuracy", "sensitivity", "specificity", "f1", "train_time_s"]
    return df.groupby(["model", "n_components", "N"])[metrics].agg(["mean", "std"])


def print_auroc_table(df):
    """Print 'mean ± std' AUROC with one row per (n_components, model), one column per N."""
    g = df.groupby(["n_components", "model", "N"])["auroc"].agg(["mean", "std"])
    cells = g.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    table = cells.unstack("N")
    table.columns = [f"N={n}" for n in table.columns]
    print("\nTest AUROC (mean ± std over seeds):\n")
    print(table.to_string())


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading PneumoniaMNIST ...")
    X_train_full, y_train_full, X_test, y_test = load_data()
    print(f"  train pool: {len(y_train_full)} images, "
          f"fixed test set: {len(y_test)} images ({int(y_test.sum())} pneumonia)")

    rows, g_rows = [], []
    for n_components in N_COMPONENTS_LIST:
        for n in N_LIST:
            for seed in SEEDS:
                # Same sample + features for every model in this loop.
                Z_train, y_train, Z_test, y_te = make_experiment(
                    X_train_full, y_train_full, X_test, y_test, n, seed, n_components
                )
                fitted = {}
                for name, factory in MODELS.items():
                    model = factory(seed)
                    start = time.perf_counter()
                    model.fit(Z_train, y_train)
                    train_time = time.perf_counter() - start
                    fitted[name] = model

                    row = {"model": name, "n_components": n_components, "N": n, "seed": seed}
                    row.update(evaluate(model, Z_test, y_te))
                    row["train_time_s"] = train_time
                    rows.append(row)
                    print(f"  pca={n_components} N={n:<3} seed={seed} {name:<8} "
                          f"AUROC={row['auroc']:.3f}  time={train_time:.2f}s")

                if "SVM-RBF" in fitted and "QK-Fidelity" in fitted:
                    g = run_geometric_difference(fitted, Z_train)
                    g_rows.append({"n_components": n_components, "N": n, "seed": seed,
                                   "g": g, "sqrt_N": np.sqrt(n)})
                    print(f"  pca={n_components} N={n:<3} seed={seed} geometric difference "
                          f"g={g:.2f} (max possible sqrt(N)={np.sqrt(n):.1f})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    summarize(df).to_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    pd.DataFrame(g_rows).to_csv(os.path.join(RESULTS_DIR, "geometric_difference.csv"), index=False)
    print(f"\nSaved {len(df)} rows to {RESULTS_DIR}/results.csv and summary to {RESULTS_DIR}/summary.csv")

    print_auroc_table(df)


if __name__ == "__main__":
    main()
