"""
ablation.py - Entanglement ablation for QK-Hybrid and QK-Aligned.

QK-NoEntangle already answers "does entanglement matter?" for QK-Fidelity.
This script asks the same question for the two newer models: it reruns
QK-Hybrid and QK-Aligned with the ZZ (entangling) gates removed, on exactly
the same PCA features, training samples, seeds, test set and tuning grids
as run_all.py.

Output:
    results/ablation.csv - same columns as results.csv, models
                           QK-Hybrid-NoEnt and QK-Aligned-NoEnt

Usage:
    python ablation.py      (after run_all.py; analysis.py reads the output)
"""

import os
import time

import pandas as pd

from data import load_data, make_experiment
from quantum import make_qk_aligned, make_qk_hybrid
from run_all import N_COMPONENTS_LIST, N_LIST, RESULTS_DIR, SEEDS, evaluate


def without_entanglement(factory):
    """Same tuned model (same grid and CV), but with product-state encoding."""
    def make(seed):
        search = factory(seed)
        search.estimator.set_params(entangle=False)
        return search
    return make


ABLATION_MODELS = {
    "QK-Hybrid-NoEnt": without_entanglement(make_qk_hybrid),
    "QK-Aligned-NoEnt": without_entanglement(make_qk_aligned),
}


def main():
    X_train_full, y_train_full, X_test, y_test = load_data()
    rows = []
    for n_components in N_COMPONENTS_LIST:
        for n in N_LIST:
            for seed in SEEDS:
                Z_train, y_train, Z_test, y_te = make_experiment(
                    X_train_full, y_train_full, X_test, y_test, n, seed, n_components
                )
                for name, factory in ABLATION_MODELS.items():
                    model = factory(seed)
                    start = time.perf_counter()
                    model.fit(Z_train, y_train)
                    train_time = time.perf_counter() - start

                    row = {"model": name, "n_components": n_components, "N": n, "seed": seed}
                    row.update(evaluate(model, Z_test, y_te))
                    row["train_time_s"] = train_time
                    rows.append(row)
                    print(f"  pca={n_components} N={n:<3} seed={seed} {name:<16} "
                          f"AUROC={row['auroc']:.3f}  time={train_time:.2f}s", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "ablation.csv"), index=False)
    print(f"\nSaved {len(rows)} rows to {RESULTS_DIR}/ablation.csv")


if __name__ == "__main__":
    main()
