"""
confirm.py - Confirmation run on NEW seeds 5-9.

Pre-registered before running (decided before any seed 5-9 result was seen):
    - Model code and settings are frozen at commit
      7a802814d68bf5ddc545e0e8991dfbbc4e6f8361. This script changes nothing
      in them: it reuses run_all.main() and ablation.main() and only swaps
      in the new seeds and a separate output folder.
    - PRIMARY TEST (one test, no multiple-testing correction):
      QK-NoEntangle vs SVM-RBF at 8 qubits, two-sided Wilcoxon signed-rank
      test on test AUROC, paired by (N, seed), pooled over N = 25, 50, 100,
      200, on seeds 5-9 only (20 pairs). Confirmed if p < 0.05 and the
      mean difference is positive.
    - Secondary: the same analysis as before over all 10 seeds (0-9),
      with Holm correction, and the stability table with 10 seeds.

All 7 models (and the entanglement ablation) are run so the tables stay
complete. Seeds 0-4 are not rerun; their results stay in results/.

Output:
    results/confirm/results.csv, summary.csv, geometric_difference.csv, ablation.csv

Usage:
    python confirm.py
"""

import ctypes
import sys

import ablation
import run_all

CONFIRM_SEEDS = [5, 6, 7, 8, 9]
CONFIRM_DIR = "results/confirm"


def keep_awake():
    """Stop Windows from sleeping while this process runs (no-op elsewhere).
    A sleep would not change any result, but it inflates the measured train times."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    keep_awake()
    run_all.SEEDS, run_all.RESULTS_DIR = CONFIRM_SEEDS, CONFIRM_DIR
    ablation.SEEDS, ablation.RESULTS_DIR = CONFIRM_SEEDS, CONFIRM_DIR
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "models"):
        run_all.main()
    if which in ("all", "ablation"):
        ablation.main()


if __name__ == "__main__":
    main()
