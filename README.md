# QuantumScan

Comparing classical and quantum machine learning for **pneumonia detection from chest X-rays when only a few training patients are available**.

Two classical models and five quantum kernel models are trained on exactly the same features, training samples, seeds and test set. Each is evaluated as the number of training patients grows from 25 to 200.

## Data

[PneumoniaMNIST](https://medmnist.com/) (MedMNIST v3): 28×28 grayscale chest X-rays, labelled normal (0) or pneumonia (1).
`data.py` downloads it with `medmnist`. If `medmnist` can't be imported (it needs torch, which Windows Smart App Control blocks), `data.py` reads the same official file, `data/pneumoniamnist.npz` from [Zenodo](https://zenodo.org/records/10519652) (MD5 `28209eda62fecd6e6a2d98b1501bb15f`), directly.

- **Test set:** a fixed, class-balanced set of 300 images (150 per class) taken from the official test split with a fixed seed (42). Every experiment uses this same set.
- **Training set:** for each experiment, a balanced sample of **N = 25, 50, 100 or 200** patients from the official train split, drawn with a seed (0–4).
- **Features:** images are flattened (784 pixels), standardised (`StandardScaler`) and reduced with PCA to **4 or 8 components**, i.e. 4 or 8 qubits. The scaler and PCA are fitted on the training sample only.

## Models

All SVMs are tuned with `GridSearchCV`, stratified 3-fold CV and AUROC scoring, on the training sample only. No grid has more than **20 combinations**, the size of the classical SVM grid, so no quantum model gets a bigger tuning budget.

| Model | Type | Description | Tuned over |
|---|---|---|---|
| `SVM-RBF` | classical | SVM with RBF kernel | C ∈ {0.1, 1, 10, 100} × gamma ∈ {scale, 0.001, 0.01, 0.1, 1} |
| `LogReg` | classical | Logistic regression (L2, C = 1), simple reference | none |
| `QK-Fidelity` | quantum | Fidelity kernel \|⟨ψ(x)\|ψ(x′)⟩\|² of the ZZ feature map (Havlíček et al., 2019) | C ∈ {0.1, 1, 10, 100} × target median kernel ∈ {0.8, 0.6, 0.368, 0.2, 0.1} (20) |
| `QK-Projected` | quantum | Same circuit; each patient is represented by ⟨X⟩, ⟨Y⟩, ⟨Z⟩ of every qubit, then an RBF kernel is applied (projected quantum kernel, Huang et al., 2021) | C ∈ {1, 10} × target median kernel (5) × gamma ∈ {scale, 1} (20) |
| `QK-NoEntangle` | quantum | Same as QK-Fidelity without the ZZ gates; ablation that tests whether entanglement matters | same as QK-Fidelity (20) |
| `QK-Hybrid` | quantum + classical | K = w·K_fidelity + (1 − w)·K_RBF, with the RBF part on the PCA features (gamma = scale). Tests whether the quantum kernel adds anything to the classical one | w ∈ {0.25, 0.5, 0.75} × C ∈ {1, 10} × target median kernel ∈ {0.8, 0.368, 0.1} (18) |
| `QK-Aligned` | quantum | QK-Fidelity with one trainable scaling weight per feature inside the ZZ feature map, fitted by maximising kernel-target alignment (quantum kernel alignment, Hubregtsen et al., 2022) | same as QK-Fidelity (20) |

**Quantum encoding.** The PCA features are MinMax-scaled to [0, π] (fitted on the training sample only) and multiplied by the bandwidth. They are then encoded with the ZZ feature map, repeated twice. Each repetition applies H on every qubit, then RZ(xᵢ) on qubit i, then ZZ((π − xᵢ)(π − xⱼ)) between neighbouring qubits. A smaller bandwidth gives a smoother kernel, which counters the "exponential concentration" of fidelity kernels (Shaydulin & Wild, 2022).

**Bandwidth: calibrated on training data only.** There is no fixed list of bandwidths. Like the median heuristic for RBF kernels, the bandwidth is set from the data. CV tunes a *target median off-diagonal value* of the fidelity kernel on the training data, one of {0.8, 0.6, 0.368, 0.2, 0.1}. In every fit, including each CV fold, `calibrate_bandwidth` solves for the bandwidth that gives this median on that training data. It scans log-spaced bandwidths from 0.001 to 2, takes the first crossing and interpolates. The middle target, e⁻¹ ≈ 0.368, is what the RBF median heuristic gives at the median distance. The range avoids the two degenerate kernels: almost constant (every patient looks the same) and almost the identity (every patient looks unrelated, the "exponential concentration" of fidelity kernels). Each model is calibrated on its own kernel, so QK-NoEntangle gets larger bandwidths than QK-Fidelity for the same target. The bandwidth each tuned model ends up with is stored in the `bandwidth` column of `results.csv`.

An earlier version used a fixed list (0.01–0.25) that was cut down after looking at test AUROC. That was test-set peeking, and this rule replaces it.

*Known issue:* the scan goes up to bandwidth 2, but features are scaled to [0, π]. Above bandwidth 1, angles pass π and the encoding wraps around: at bandwidth 2, the lowest and highest value of a feature give the same state. The median-kernel curve then stops being monotone. For QK-NoEntangle, which needs larger bandwidths, low targets sometimes land in this region. Two runs collapsed: AUROC 0.54 at 4 qubits, N = 50, seed 4 (bandwidth 1.96) and AUROC 0.76 at 8 qubits, N = 200, seed 1 (bandwidth 1.67). Capping the scan at bandwidth 1 would fix this, but the problem was found by inspecting test results, so the results here are reported without that change.

**Kernel alignment (QK-Aligned).** Each feature gets a weight wᵢ, so the circuit angle is bandwidth · wᵢ · xᵢ, also inside the ZZ terms. The bandwidth is calibrated first, with all wᵢ = 1. Starting from all wᵢ = 1, which is QK-Fidelity, L-BFGS-B maximises the kernel-target alignment ⟨K, yyᵀ⟩ / (‖K‖ · ‖yyᵀ‖) on the training sample. It runs at most 10 iterations and 100 loss evaluations, with wᵢ ∈ [0, 5]. Inside cross-validation, the weights are refitted on each training fold. The start point is fixed, so the result is deterministic.

**Entanglement ablations.** `QK-NoEntangle` answers "does entanglement matter?" for QK-Fidelity. `ablation.py` asks the same question for QK-Hybrid and QK-Aligned by rerunning them without ZZ gates (`QK-Hybrid-NoEnt`, `QK-Aligned-NoEnt`) on the same samples and grids. Without ZZ gates each patient is a product state, so the kernel is a product of one-qubit terms that a classical computer evaluates directly.

**Simulation.** The circuits run in a small, batched pure-NumPy statevector simulator in `quantum.py`. Each patient's state is computed once, and every kernel entry is an inner product between stored states, so no circuit is run per pair of patients. The simulator was checked against an independent dense-matrix implementation, where states, kernels and Pauli expectation values agree to about 1e-16. `quantum.py` also contains the same circuit in PennyLane (`default.qubit`, `qml.state()`) and `check_pennylane_matches()`, which asserts that both kernels agree to 1e-6. That check runs in Colab.

**Geometric difference.** For every run, `run_all.py` also stores g(K_C ‖ K_Q) = √‖√K_Q √K_C (K_C + λI)⁻² √K_C √K_Q‖∞. This is the regularised form from Huang et al. (2021), with λ = 0.01. Here K_C is the tuned classical RBF kernel and K_Q is the tuned quantum fidelity kernel on the training sample. If g is close to 1, the classical kernel can learn anything the quantum kernel can, so a quantum advantage is impossible on this data. Only g of order √N leaves room for an advantage.

The ridge term λ matters. The tuned SVM often picks a very small gamma (0.001), which makes K_C almost singular. Without λ, or with a tiny one, g blows up for numerical rather than physical reasons: one run gave g = 23.6 at λ = 10⁻³, above the theoretical maximum √50 ≈ 7.1. `analysis.py` therefore reports the median g over seeds.

## Metrics

For each model, PCA size, N and seed: **AUROC**, accuracy, **sensitivity** (pneumonia correctly detected), **specificity** (normal correctly cleared), F1 and training time. Training time includes the hyperparameter search.

## Files

| File | Purpose |
|---|---|
| `data.py` | Loading, balanced sampling, scaling and PCA |
| `classical.py` | Classical model factories (`CLASSICAL_MODELS`) |
| `quantum.py` | NumPy statevector simulator, quantum kernels, `QUANTUM_MODELS`, geometric difference, PennyLane check |
| `run_all.py` | Experiment loop; writes `results/results.csv`, `summary.csv` and `geometric_difference.csv` |
| `ablation.py` | Entanglement ablation for QK-Hybrid and QK-Aligned; writes `results/ablation.csv` |
| `analysis.py` | Learning-curve plot, summary and stability tables, Holm-corrected Wilcoxon tests and plain-English summary |
| `QuantumScan_colab.ipynb` | Runs everything on Google Colab, including the PennyLane check |
| `requirements.txt` | Dependencies |

## How to run

### Google Colab (recommended)

1. Upload this folder to Google Drive as `MyDrive/QuantumScan`.
2. Open `QuantumScan_colab.ipynb` in Colab and run all cells.

### Locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
.venv/bin/python run_all.py                 # all experiments (~30 min, mostly QK-Aligned)
.venv/bin/python ablation.py                # entanglement ablation for QK-Hybrid / QK-Aligned (~15 min)
.venv/bin/python analysis.py                # plots, tables, tests, summary
.venv/bin/python quantum.py                 # PennyLane vs NumPy check (needs PennyLane)
```

On Windows with Smart App Control enabled, PennyLane, torch and recent pandas wheels are blocked. Use Colab instead.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `results.csv` | One row per model × PCA size × N × seed, with all metrics and the calibrated bandwidth of quantum models |
| `geometric_difference.csv` | g per PCA size × N × seed, with √N for reference |
| `ablation.csv` | Same columns as `results.csv`, for `QK-Hybrid-NoEnt` and `QK-Aligned-NoEnt` |
| `summary.csv` | Numeric mean and std of every metric |
| `summary_table.csv` | Mean ± std of AUROC, sensitivity, specificity, F1 and training time |
| `stability.csv` | Std and range of AUROC over seeds at N = 25 and 50, and the std relative to SVM-RBF |
| `wilcoxon.csv` | Paired Wilcoxon signed-rank tests: each quantum model vs SVM-RBF, and each entangled model vs its no-ZZ ablation. Raw p and Holm-corrected `p_holm` |
| `learning_curves.png` | AUROC vs N, one panel each for 4 and 8 qubits |
| `summary.txt` | Plain-English summary |

With 5 seeds, the smallest possible two-sided Wilcoxon p-value is 0.0625, so per-N tests can never reach p < 0.05. The pooled tests (20 pairs per PCA size) are the ones to report. Holm correction is applied separately to the pooled family and to the per-N family, and the summary calls a result significant only if its Holm-corrected p < 0.05.
