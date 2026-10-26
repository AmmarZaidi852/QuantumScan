"""
quantum.py - Quantum kernel models for QuantumScan.

All quantum circuits are simulated with a small pure-NumPy statevector
simulator (4-8 qubits = 16-256 amplitudes, which is tiny). A PennyLane version
of the same circuit is included at the bottom to check the simulator; it only
runs where PennyLane is installed (e.g. Google Colab).

Pipeline for every quantum model:
    PCA features --MinMax to [0, pi]--> x --* bandwidth--> circuit angles
    (the bandwidth is calibrated on the training data, see calibrate_bandwidth)
    --ZZ feature map--> one quantum state per patient --> kernel matrix
    --> SVC(kernel="precomputed")

Each patient's state is computed ONCE per fit/predict, and all kernel entries
are built from those stored states (an inner product between state vectors).
We never simulate one circuit per pair of patients.

Models (registered in QUANTUM_MODELS, used by run_all.py):
    QK-Fidelity   : fidelity kernel k(x, x') = |<psi(x)|psi(x')>|^2
                    with the ZZ feature map (Havlicek et al., Nature 2019).
    QK-Projected  : same circuit, each patient represented by the <X>, <Y>, <Z>
                    expectation value of every qubit, then an RBF kernel
                    on those (projected quantum kernel, Huang et al. 2021).
    QK-NoEntangle : same as QK-Fidelity but without the ZZ (entangling) gates.
                    Ablation: does entanglement matter?
    QK-Hybrid     : K = w * K_fidelity + (1 - w) * K_rbf, with the weight w tuned
                    by CV. A sum of two kernels is still a valid kernel, so this
                    tests whether the quantum kernel adds anything on top of the
                    classical RBF kernel.
    QK-Aligned    : QK-Fidelity with one trainable scaling weight per feature
                    inside the ZZ feature map, fitted by maximising
                    kernel-target alignment on the training sample
                    (quantum kernel alignment, Hubregtsen et al. 2022).
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

N_REPS = 2  # repetitions of the feature-map layer

# ===========================================================================
# 1. Minimal statevector simulator (batched: many patients at once)
# ===========================================================================
# A state of n qubits for a batch of B patients is stored as a tensor of shape
# (B, 2, 2, ..., 2): axis q+1 is qubit q. Flattening it to (B, 2**n) gives
# the usual state vector with qubit 0 as the most significant bit (same
# convention as PennyLane).

H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def zero_state(n_samples, n_qubits):
    """|00...0> for every patient in the batch."""
    state = np.zeros((n_samples,) + (2,) * n_qubits, dtype=complex)
    state[(slice(None),) + (0,) * n_qubits] = 1.0
    return state


def z_eigenvalues(n_qubits, q):
    """Eigenvalues of Pauli Z on qubit q (+1 for |0>, -1 for |1>),
    shaped so they broadcast against the (B, 2, ..., 2) state tensor."""
    shape = [1] * (n_qubits + 1)
    shape[q + 1] = 2
    return np.array([1.0, -1.0]).reshape(shape)


def per_patient(angles, n_qubits):
    """Reshape a (B,) vector of angles so it broadcasts against the state."""
    return angles.reshape((-1,) + (1,) * n_qubits)


def apply_single_qubit_gate(state, gate, q):
    """Apply a 2x2 gate (same for all patients) to qubit q."""
    state = np.tensordot(gate, state, axes=([1], [q + 1]))  # gate acts on axis q+1
    return np.moveaxis(state, 0, q + 1)                      # put the qubit axis back


def apply_rz(state, theta, q):
    """RZ(theta) = exp(-i theta Z / 2) on qubit q, one angle per patient.
    RZ is diagonal, so it just multiplies each amplitude by a phase."""
    n = state.ndim - 1
    return state * np.exp(-0.5j * per_patient(theta, n) * z_eigenvalues(n, q))


def apply_zz(state, phi, q1, q2):
    """ZZ(phi) = exp(-i phi Z_q1 Z_q2 / 2), one angle per patient.
    Equivalent to CNOT - RZ(phi) - CNOT, and to PennyLane's IsingZZ(phi)."""
    n = state.ndim - 1
    zz = z_eigenvalues(n, q1) * z_eigenvalues(n, q2)
    return state * np.exp(-0.5j * per_patient(phi, n) * zz)


def zz_feature_map_states(X, n_reps=N_REPS, entangle=True):
    """Simulate the ZZ feature map for every row of X (one angle per qubit).

    One repetition = H on all qubits, RZ(x_i) on qubit i, then
    ZZ((pi - x_i)(pi - x_j)) between neighbouring qubits (i, i+1).
    With entangle=False the ZZ gates are skipped (product-state ablation).

    Returns the state vectors, shape (n_samples, 2**n_qubits).
    """
    n_samples, n_qubits = X.shape
    state = zero_state(n_samples, n_qubits)
    for _ in range(n_reps):
        for q in range(n_qubits):
            state = apply_single_qubit_gate(state, H_GATE, q)
        for q in range(n_qubits):
            state = apply_rz(state, X[:, q], q)
        if entangle:
            for q in range(n_qubits - 1):
                phi = (np.pi - X[:, q]) * (np.pi - X[:, q + 1])
                state = apply_zz(state, phi, q, q + 1)
    return state.reshape(n_samples, -1)


# ===========================================================================
# 2. Kernels built from the stored states
# ===========================================================================

def fidelity_kernel(states_a, states_b):
    """K[i, j] = |<psi_a_i | psi_b_j>|^2, all pairs with one matrix product."""
    return np.abs(states_a.conj() @ states_b.T) ** 2


def kernel_target_alignment(K, y):
    """A(K, yy^T) = <K, yy^T>_F / (||K||_F ||yy^T||_F) with labels in {-1, +1}
    (Cristianini et al. 2001). Close to 1 when same-class patients have high
    kernel values and different-class patients low ones."""
    signs = 2 * np.asarray(y) - 1
    return float(signs @ K @ signs / (np.linalg.norm(K) * len(signs)))


def pauli_expectations(states):
    """<X>, <Y>, <Z> of every qubit -> features of shape (n_samples, 3 * n_qubits).

    For each qubit we compute its 1-qubit reduced density matrix rho
    (trace out all other qubits), then read off
    <X> = 2 Re rho_01,  <Y> = -2 Im rho_01,  <Z> = rho_00 - rho_11.
    """
    n_samples = len(states)
    n_qubits = int(np.log2(states.shape[1]))
    psi = states.reshape((n_samples,) + (2,) * n_qubits)
    features = []
    for q in range(n_qubits):
        p = np.moveaxis(psi, q + 1, 1).reshape(n_samples, 2, -1)  # (B, qubit q, rest)
        rho = np.einsum("bir,bjr->bij", p, p.conj())
        features += [2 * rho[:, 0, 1].real, -2 * rho[:, 0, 1].imag, (rho[:, 0, 0] - rho[:, 1, 1]).real]
    return np.stack(features, axis=1)


def matrix_sqrt(K):
    """Square root of a symmetric positive semi-definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(K)
    return eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0, None))) @ eigvecs.T


def geometric_difference(K_classical, K_quantum, reg=1e-2):
    """Geometric difference g(K_C || K_Q) from Huang et al., Nat. Commun. 2021
    (regularised form from their Methods section):

        g = sqrt( || sqrt(K_Q) sqrt(K_C) (K_C + reg*I)^-2 sqrt(K_C) sqrt(K_Q) ||_inf )

    With reg = 0 this is the original sqrt(|| sqrt(K_Q) K_C^-1 sqrt(K_Q) ||).
    The ridge term `reg` matters because tuned RBF kernels with a small gamma
    are nearly singular, and inverting them without it makes g explode.

    Both kernels are first normalised to trace N. If g is small (close to 1),
    the classical kernel can fit any function the quantum kernel can, so a
    quantum advantage on this data is impossible. Only g of the order of
    sqrt(N) leaves room for an advantage.
    """
    n = len(K_classical)
    K_c = K_classical * n / np.trace(K_classical)
    K_q = K_quantum * n / np.trace(K_quantum)

    sqrt_K_c, sqrt_K_q = matrix_sqrt(K_c), matrix_sqrt(K_q)
    inv = np.linalg.inv(K_c + reg * np.eye(n))
    M = sqrt_K_q @ sqrt_K_c @ inv @ inv @ sqrt_K_c @ sqrt_K_q
    return float(np.sqrt(np.linalg.norm(M, ord=2)))  # spectral norm


# ===========================================================================
# 3. Bandwidth calibration (training data only)
# ===========================================================================
# Like the median heuristic for RBF kernels (gamma = 1 / median squared
# distance), the bandwidth is not taken from a fixed list. It is solved for so
# that the MEDIAN OFF-DIAGONAL value of the fidelity kernel on the training
# data equals a target. The target is the hyperparameter that CV tunes.
# Because the rule only looks at the training kernel, the same target gives
# different bandwidths for 4 and 8 qubits and for different training samples.

BANDWIDTH_SCAN = np.geomspace(1e-3, 2.0, 41)  # bandwidths tried when calibrating
_CURVE_CACHE = {}  # (training data, entangle, n_reps) -> median kernel value per scanned bandwidth


def median_offdiagonal(K):
    return float(np.median(K[np.triu_indices(len(K), k=1)]))


def calibrate_bandwidth(X_scaled, target, entangle=True, n_reps=N_REPS):
    """Bandwidth at which the median off-diagonal fidelity kernel value on
    X_scaled (training data, already scaled to [0, pi]) equals `target`.

    The median kernel value falls from 1 (bandwidth -> 0, all patients look
    the same) as the bandwidth grows. We scan BANDWIDTH_SCAN, take the first
    crossing of `target` and interpolate (log-linearly) between the two
    scanned points around it. If the target is never reached (the kernel
    cannot get that small), the scanned bandwidth whose median is closest to
    the target is used. The scan does not depend on the target, so it is
    cached per training set.
    """
    key = (X_scaled.tobytes(), X_scaled.shape, entangle, n_reps)
    if key not in _CURVE_CACHE:
        medians = []
        for bw in BANDWIDTH_SCAN:
            states = zz_feature_map_states(bw * X_scaled, n_reps, entangle=entangle)
            medians.append(median_offdiagonal(fidelity_kernel(states, states)))
        if len(_CURVE_CACHE) > 256:
            _CURVE_CACHE.clear()
        _CURVE_CACHE[key] = np.array(medians)
    medians = _CURVE_CACHE[key]

    below = np.where(medians <= target)[0]
    if len(below) == 0:
        return float(BANDWIDTH_SCAN[np.argmin(np.abs(medians - target))])
    i = below[0]
    if i == 0:
        return float(BANDWIDTH_SCAN[0])
    m_hi, m_lo = medians[i - 1], medians[i]  # m_hi > target >= m_lo
    t = (m_hi - target) / (m_hi - m_lo)
    log_lo, log_hi = np.log(BANDWIDTH_SCAN[i - 1]), np.log(BANDWIDTH_SCAN[i])
    return float(np.exp(log_lo + t * (log_hi - log_lo)))


# ===========================================================================
# 4. scikit-learn compatible classifier
# ===========================================================================

_ALIGN_CACHE = {}  # (training data, bandwidth, ...) -> (alignment before/after, weights)


class QuantumKernelSVC(ClassifierMixin, BaseEstimator):
    """SVC with a precomputed quantum kernel.

    Parameters
    ----------
    kernel    : "fidelity", "no_entangle", "projected", "hybrid" or "aligned"
    C         : SVM regularisation strength
    kernel_median : target median off-diagonal value of the fidelity kernel
                on the training data. In fit, the bandwidth (which scales the
                [0, pi] features before encoding) is calibrated to hit it.
                High target = small bandwidth = smooth kernel; low target =
                large bandwidth = "spiky" kernel (Shaydulin & Wild 2022;
                Canatar et al. 2022)
    bandwidth : fixed bandwidth instead of calibration (None = calibrate).
                The bandwidth actually used is stored in bandwidth_
    gamma     : RBF width for the projected kernel (on the Pauli expectations)
                and for the RBF part of the hybrid kernel (on the PCA features).
                "scale" = 1 / (n_features * var), same rule as scikit-learn's SVC
    weight    : hybrid kernel only, weight w of the quantum kernel in
                w * K_fidelity + (1 - w) * K_rbf
    align_iters : aligned kernel only, maximum L-BFGS-B iterations when
                training the per-feature scaling weights (at most 10 loss
                evaluations per iteration are allowed)
    entangle  : False drops the ZZ gates (product states). kernel="no_entangle"
                is the same as kernel="fidelity" with entangle=False; this
                switch gives the same ablation for the hybrid and aligned kernels.
    """

    def __init__(self, kernel="fidelity", C=1.0, kernel_median=float(np.exp(-1)), bandwidth=None,
                 gamma="scale", weight=0.5, align_iters=10, entangle=True, n_reps=N_REPS):
        self.kernel = kernel
        self.C = C
        self.kernel_median = kernel_median
        self.bandwidth = bandwidth
        self.gamma = gamma
        self.weight = weight
        self.align_iters = align_iters
        self.entangle = entangle
        self.n_reps = n_reps

    def _entangled(self):
        return self.entangle and self.kernel != "no_entangle"

    def _states(self, X_scaled, feature_weights):
        angles = self.bandwidth_ * feature_weights * X_scaled
        return zz_feature_map_states(angles, self.n_reps, entangle=self._entangled())

    def _encode(self, X):
        """PCA features -> circuit angles -> one quantum state per patient."""
        return self._states(self.scaler_.transform(X), self.feature_weights_)

    def _align(self, X, y):
        """Per-feature scaling weights that maximise kernel-target alignment
        of the fidelity kernel on the training sample.

        Starts from all weights = 1 (= QK-Fidelity) and runs a few L-BFGS-B
        iterations with finite-difference gradients. The start point is fixed,
        so the result is deterministic. Weights are kept in [0, 5]; a weight
        of 0 switches that feature off.

        The weights do not depend on C, so they are cached per (training
        data, bandwidth): GridSearchCV then aligns once per fold and bandwidth
        instead of once per grid point. The result is the same either way.
        """
        X_scaled = self.scaler_.transform(X)
        key = (X_scaled.tobytes(), np.asarray(y).tobytes(), self.bandwidth_, self.n_reps,
               self.align_iters, self.entangle)
        if key in _ALIGN_CACHE:
            self.alignment_, weights = _ALIGN_CACHE[key]
            return weights

        def loss(w):
            states = self._states(X_scaled, w)
            return -kernel_target_alignment(fidelity_kernel(states, states), y)

        w0 = np.ones(X.shape[1])
        result = minimize(loss, w0, method="L-BFGS-B", bounds=[(0.0, 5.0)] * len(w0),
                          options={"maxiter": self.align_iters, "maxfun": 10 * self.align_iters})
        self.alignment_ = (-loss(w0), -result.fun)  # before, after
        if len(_ALIGN_CACHE) > 256:
            _ALIGN_CACHE.clear()
        _ALIGN_CACHE[key] = (self.alignment_, result.x)
        return result.x

    def _represent(self, X):
        """What we store per patient: the state, its Pauli expectations,
        or (hybrid) the state together with the PCA features."""
        states = self._encode(X)
        if self.kernel == "projected":
            return pauli_expectations(states)
        if self.kernel == "hybrid":
            return states, X
        return states

    def _kernel(self, A, B):
        if self.kernel == "projected":
            return rbf_kernel(A, B, gamma=self.gamma_)
        if self.kernel == "hybrid":
            (states_a, X_a), (states_b, X_b) = A, B
            return (self.weight * fidelity_kernel(states_a, states_b)
                    + (1 - self.weight) * rbf_kernel(X_a, X_b, gamma=self.gamma_))
        return fidelity_kernel(A, B)

    def fit(self, X, y):
        # MinMax scaling to [0, pi] is fitted on the training data only.
        self.scaler_ = MinMaxScaler(feature_range=(0, np.pi)).fit(X)
        # Bandwidth from the training kernel only (for kernel="aligned" this
        # is the starting bandwidth, before the per-feature weights are trained).
        self.bandwidth_ = (self.bandwidth if self.bandwidth is not None else calibrate_bandwidth(
            self.scaler_.transform(X), self.kernel_median, self._entangled(), self.n_reps))
        self.feature_weights_ = np.ones(X.shape[1])
        if self.kernel == "aligned":
            self.feature_weights_ = self._align(X, y)
        self.train_repr_ = self._represent(X)
        if self.kernel in ("projected", "hybrid"):
            rbf_input = self.train_repr_ if self.kernel == "projected" else X
            self.gamma_ = (1.0 / (rbf_input.shape[1] * rbf_input.var())
                           if self.gamma == "scale" else self.gamma)
        self.K_train_ = self._kernel(self.train_repr_, self.train_repr_)
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(self.K_train_, y)
        self.classes_ = self.svc_.classes_
        return self

    def decision_function(self, X):
        K = self._kernel(self._represent(X), self.train_repr_)  # (n_test, n_train)
        return self.svc_.decision_function(K)

    def predict(self, X):
        K = self._kernel(self._represent(X), self.train_repr_)
        return self.svc_.predict(K)


# ===========================================================================
# 5. Model factories with hyperparameter tuning
# ===========================================================================
# Each grid has at most 20 combinations, the same as the classical SVM grid
# (4 C x 5 gamma), so no quantum model gets a bigger tuning budget.

# Target median off-diagonal training-kernel values. The middle one, e^-1 = 0.368,
# is what the median heuristic gives an RBF kernel at the median distance. The
# range stays away from the two degenerate kernels: almost constant (median
# >= 0.9, every patient looks the same) and almost the identity (median near 0,
# every patient looks unrelated to every other: "exponential concentration").
KERNEL_MEDIANS = [0.8, 0.6, 0.368, 0.2, 0.1]

FIDELITY_GRID = {"C": [0.1, 1, 10, 100], "kernel_median": KERNEL_MEDIANS}                    # 4 x 5 = 20
PROJECTED_GRID = {"C": [1, 10], "kernel_median": KERNEL_MEDIANS, "gamma": ["scale", 1.0]}   # 2 x 5 x 2 = 20
# Hybrid: 3 weights x 2 C x 3 targets (highest, middle, lowest) = 18 <= 20.
# The RBF part uses gamma="scale", scikit-learn's default for SVC.
HYBRID_GRID = {"C": [1, 10], "kernel_median": KERNEL_MEDIANS[::2], "weight": [0.25, 0.5, 0.75]}


def _tuned(kernel, grid, seed):
    """GridSearchCV with the same stratified 3-fold CV and AUROC scoring
    as the classical SVM, on the training sample only."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return GridSearchCV(QuantumKernelSVC(kernel=kernel), grid, cv=cv,
                        scoring="roc_auc", refit=True, n_jobs=1)


def make_qk_fidelity(seed):
    return _tuned("fidelity", FIDELITY_GRID, seed)


def make_qk_projected(seed):
    return _tuned("projected", PROJECTED_GRID, seed)


def make_qk_no_entangle(seed):
    return _tuned("no_entangle", FIDELITY_GRID, seed)


def make_qk_hybrid(seed):
    return _tuned("hybrid", HYBRID_GRID, seed)


def make_qk_aligned(seed):
    return _tuned("aligned", FIDELITY_GRID, seed)


# Registry of quantum models: display name -> factory function.
QUANTUM_MODELS = {
    "QK-Fidelity": make_qk_fidelity,
    "QK-Projected": make_qk_projected,
    "QK-NoEntangle": make_qk_no_entangle,
    "QK-Hybrid": make_qk_hybrid,
    "QK-Aligned": make_qk_aligned,
}


# ===========================================================================
# 6. PennyLane reference implementation (only used if PennyLane is installed)
# ===========================================================================

def pennylane_states(X, n_reps=N_REPS, entangle=True):
    """The same ZZ feature map written in PennyLane, simulated with default.qubit."""
    import pennylane as qml

    n_qubits = X.shape[1]
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        for _ in range(n_reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits):
                qml.RZ(x[i], wires=i)
            if entangle:
                for i in range(n_qubits - 1):
                    qml.IsingZZ((np.pi - x[i]) * (np.pi - x[i + 1]), wires=[i, i + 1])
        return qml.state()

    return np.array([circuit(x) for x in X])


def check_pennylane_matches(n_qubits_list=(4, 8), n_samples=20, atol=1e-6, seed=0):
    """Check that the PennyLane and NumPy fidelity kernels agree to `atol`.

    Returns True if all checks pass, None if PennyLane is not available.
    """
    try:
        import pennylane  # noqa: F401
    except ImportError as err:
        print(f"PennyLane not available ({err}); skipping the check.")
        return None

    rng = np.random.default_rng(seed)
    for n_qubits in n_qubits_list:
        X = rng.uniform(0, np.pi, size=(n_samples, n_qubits))
        S_numpy = zz_feature_map_states(X)
        S_penny = pennylane_states(X)
        K_numpy = fidelity_kernel(S_numpy, S_numpy)
        K_penny = fidelity_kernel(S_penny, S_penny)
        max_diff = np.max(np.abs(K_numpy - K_penny))
        print(f"{n_qubits} qubits: max |K_numpy - K_pennylane| = {max_diff:.2e}")
        assert max_diff < atol, f"Kernels differ by {max_diff:.2e} (> {atol})"
    print("PennyLane and NumPy kernels match.")
    return True


if __name__ == "__main__":
    check_pennylane_matches()
