"""
quantum.py - Quantum kernel models for QuantumScan.

All quantum circuits are simulated with a small pure-NumPy statevector
simulator (4-8 qubits = 16-256 amplitudes, which is tiny). A PennyLane version
of the same circuit is included at the bottom to check the simulator; it only
runs where PennyLane is installed (e.g. Google Colab).

Pipeline for every quantum model:
    PCA features --MinMax to [0, pi]--> x --* bandwidth--> circuit angles
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
"""

import numpy as np
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
# 3. scikit-learn compatible classifier
# ===========================================================================

class QuantumKernelSVC(ClassifierMixin, BaseEstimator):
    """SVC with a precomputed quantum kernel.

    Parameters
    ----------
    kernel    : "fidelity", "no_entangle" or "projected"
    C         : SVM regularisation strength
    bandwidth : scales the [0, pi] features before encoding. Small bandwidth
                = small rotation angles = smoother, less "spiky" kernel
                (Shaydulin & Wild 2022; Canatar et al. 2022)
    gamma     : RBF width for the projected kernel ("scale" = 1 / (n_features * var),
                same rule as scikit-learn's SVC)
    """

    def __init__(self, kernel="fidelity", C=1.0, bandwidth=1.0, gamma="scale", n_reps=N_REPS):
        self.kernel = kernel
        self.C = C
        self.bandwidth = bandwidth
        self.gamma = gamma
        self.n_reps = n_reps

    def _encode(self, X):
        """PCA features -> circuit angles -> one quantum state per patient."""
        angles = self.bandwidth * self.scaler_.transform(X)
        return zz_feature_map_states(angles, self.n_reps, entangle=self.kernel != "no_entangle")

    def _represent(self, X):
        """What we store per patient: the state, or its Pauli expectations."""
        states = self._encode(X)
        return pauli_expectations(states) if self.kernel == "projected" else states

    def _kernel(self, A, B):
        if self.kernel == "projected":
            return rbf_kernel(A, B, gamma=self.gamma_)
        return fidelity_kernel(A, B)

    def fit(self, X, y):
        # MinMax scaling to [0, pi] is fitted on the training data only.
        self.scaler_ = MinMaxScaler(feature_range=(0, np.pi)).fit(X)
        self.train_repr_ = self._represent(X)
        if self.kernel == "projected":
            self.gamma_ = (1.0 / (self.train_repr_.shape[1] * self.train_repr_.var())
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
# 4. Model factories with hyperparameter tuning
# ===========================================================================
# Each grid has 20 combinations, the same as the classical SVM grid
# (4 C x 5 gamma), so every SVM gets the same tuning budget.

BANDWIDTHS = [0.05, 0.1, 0.25, 0.5, 1.0]

FIDELITY_GRID = {"C": [0.1, 1, 10, 100], "bandwidth": BANDWIDTHS}                    # 4 x 5 = 20
PROJECTED_GRID = {"C": [1, 10], "bandwidth": BANDWIDTHS, "gamma": ["scale", 1.0]}   # 2 x 5 x 2 = 20


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


# Registry of quantum models: display name -> factory function.
QUANTUM_MODELS = {
    "QK-Fidelity": make_qk_fidelity,
    "QK-Projected": make_qk_projected,
    "QK-NoEntangle": make_qk_no_entangle,
}


# ===========================================================================
# 5. PennyLane reference implementation (only used if PennyLane is installed)
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
