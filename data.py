"""
data.py - Loading and preprocessing PneumoniaMNIST for QuantumScan.

Pipeline (identical for every model, classical or quantum):
    1. Load the official PneumoniaMNIST splits (28x28 grayscale chest X-rays).
       Label 0 = normal, label 1 = pneumonia.
    2. Build a FIXED, class-balanced test set of 300 images from the official
       test split (150 normal + 150 pneumonia). Same images for every run.
    3. For each experiment, sample a small balanced training set of N patients
       from the official train split, using a random seed.
    4. Flatten images (28*28 = 784 features), StandardScaler, then PCA down to
       n_components (4 or 8). The scaler and PCA are fitted on the small
       TRAINING sample only, then applied to the test set, so no information
       from the test set leaks into training.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"          # where medmnist downloads the .npz file
TEST_SIZE = 300            # total size of the fixed balanced test set
TEST_SEED = 42             # seed for choosing the test images (never changes)


def load_split(split, root=DATA_DIR):
    """Load one official split ('train', 'val' or 'test') of PneumoniaMNIST.

    Returns
    -------
    X : array of shape (n_images, 784), pixel values in [0, 1]
    y : array of shape (n_images,), 0 = normal, 1 = pneumonia
    """
    import os
    from medmnist import PneumoniaMNIST  # imported here so the rest of the module works without it

    os.makedirs(root, exist_ok=True)
    ds = PneumoniaMNIST(split=split, download=True, root=root, size=28)
    X = ds.imgs.reshape(len(ds.imgs), -1).astype(np.float64) / 255.0  # flatten 28x28 -> 784
    y = ds.labels.ravel().astype(int)
    return X, y


def balanced_indices(y, n_total, rng):
    """Pick n_total indices with (as close as possible) equal numbers per class.

    For odd n_total the extra example goes to class 1 (pneumonia).
    Sampling is without replacement.
    """
    n_class0 = n_total // 2
    n_class1 = n_total - n_class0
    idx0 = rng.choice(np.where(y == 0)[0], size=n_class0, replace=False)
    idx1 = rng.choice(np.where(y == 1)[0], size=n_class1, replace=False)
    idx = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    return idx


def get_test_set(X_test_full, y_test_full, size=TEST_SIZE, seed=TEST_SEED):
    """Fixed balanced test set drawn from the official test split.

    Uses its own fixed seed so the test set is the same for every
    experiment, model, N and training seed.
    """
    rng = np.random.default_rng(seed)
    idx = balanced_indices(y_test_full, size, rng)
    return X_test_full[idx], y_test_full[idx]


def sample_train_set(X_train_full, y_train_full, n, seed):
    """Sample a small balanced training set of n patients using `seed`.

    The same (n, seed) pair always gives the same patients, so classical
    and quantum models can be trained on exactly the same data.
    """
    rng = np.random.default_rng(seed)
    idx = balanced_indices(y_train_full, n, rng)
    return X_train_full[idx], y_train_full[idx]


def preprocess(X_train, X_test, n_components):
    """StandardScaler + PCA, fitted on the training sample only.

    Returns the reduced training and test features, each with
    `n_components` columns.
    """
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=n_components, random_state=0).fit(scaler.transform(X_train))

    Z_train = pca.transform(scaler.transform(X_train))
    Z_test = pca.transform(scaler.transform(X_test))
    return Z_train, Z_test


def load_data(root=DATA_DIR):
    """Load the full train split and the fixed balanced test set once.

    Returns X_train_full, y_train_full, X_test, y_test (flattened, unscaled).
    """
    X_train_full, y_train_full = load_split("train", root)
    X_test_full, y_test_full = load_split("test", root)
    X_test, y_test = get_test_set(X_test_full, y_test_full)
    return X_train_full, y_train_full, X_test, y_test


def make_experiment(X_train_full, y_train_full, X_test, y_test, n, seed, n_components):
    """Everything one experiment needs: the N-patient sample and PCA features.

    Returns Z_train, y_train, Z_test, y_test. Every model (classical or
    quantum) should be given these exact arrays for a fair comparison.
    """
    X_tr, y_tr = sample_train_set(X_train_full, y_train_full, n, seed)
    Z_train, Z_test = preprocess(X_tr, X_test, n_components)
    return Z_train, y_tr, Z_test, y_test
