"""
classical.py - Classical baseline models for QuantumScan.

Every model here is created by a small "factory" function with the signature

    make_<model>(seed) -> unfitted scikit-learn style estimator

The returned object must provide .fit(X, y), .predict(X) and either
.predict_proba(X) or .decision_function(X). run_all.py only relies on this
interface, so quantum models just need to follow the same pattern.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

# Hyperparameter grid for the RBF SVM.
# C     = how strongly misclassified training points are penalised
# gamma = width of the RBF kernel (large gamma = very local, wiggly boundary)
SVM_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.001, 0.01, 0.1, 1],
}


def make_svm_rbf(seed):
    """RBF-kernel SVM tuned by GridSearchCV.

    The grid search uses stratified 3-fold cross-validation on the small
    training sample only (the test set is never seen during tuning).
    Models are compared by AUROC during tuning. After the search, the best
    C/gamma combination is refitted on the whole training sample.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return GridSearchCV(
        estimator=SVC(kernel="rbf"),
        param_grid=SVM_PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        refit=True,
        n_jobs=1,
    )


def make_logreg(seed):
    """Plain logistic regression (default L2 regularisation, C=1).

    No tuning: a simple, well-understood reference point.
    """
    return LogisticRegression(max_iter=1000, random_state=seed)


# Registry of classical models: display name -> factory function.
CLASSICAL_MODELS = {
    "SVM-RBF": make_svm_rbf,
    "LogReg": make_logreg,
}
