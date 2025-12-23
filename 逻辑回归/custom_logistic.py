import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)  # 避免溢出
    return 1.0 / (1.0 + np.exp(-z))


class CustomLogisticRegression(ClassifierMixin, BaseEstimator):
    """手写二分类逻辑回归，支持 L1/L2 正则与 class_weight。

    说明：
    - 仅支持二分类 {0,1}
    - 为了兼容 sklearn 的 Pipeline / CV，继承 BaseEstimator + ClassifierMixin
    - 正则项不作用于截距项
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        lr: float = 0.05,
        max_iter: int = 2000,
        C: float = 1.0,
        penalty: str = "l2",
        class_weight=None,
        fit_intercept: bool = True,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.C = C
        self.penalty = penalty
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.verbose = verbose

    def _prepare_xy(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        y = y.astype(np.float64)
        self.classes_ = unique_labels(y)
        if set(self.classes_) - {0, 1}:
            raise ValueError("CustomLogisticRegression 仅支持二分类标签 0/1")
        self.n_features_in_ = X.shape[1]
        return X, y

    def _append_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _compute_sample_weight(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight == "balanced":
            pos = (y == 1).sum()
            neg = (y == 0).sum()
            if pos == 0 or neg == 0:
                return np.ones_like(y)
            weight_pos = neg / pos
            weights = np.where(y == 1, weight_pos, 1.0)
            return weights
        if isinstance(self.class_weight, dict):
            return np.vectorize(lambda v: self.class_weight.get(v, 1.0))(y)
        return np.ones_like(y)

    def fit(self, X, y):
        X, y = self._prepare_xy(X, y)
        sample_weight = self._compute_sample_weight(y)

        Xw = X * sample_weight.reshape(-1, 1)
        X_ext = self._append_intercept(Xw)
        y_w = y * sample_weight

        n_samples, n_features_ext = X_ext.shape
        self.coef_ = np.zeros((1, n_features_ext - (1 if self.fit_intercept else 0)))
        self.intercept_ = np.zeros(1) if self.fit_intercept else np.array([])
        w = np.zeros(n_features_ext)

        for i in range(self.max_iter):
            logits = X_ext @ w
            probs = _sigmoid(logits)
            error = probs - y_w
            grad = (X_ext.T @ error) / max(1, sample_weight.sum())

            if self.penalty == "l2":
                reg_mask = np.ones_like(w)
                if self.fit_intercept:
                    reg_mask[0] = 0.0
                grad += (1.0 / self.C) * reg_mask * w
            elif self.penalty == "l1":
                reg_mask = np.ones_like(w)
                if self.fit_intercept:
                    reg_mask[0] = 0.0
                grad += (1.0 / self.C) * reg_mask * np.sign(w)

            w_new = w - self.lr * grad
            if np.linalg.norm(w_new - w) < self.tol:
                w = w_new
                if self.verbose:
                    print(f"converged at iter {i}")
                break
            w = w_new

        if self.fit_intercept:
            self.intercept_ = np.array([w[0]])
            self.coef_ = w[1:].reshape(1, -1)
        else:
            self.intercept_ = np.array([0.0])
            self.coef_ = w.reshape(1, -1)
        return self

    def decision_function(self, X):
        X = check_array(X, accept_sparse=False)
        X_ext = self._append_intercept(X)
        w_full = np.concatenate([self.intercept_, self.coef_.ravel()]) if self.fit_intercept else self.coef_.ravel()
        return X_ext @ w_full

    def predict_proba(self, X):
        logits = self.decision_function(X)
        prob_pos = _sigmoid(logits).reshape(-1, 1)
        return np.hstack([1 - prob_pos, prob_pos])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
