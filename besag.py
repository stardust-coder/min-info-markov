import numpy as np


class LogisticRegression:
    """
    Intercept-free logistic regression for objectives of the form

        maximize  sum_i [ y_i log(sigmoid(x_i^T w)) + (1-y_i) log(1-sigmoid(x_i^T w)) ]

    This version is written to be numerically stable and more reliable than the
    previous plain gradient-ascent implementation.

    Features
    --------
    - no intercept term
    - optional L2 regularization
    - backtracking line search
    - convergence check by gradient norm and loss improvement
    - warm-start support via fit_add()

    Notes
    -----
    For Besag PMLE in your use case, you probably want:
        fit_intercept = False
        l2 = 0.0
    and y = np.ones(m).
    """

    def __init__(
        self,
        eta: float = 1.0,
        n_iter: int = 1000,
        tol: float = 1e-8,
        grad_tol: float = 1e-6,
        l2: float = 0.0,
        fit_intercept: bool = False,
        line_search: bool = True,
        verbose: bool = False,
    ):
        self.eta = float(eta)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.grad_tol = float(grad_tol)
        self.l2 = float(l2)
        self.fit_intercept = bool(fit_intercept)
        self.line_search = bool(line_search)
        self.verbose = bool(verbose)

        self.w = None
        self.loss_history_ = []
        self.n_iter_run_ = 0
        self.converged_ = False

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def fit(self, X, y, verbose=None):
        """
        Fit from scratch.
        """
        X, y = self._validate_inputs(X, y)
        X_aug = self._augment_X(X)

        self.w = np.zeros(X_aug.shape[1], dtype=float)
        self.loss_history_ = []
        self.n_iter_run_ = 0
        self.converged_ = False

        self._optimize(X_aug, y, verbose=self.verbose if verbose is None else verbose)
        return self

    def fit_add(self, X, y, verbose=None):
        """
        Continue optimization from current weights.
        If self.w is None, starts from zero.
        """
        X, y = self._validate_inputs(X, y)
        X_aug = self._augment_X(X)

        if self.w is None:
            self.w = np.zeros(X_aug.shape[1], dtype=float)
            self.loss_history_ = []
            self.n_iter_run_ = 0
            self.converged_ = False
        elif self.w.shape[0] != X_aug.shape[1]:
            raise ValueError(
                f"Weight dimension mismatch: w has length {self.w.shape[0]}, "
                f"but X requires length {X_aug.shape[1]}."
            )

        self._optimize(X_aug, y, verbose=self.verbose if verbose is None else verbose)
        return self

    def predict_proba(self, X):
        """
        Return P(y=1|x).
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        X_aug = self._augment_X(X)
        z = X_aug @ self.w
        p1 = self._sigmoid(z)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        """
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

    def score(self, X, y):
        """
        Classification accuracy.
        """
        y = np.asarray(y).reshape(-1)
        pred = self.predict(X)
        if pred.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible lengths.")
        return float(np.mean(pred == y))

    # -------------------------------------------------------------------------
    # Optimization core
    # -------------------------------------------------------------------------
    def _optimize(self, X, y, verbose=False):
        prev_loss = np.inf

        for it in range(self.n_iter):
            loss = self._log_loss_from_augmented(X, y, self.w)
            grad = self._gradient_from_augmented(X, y, self.w)

            grad_norm = np.linalg.norm(grad)
            self.loss_history_.append(loss)
            self.n_iter_run_ += 1

            if verbose:
                print(
                    f"Iter {self.n_iter_run_:4d}: "
                    f"loss={loss:.10f}, grad_norm={grad_norm:.6e}"
                )

            # Gradient-based convergence
            if grad_norm < self.grad_tol:
                self.converged_ = True
                if verbose:
                    print(f"Converged by grad_tol at iteration {self.n_iter_run_}.")
                break

            # Descent direction for negative log-likelihood
            direction = -grad

            # Step size
            if self.line_search:
                step_size = self._backtracking_line_search(X, y, self.w, loss, grad, direction)
            else:
                step_size = self.eta

            w_new = self.w + step_size * direction
            new_loss = self._log_loss_from_augmented(X, y, w_new)

            # Improvement-based convergence
            if abs(prev_loss - loss) < self.tol and abs(loss - new_loss) < self.tol:
                self.w = w_new
                self.converged_ = True
                if verbose:
                    print(f"Converged by loss change at iteration {self.n_iter_run_}.")
                break

            self.w = w_new
            prev_loss = loss

        if verbose and not self.converged_:
            print(f"Optimization ended after {self.n_iter_run_} iterations without declared convergence.")

    # -------------------------------------------------------------------------
    # Loss / gradient / Hessian
    # -------------------------------------------------------------------------
    def _log_loss(self, X, y):
        """
        Public loss function using raw X.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        X_aug = self._augment_X(X)
        return self._log_loss_from_augmented(X_aug, y, self.w)

    def _log_loss_from_augmented(self, X, y, w):
        """
        Negative log-likelihood + optional L2 penalty.
        """
        z = X @ w

        # Stable logistic loss:
        # log(1 + exp(z)) - y*z
        loss_vec = np.logaddexp(0.0, z) - y * z
        loss = np.sum(loss_vec)

        if self.l2 > 0.0:
            if self.fit_intercept:
                loss += 0.5 * self.l2 * np.sum(w[1:] ** 2)
            else:
                loss += 0.5 * self.l2 * np.sum(w ** 2)

        return float(loss)

    def _gradient_from_augmented(self, X, y, w):
        """
        Gradient of negative log-likelihood + optional L2 penalty.
        """
        z = X @ w
        p = self._sigmoid(z)
        grad = X.T @ (p - y)

        if self.l2 > 0.0:
            reg = self.l2 * w.copy()
            if self.fit_intercept:
                reg[0] = 0.0
            grad += reg

        return grad

    def hessian(self, X, y):
        """
        Hessian of negative log-likelihood at current weights.
        """
        self._check_is_fitted()
        X, y = self._validate_inputs(X, y)
        X_aug = self._augment_X(X)

        z = X_aug @ self.w
        p = self._sigmoid(z)
        s = p * (1.0 - p)

        H = X_aug.T @ (X_aug * s[:, None])

        if self.l2 > 0.0:
            reg = np.eye(H.shape[0]) * self.l2
            if self.fit_intercept:
                reg[0, 0] = 0.0
            H += reg

        return H

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def _backtracking_line_search(self, X, y, w, loss, grad, direction):
        """
        Armijo backtracking line search for minimizing negative log-likelihood.
        """
        alpha = self.eta
        c = 1e-4
        beta = 0.5

        directional_derivative = grad @ direction
        if directional_derivative >= 0:
            # Fallback: if direction is not descent, use steepest descent
            direction = -grad
            directional_derivative = grad @ direction

        for _ in range(50):
            candidate = w + alpha * direction
            candidate_loss = self._log_loss_from_augmented(X, y, candidate)

            if candidate_loss <= loss + c * alpha * directional_derivative:
                return alpha

            alpha *= beta

        return alpha

    def _augment_X(self, X):
        if not self.fit_intercept:
            return X

        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])

    def _validate_inputs(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows. "
                f"Got X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}."
            )

        unique_y = np.unique(y)
        if not np.all(np.isin(unique_y, [0, 1])):
            raise ValueError(f"y must contain only 0/1. Got values: {unique_y}")

        return X, y

    def _check_is_fitted(self):
        if self.w is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    @staticmethod
    def _sigmoid(x):
        """
        Numerically stable sigmoid.
        """
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)

        pos = x >= 0
        neg = ~pos

        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[neg])
        out[neg] = exp_x / (1.0 + exp_x)

        return out