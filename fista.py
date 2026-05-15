import copy
from itertools import combinations
from time import time
from typing import Optional
import numpy as np
from tqdm import tqdm


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    """Prox for L1 norm."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def group_soft_threshold(theta, groups, thresh):
    theta_new = theta.copy()

    for g in groups:
        v = theta[g]
        norm = np.linalg.norm(v, 2)

        if norm == 0:
            continue

        factor = max(0.0, 1 - thresh / norm)
        theta_new[g] = factor * v

    return theta_new


class LogisticRegressionFISTA:
    """
    L1-regularized logistic regression by FISTA.

    Objective:
        minimize_w
            mean_i log(1 + exp(-y_i * x_i^T w)) + l1 * penalty(w)

        penalty(w) = ||w||_1 for lasso
        penalty(w) = sum_g ||w_g||_2 for group lasso

    where y in {-1, +1} ideally.
    If y is in {0, 1}, it is internally converted to {-1, +1}.

    Important
    ---------
    This version uses MEAN logistic loss, not SUM logistic loss.

    If your previous code used the objective

        sum_i logistic_loss_i + l1_sum * penalty(w)

    then the equivalent mean-loss regularization is

        l1_mean = l1_sum / n_samples

    because

        sum_loss + l1_sum * penalty
        = n_samples * (mean_loss + (l1_sum / n_samples) * penalty)

    Parameters
    ----------
    eta : float or None, default=None
        Initial/max step size for backtracking. If None, fit() computes
        eta = 1 / L, where L is a Lipschitz upper bound for the gradient
        of the MEAN logistic loss.
        If a positive float is supplied, it is used as the max backtracking
        step size.
    n_iter : int, default=1000
        Maximum number of FISTA iterations.
    tol : float, default=1e-6
        Tolerance on objective decrease / iterate change.
    l1 : float, default=1.0
        L1 regularization strength for the MEAN-loss objective.
    fit_intercept : bool, default=False
        Whether to fit intercept. If True, intercept is NOT regularized.
    line_search : bool, default=True
        Whether to use backtracking line search.
    verbose : bool, default=False
        Whether to print optimization progress.
    init_w : Optional[np.ndarray], default=None
        Initial parameter. If fit_intercept=True, pass an array of length
        n_features + 1, with intercept last.
    """

    def __init__(
        self,
        eta: Optional[float] = None,
        n_iter: int = 1000,
        tol: float = 1e-6,
        l1: float = 1.0,
        fit_intercept: bool = False,
        line_search: bool = True,
        verbose: bool = False,
        init_w: Optional[np.ndarray] = None,
    ):
        self.eta = eta
        self.n_iter = n_iter
        self.tol = tol
        self.l1 = l1
        self.fit_intercept = fit_intercept
        self.line_search = line_search
        self.verbose = verbose
        self.init_w = init_w

        self.w = None
        self.b = 0.0
        self.objective_history_ = []
        self.n_iter_ = 0
        self.converged_ = False
        self.groups = None

        # Diagnostics
        self.L_ = None
        self.final_step_ = None
        self.max_step_ = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # numerically stable sigmoid
        out = np.empty_like(z)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    @staticmethod
    def _log1pexp(z: np.ndarray) -> np.ndarray:
        # stable log(1 + exp(z))
        return np.logaddexp(0.0, z)

    def _prepare_labels(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        uniq = np.unique(y)

        if np.array_equal(uniq, [0.0, 1.0]) or np.array_equal(uniq, [0.0]) or np.array_equal(uniq, [1.0]):
            y = 2.0 * y - 1.0
        elif np.array_equal(uniq, [-1.0, 1.0]) or np.array_equal(uniq, [-1.0]) or np.array_equal(uniq, [1.0]):
            pass
        else:
            raise ValueError("y must be binary labels in {0,1} or {-1,+1}.")
        return y

    def _split_params(self, theta: np.ndarray):
        if self.fit_intercept:
            return theta[:-1], theta[-1]
        return theta, 0.0

    def _merge_params(self, w: np.ndarray, b: float) -> np.ndarray:
        if self.fit_intercept:
            return np.concatenate([w, np.array([b])])
        return w.copy()

    def _decision_function(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        w, b = self._split_params(theta)
        return X @ w + b

    def _smooth_loss(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """Mean logistic loss: mean_i log(1 + exp(-y_i * x_i^T theta))."""
        z = self._decision_function(X, theta)
        yz = y * z
        return float(np.mean(self._log1pexp(-yz)))

    def _penalty(self, theta: np.ndarray, is_group: bool = False) -> float:
        """Regularization penalty matching the prox used in fit()."""
        w, _ = self._split_params(theta)

        if is_group:
            if self.groups is None:
                raise ValueError("groups must be set before fitting with is_group=True.")
            return float(sum(np.linalg.norm(w[g], 2) for g in self.groups))

        return float(np.sum(np.abs(w)))

    def _full_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        is_group: bool = False,
    ) -> float:
        return self._smooth_loss(X, y, theta) + self.l1 * self._penalty(theta, is_group=is_group)

    def _smooth_grad(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Gradient of the smooth part:
            mean_i log(1 + exp(-y_i z_i))
        """
        n_samples = X.shape[0]
        z = self._decision_function(X, theta)
        yz = y * z

        # derivative wrt z_i: -y_i / (1 + exp(y_i z_i))
        # For mean loss, divide by n_samples.
        coeff = -y * self._sigmoid(-yz) / n_samples  # shape (n,)
        grad_w = X.T @ coeff

        if self.fit_intercept:
            grad_b = np.sum(coeff)
            return np.concatenate([grad_w, np.array([grad_b])])
        return grad_w

    def _prox(self, theta: np.ndarray, step: float) -> np.ndarray:
        """
        Prox for L1 regularization.
        Intercept, if present, is not regularized.
        """
        if self.fit_intercept:
            w_new = soft_threshold(theta[:-1], step * self.l1)
            b_new = theta[-1]
            return np.concatenate([w_new, np.array([b_new])])
        return soft_threshold(theta, step * self.l1)

    def _group_prox(self, theta: np.ndarray, step: float) -> np.ndarray:
        thresh = step * self.l1

        if self.fit_intercept:
            w = theta[:-1]
            b = theta[-1]

            w_new = group_soft_threshold(w, self.groups, thresh)

            return np.concatenate([w_new, np.array([b])])

        return group_soft_threshold(theta, self.groups, thresh)

    def _quadratic_upper_bound(
        self,
        X: np.ndarray,
        y: np.ndarray,
        x_new: np.ndarray,
        yk: np.ndarray,
        grad_yk: np.ndarray,
        step: float,
    ) -> float:
        diff = x_new - yk
        return (
            self._smooth_loss(X, y, yk)
            + grad_yk @ diff
            + 0.5 / step * np.dot(diff, diff)
        )

    def _compute_lipschitz(self, X: np.ndarray) -> float:
        """Lipschitz upper bound for gradient of MEAN logistic loss."""
        n_samples = X.shape[0]

        if self.fit_intercept:
            X_aug = np.column_stack([X, np.ones(n_samples)])
        else:
            X_aug = X

        # For logistic loss, max sigmoid'(z) <= 1/4.
        # Mean loss divides the Hessian by n_samples.
        eigmax = np.linalg.eigvalsh(X_aug.T @ X_aug).max()
        L = 0.25 * eigmax / n_samples

        if not np.isfinite(L) or L <= 0.0:
            L = 1.0

        return float(L)

    def fit(self, X: np.ndarray, y: np.ndarray, is_group=False, *_args, **_kwargs):
        self.converged_ = False
        self.n_iter_ = 0

        X = np.asarray(X, dtype=float)
        y = self._prepare_labels(y)

        n_samples, n_features = X.shape
        dim = n_features + int(self.fit_intercept)

        if self.init_w is None:
            xk = np.zeros(dim, dtype=float)
        else:
            xk = np.asarray(self.init_w, dtype=float).copy()
            if xk.shape[0] != dim:
                raise ValueError(
                    f"init_w has length {xk.shape[0]}, but expected {dim}."
                )

        L = self._compute_lipschitz(X)
        self.L_ = L

        # max_step is the largest step size used as the starting point for
        # backtracking. If eta is None, use the theoretically safe 1/L.
        if self.eta is None:
            max_step = 1.0 / L
        else:
            max_step = float(self.eta)
            if not np.isfinite(max_step) or max_step <= 0.0:
                raise ValueError("eta must be None or a positive finite float.")

        self.max_step_ = max_step
        step = max_step

        yk = xk.copy()
        tk = 1.0

        obj_prev = self._full_objective(X, y, xk, is_group=is_group)
        self.objective_history_ = [obj_prev]

        for it in range(1, self.n_iter + 1):
            grad_yk = self._smooth_grad(X, y, yk)

            bt = 0
            if self.line_search:
                # Let step recover if it was reduced previously, but never
                # exceed max_step.
                step_local = min(step * 2.0, max_step)

                while True:
                    if is_group:
                        x_next = self._group_prox(yk - step_local * grad_yk, step_local)
                    else:
                        x_next = self._prox(yk - step_local * grad_yk, step_local)

                    lhs = self._smooth_loss(X, y, x_next)
                    rhs = self._quadratic_upper_bound(X, y, x_next, yk, grad_yk, step_local)

                    if lhs <= rhs + 1e-14:
                        step = step_local
                        break

                    step_local *= 0.5
                    bt += 1

                    if step_local < 1e-20:
                        step = step_local
                        break
            else:
                if is_group:
                    x_next = self._group_prox(yk - step * grad_yk, step)
                else:
                    x_next = self._prox(yk - step * grad_yk, step)

            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
            y_next = x_next + ((tk - 1.0) / t_next) * (x_next - xk)

            obj = self._full_objective(X, y, x_next, is_group=is_group)

            # Monotone restart: if acceleration worsens the full objective,
            # take one prox-gradient step from xk and reset momentum.
            if obj > obj_prev:
                grad_xk = self._smooth_grad(X, y, xk)
                bt_restart = 0

                if self.line_search:
                    step_local = min(step * 2.0, max_step)

                    while True:
                        if is_group:
                            x_next = self._group_prox(xk - step_local * grad_xk, step_local)
                        else:
                            x_next = self._prox(xk - step_local * grad_xk, step_local)

                        lhs = self._smooth_loss(X, y, x_next)
                        rhs = self._quadratic_upper_bound(X, y, x_next, xk, grad_xk, step_local)

                        if lhs <= rhs + 1e-14:
                            step = step_local
                            break

                        step_local *= 0.5
                        bt_restart += 1

                        if step_local < 1e-20:
                            step = step_local
                            break
                else:
                    if is_group:
                        x_next = self._group_prox(xk - step * grad_xk, step)
                    else:
                        x_next = self._prox(xk - step * grad_xk, step)

                t_next = 1.0
                y_next = x_next.copy()
                obj = self._full_objective(X, y, x_next, is_group=is_group)
                bt += bt_restart

            self.objective_history_.append(obj)

            # Gradient mapping norm at yk.
            grad_map = (yk - x_next) / step
            grad_map_norm = np.linalg.norm(grad_map)

            # stopping checks
            rel_obj = abs(obj_prev - obj) / max(1.0, abs(obj_prev))
            step_norm = np.linalg.norm(x_next - xk)
            rel_step = step_norm / max(1.0, np.linalg.norm(xk))

            if self.verbose and (it == 1 or it % 10 == 0):
                nnz = np.count_nonzero(np.abs(self._split_params(x_next)[0]) > 0)
                print(
                    f"[FISTA] iter={it:4d}  obj={obj:.12e}  "
                    f"rel_obj={rel_obj:.3e}  step={step:.3e}  "
                    f"step_norm={step_norm:.3e}  grad_map={grad_map_norm:.3e}  "
                    f"bt={bt}  nnz={nnz}"
                )

            if rel_obj < self.tol or rel_step < self.tol * 10:
                xk = x_next
                self.converged_ = True
                self.n_iter_ = it
                break

            xk = x_next
            yk = y_next
            tk = t_next
            obj_prev = obj
            self.n_iter_ = it

        w, b = self._split_params(xk)
        self.w = w
        self.b = b
        self.final_step_ = step

        print("Is converged? ... ", self.converged_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = X @ self.w + self.b
        p1 = self._sigmoid(z)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)
