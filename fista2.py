from typing import Optional

import numpy as np


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    """Proximal operator for the L1 norm."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def group_soft_threshold(
    theta: np.ndarray,
    groups,
    thresh: float,
) -> np.ndarray:
    """Block soft-thresholding for non-overlapping group lasso groups."""
    theta_new = theta.copy()

    for g in groups:
        v = theta[g]
        norm = np.linalg.norm(v, 2)

        if norm == 0.0:
            continue

        factor = max(0.0, 1.0 - thresh / norm)
        theta_new[g] = factor * v

    return theta_new


class LogisticRegressionFISTA:
    """
    L1/group-lasso regularized logistic regression solved by FISTA.

    Objective
    ---------
        minimize_theta
            mean_i log(1 + exp(-y_i * z_i)) + l1 * penalty(theta)

    where
        z_i = x_i^T w + b

    and y is in {-1, +1}. Labels in {0, 1} are converted internally.

    Notes
    -----
    - The logistic loss is a MEAN loss, not a SUM loss.
    - If ``fit_intercept=True``, the intercept is not regularized.
    - ``X`` is not forcibly converted to float64. A float32 design matrix
      therefore remains float32, avoiding a potentially huge full-matrix copy.
    - ``lipschitz`` can be supplied externally and reused across a lambda path.
      When omitted, a power-iteration estimate is computed without forming X.T @ X.
      Backtracking line search then enforces the local quadratic upper bound.

    Parameters
    ----------
    eta : float or None, default=None
        Maximum step size used by backtracking. If None, use 1 / L, where L is
        supplied through ``lipschitz`` or estimated in fit().
    n_iter : int, default=1000
        Maximum number of FISTA iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    l1 : float, default=1.0
        Regularization strength for the mean-loss objective.
    fit_intercept : bool, default=False
        Whether to fit an unregularized intercept.
    line_search : bool, default=True
        Whether to use backtracking line search.
    verbose : bool, default=False
        Whether to print optimization diagnostics.
    init_w : np.ndarray or None, default=None
        Initial parameter vector. With intercept, its length must be
        n_features + 1 and the last element is the intercept.
    lipschitz : float or None, default=None
        Optional precomputed Lipschitz estimate/bound for the smooth gradient.
        Reuse the same value for repeated fits with the same X.
    lipschitz_power_iter : int, default=20
        Number of power iterations used when ``lipschitz`` is not supplied.
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
        lipschitz: Optional[float] = None,
        lipschitz_power_iter: int = 20,
    ):
        self.eta = eta
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.l1 = float(l1)
        self.fit_intercept = bool(fit_intercept)
        self.line_search = bool(line_search)
        self.verbose = bool(verbose)
        self.init_w = init_w
        self.lipschitz = lipschitz
        self.lipschitz_power_iter = int(lipschitz_power_iter)

        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if self.tol <= 0.0 or not np.isfinite(self.tol):
            raise ValueError("tol must be a positive finite float.")
        if self.l1 < 0.0 or not np.isfinite(self.l1):
            raise ValueError("l1 must be a nonnegative finite float.")
        if self.lipschitz_power_iter <= 0:
            raise ValueError("lipschitz_power_iter must be positive.")

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
        self.grad_map_norm_ = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        out = np.empty_like(z)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    @staticmethod
    def _log1pexp(z: np.ndarray) -> np.ndarray:
        """Numerically stable log(1 + exp(z))."""
        return np.logaddexp(0.0, z)

    def _prepare_labels(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        uniq = np.unique(y)

        if (
            np.array_equal(uniq, [0.0, 1.0])
            or np.array_equal(uniq, [0.0])
        ):
            y = 2.0 * y - 1.0
        elif (
            np.array_equal(uniq, [-1.0, 1.0])
            or np.array_equal(uniq, [-1.0])
            or np.array_equal(uniq, [1.0])
        ):
            pass
        else:
            raise ValueError("y must be binary labels in {0,1} or {-1,+1}.")

        return y

    def _split_params(self, theta: np.ndarray):
        if self.fit_intercept:
            return theta[:-1], theta[-1]
        return theta, 0.0

    def _decision_function(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        w, b = self._split_params(theta)
        return X @ w + b

    def _smooth_loss_from_z(self, y: np.ndarray, z: np.ndarray) -> float:
        yz = y * z
        return float(np.mean(self._log1pexp(-yz)))

    def _smooth_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
    ) -> float:
        z = self._decision_function(X, theta)
        return self._smooth_loss_from_z(y, z)

    def _smooth_grad_from_z(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        n_samples = X.shape[0]
        yz = y * z

        # d/dz log(1 + exp(-y z)) = -y / (1 + exp(y z))
        coeff = -y * self._sigmoid(-yz) / n_samples
        grad_w = X.T @ coeff

        if self.fit_intercept:
            grad_b = np.sum(coeff)
            return np.concatenate([grad_w, np.array([grad_b])])

        return grad_w

    def _penalty(self, theta: np.ndarray, is_group: bool = False) -> float:
        """Regularization penalty matching the proximal operator."""
        w, _ = self._split_params(theta)

        if is_group:
            if self.groups is None:
                raise ValueError(
                    "groups must be set before fitting with is_group=True."
                )
            return float(sum(np.linalg.norm(w[g], 2) for g in self.groups))

        return float(np.sum(np.abs(w)))

    def _full_objective_from_smooth(
        self,
        smooth_loss: float,
        theta: np.ndarray,
        is_group: bool,
    ) -> float:
        return smooth_loss + self.l1 * self._penalty(theta, is_group=is_group)

    def _full_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        is_group: bool = False,
    ) -> float:
        smooth = self._smooth_loss(X, y, theta)
        return self._full_objective_from_smooth(smooth, theta, is_group)

    def _prox(self, theta: np.ndarray, step: float) -> np.ndarray:
        """Proximal operator for L1 regularization."""
        if self.fit_intercept:
            w_new = soft_threshold(theta[:-1], step * self.l1)
            return np.concatenate([w_new, np.array([theta[-1]])])

        return soft_threshold(theta, step * self.l1)

    def _group_prox(self, theta: np.ndarray, step: float) -> np.ndarray:
        thresh = step * self.l1

        if self.fit_intercept:
            w_new = group_soft_threshold(theta[:-1], self.groups, thresh)
            return np.concatenate([w_new, np.array([theta[-1]])])

        return group_soft_threshold(theta, self.groups, thresh)

    @staticmethod
    def _quadratic_upper_bound_from_smooth(
        smooth_at_base: float,
        x_new: np.ndarray,
        base: np.ndarray,
        grad_base: np.ndarray,
        step: float,
    ) -> float:
        diff = x_new - base
        return float(
            smooth_at_base
            + grad_base @ diff
            + 0.5 / step * np.dot(diff, diff)
        )

    def _compute_lipschitz(self, X: np.ndarray) -> float:
        """
        Estimate a Lipschitz constant for the gradient of mean logistic loss.

        Uses power iteration on the augmented linear operator without forming
        X.T @ X. For logistic loss,

            L <= 0.25 * ||X_aug||_2^2 / n_samples.

        The power iteration gives an estimate rather than a certified upper
        bound; when line_search=True, backtracking provides the final safeguard.
        """
        n_samples, n_features = X.shape
        dim = n_features + int(self.fit_intercept)

        # Deterministic initialization keeps repeated runs reproducible.
        v = np.ones(dim, dtype=np.float64)
        v /= np.linalg.norm(v)

        eig_est = 0.0

        for _ in range(self.lipschitz_power_iter):
            if self.fit_intercept:
                w = v[:-1]
                b = v[-1]
                Xv = X @ w + b
                z = np.empty_like(v)
                z[:-1] = X.T @ Xv
                z[-1] = np.sum(Xv)
            else:
                Xv = X @ v
                z = X.T @ Xv

            norm_z = np.linalg.norm(z)
            if not np.isfinite(norm_z) or norm_z == 0.0:
                return 1.0

            v = z / norm_z
            eig_est = float(v @ z)

        L = 0.25 * eig_est / n_samples

        if not np.isfinite(L) or L <= 0.0:
            return 1.0

        return float(L)

    def _validate_groups(self, n_features: int) -> None:
        if self.groups is None:
            raise ValueError("groups must be set before fitting with is_group=True.")

        for idx, g in enumerate(self.groups):
            g_arr = np.asarray(g)
            if g_arr.ndim != 1 or g_arr.size == 0:
                raise ValueError(f"groups[{idx}] must be a non-empty 1D index array.")
            if np.any(g_arr < 0) or np.any(g_arr >= n_features):
                raise ValueError(f"groups[{idx}] contains an out-of-range index.")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_group: bool = False,
    ):
        self.converged_ = False
        self.n_iter_ = 0
        self.grad_map_norm_ = None

        # Important: preserve float32 X instead of forcing a huge float64 copy.
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float32, copy=False)

        y = self._prepare_labels(y)

        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X has {n_samples} rows, but y has length {y.shape[0]}."
            )
        if n_samples == 0 or n_features == 0:
            raise ValueError("X must have at least one row and one column.")

        if is_group:
            self._validate_groups(n_features)

        dim = n_features + int(self.fit_intercept)

        if self.init_w is None:
            xk = np.zeros(dim, dtype=np.float64)
        else:
            xk = np.asarray(self.init_w, dtype=np.float64).copy()
            if xk.ndim != 1 or xk.shape[0] != dim:
                raise ValueError(
                    f"init_w has shape {xk.shape}, but expected ({dim},)."
                )
            if not np.all(np.isfinite(xk)):
                raise ValueError("init_w must contain only finite values.")

        if self.lipschitz is None:
            L = self._compute_lipschitz(X)
        else:
            L = float(self.lipschitz)
            if not np.isfinite(L) or L <= 0.0:
                raise ValueError("lipschitz must be a positive finite float.")

        self.L_ = L

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

        z_xk = self._decision_function(X, xk)
        smooth_xk = self._smooth_loss_from_z(y, z_xk)
        obj_prev = self._full_objective_from_smooth(
            smooth_xk,
            xk,
            is_group,
        )
        self.objective_history_ = [obj_prev]

        for it in range(1, self.n_iter + 1):
            # Compute X @ yk once and reuse it for both loss and gradient.
            z_yk = self._decision_function(X, yk)
            smooth_yk = self._smooth_loss_from_z(y, z_yk)
            grad_yk = self._smooth_grad_from_z(X, y, z_yk)

            bt = 0
            prox_base = yk

            if self.line_search:
                step_local = min(step * 2.0, max_step)

                while True:
                    candidate = yk - step_local * grad_yk
                    if is_group:
                        x_next = self._group_prox(candidate, step_local)
                    else:
                        x_next = self._prox(candidate, step_local)

                    z_next = self._decision_function(X, x_next)
                    smooth_next = self._smooth_loss_from_z(y, z_next)
                    rhs = self._quadratic_upper_bound_from_smooth(
                        smooth_yk,
                        x_next,
                        yk,
                        grad_yk,
                        step_local,
                    )

                    if smooth_next <= rhs + 1e-14:
                        step = step_local
                        break

                    step_local *= 0.5
                    bt += 1

                    if step_local < 1e-20:
                        raise RuntimeError(
                            "Backtracking line search failed: step size became "
                            "smaller than 1e-20."
                        )
            else:
                candidate = yk - step * grad_yk
                if is_group:
                    x_next = self._group_prox(candidate, step)
                else:
                    x_next = self._prox(candidate, step)

                z_next = self._decision_function(X, x_next)
                smooth_next = self._smooth_loss_from_z(y, z_next)

            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
            y_next = x_next + ((tk - 1.0) / t_next) * (x_next - xk)

            obj = self._full_objective_from_smooth(
                smooth_next,
                x_next,
                is_group,
            )

            # Monotone restart if acceleration worsens the full objective.
            if obj > obj_prev:
                # Reuse z_xk / smooth_xk from the current accepted iterate.
                grad_xk = self._smooth_grad_from_z(X, y, z_xk)
                bt_restart = 0
                prox_base = xk

                if self.line_search:
                    step_local = min(step * 2.0, max_step)

                    while True:
                        candidate = xk - step_local * grad_xk
                        if is_group:
                            x_next = self._group_prox(candidate, step_local)
                        else:
                            x_next = self._prox(candidate, step_local)

                        z_next = self._decision_function(X, x_next)
                        smooth_next = self._smooth_loss_from_z(y, z_next)
                        rhs = self._quadratic_upper_bound_from_smooth(
                            smooth_xk,
                            x_next,
                            xk,
                            grad_xk,
                            step_local,
                        )

                        if smooth_next <= rhs + 1e-14:
                            step = step_local
                            break

                        step_local *= 0.5
                        bt_restart += 1

                        if step_local < 1e-20:
                            raise RuntimeError(
                                "Backtracking line search failed during restart: "
                                "step size became smaller than 1e-20."
                            )
                else:
                    candidate = xk - step * grad_xk
                    if is_group:
                        x_next = self._group_prox(candidate, step)
                    else:
                        x_next = self._prox(candidate, step)

                    z_next = self._decision_function(X, x_next)
                    smooth_next = self._smooth_loss_from_z(y, z_next)

                t_next = 1.0
                y_next = x_next.copy()
                obj = self._full_objective_from_smooth(
                    smooth_next,
                    x_next,
                    is_group,
                )
                bt += bt_restart

            self.objective_history_.append(obj)

            # Correct base point even after a monotone restart.
            grad_map = (prox_base - x_next) / step
            grad_map_norm = float(np.linalg.norm(grad_map))
            self.grad_map_norm_ = grad_map_norm

            rel_obj = abs(obj_prev - obj) / max(1.0, abs(obj_prev))
            step_norm = float(np.linalg.norm(x_next - xk))
            rel_step = step_norm / max(1.0, float(np.linalg.norm(xk)))

            if self.verbose and (it == 1 or it % 10 == 0):
                nnz = np.count_nonzero(
                    np.abs(self._split_params(x_next)[0]) > 0
                )
                print(
                    f"[FISTA] iter={it:4d}  obj={obj:.12e}  "
                    f"rel_obj={rel_obj:.3e}  step={step:.3e}  "
                    f"step_norm={step_norm:.3e}  "
                    f"grad_map={grad_map_norm:.3e}  "
                    f"bt={bt}  nnz={nnz}"
                )

            # Require both objective and iterate stabilization.
            if rel_obj < self.tol and rel_step < self.tol * 10.0:
                xk = x_next
                z_xk = z_next
                smooth_xk = smooth_next
                self.converged_ = True
                self.n_iter_ = it
                break

            xk = x_next
            z_xk = z_next
            smooth_xk = smooth_next
            yk = y_next
            tk = t_next
            obj_prev = obj
            self.n_iter_ = it

        w, b = self._split_params(xk)
        self.w = w
        self.b = float(b)
        self.final_step_ = step

        if self.verbose:
            print("Is converged? ...", self.converged_)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Call fit() before predict_proba().")

        X = np.asarray(X)
        z = X @ self.w + self.b
        p1 = self._sigmoid(z)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)
