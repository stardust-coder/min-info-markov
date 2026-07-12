import numpy as np
from scipy.optimize import minimize
from scipy.special import i0e, logsumexp
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


def _wrap_angle(x):
    """Map angles to [-pi, pi)."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def _log_i0(kappa):
    """
    Stable log I0(kappa), where I0 is modified Bessel function.
    scipy.special.i0e(kappa) = exp(-abs(kappa)) I0(kappa).
    """
    kappa = np.asarray(kappa)
    return np.log(i0e(kappa)) + np.abs(kappa)


def _make_features(X, target_j, source_k=None, lag=1):
    """
    Construct design matrix for AR-von-Mises regression.

    Reduced model:
        y_t ~ y_{t-lag}

    Full model:
        y_t ~ y_{t-lag}, x_{t-lag,k}

    Natural parameter:
        eta_t = [a_t, b_t]
        log p(y_t | features)
        = a_t cos(y_t) + b_t sin(y_t) - log(2 pi I0(kappa_t))
        where kappa_t = sqrt(a_t^2 + b_t^2).
    """
    X = _wrap_angle(np.asarray(X))
    N, p = X.shape

    if lag < 1:
        raise ValueError("lag must be >= 1.")
    if N <= lag:
        raise ValueError("N must be larger than lag.")

    y = X[lag:, target_j]
    y_lag = X[:-lag, target_j]

    cols = [
        np.ones_like(y),
        np.cos(y_lag),
        np.sin(y_lag),
    ]

    if source_k is not None:
        x_lag = X[:-lag, source_k]
        cols += [
            np.cos(x_lag),
            np.sin(x_lag),
        ]

    Z = np.column_stack(cols)
    return Z, y


def _neg_loglik_and_grad(beta, Z, y, l2=1e-6):
    """
    Negative log-likelihood and gradient for von Mises regression.

    beta has shape (2 * q,).
    First q parameters map features to a_t.
    Second q parameters map features to b_t.
    """
    n, q = Z.shape
    beta_a = beta[:q]
    beta_b = beta[q:]

    a = Z @ beta_a
    b = Z @ beta_b

    cy = np.cos(y)
    sy = np.sin(y)

    kappa = np.sqrt(a * a + b * b)
    log_norm = np.log(2 * np.pi) + _log_i0(kappa)

    logp = a * cy + b * sy - log_norm
    nll = -np.mean(logp)

    # L2 penalty, excluding intercepts beta_a[0], beta_b[0]
    if l2 > 0:
        pen = 0.5 * l2 * (
            np.sum(beta_a[1:] ** 2) + np.sum(beta_b[1:] ** 2)
        )
        nll += pen

    # Gradient of log I0(kappa):
    # d/dkappa log I0(kappa) = I1(kappa) / I0(kappa).
    # Use stable approximation via scipy i0e and finite safe expression.
    # scipy.special.i1e could also be used, but using np where below is enough
    # if we compute ratio by asymptotic fallback.
    from scipy.special import i1e

    ratio = np.empty_like(kappa)
    small = kappa < 1e-8
    ratio[small] = kappa[small] / 2.0
    ratio[~small] = i1e(kappa[~small]) / i0e(kappa[~small])

    # d nll / d a_t = -cos(y_t) + ratio * a_t/kappa
    # d nll / d b_t = -sin(y_t) + ratio * b_t/kappa
    a_over_k = np.zeros_like(a)
    b_over_k = np.zeros_like(b)
    a_over_k[~small] = a[~small] / kappa[~small]
    b_over_k[~small] = b[~small] / kappa[~small]

    grad_a_t = -cy + ratio * a_over_k
    grad_b_t = -sy + ratio * b_over_k

    grad_a = Z.T @ grad_a_t / n
    grad_b = Z.T @ grad_b_t / n

    if l2 > 0:
        grad_a[1:] += l2 * beta_a[1:]
        grad_b[1:] += l2 * beta_b[1:]

    grad = np.concatenate([grad_a, grad_b])
    return nll, grad


def fit_von_mises_regression(Z, y, l2=1e-6, max_iter=500):
    """
    Fit von Mises regression by maximum likelihood.

    Returns
    -------
    beta : ndarray, shape (2 * q,)
    result : scipy OptimizeResult
    """
    q = Z.shape[1]
    beta0 = np.zeros(2 * q)

    # Initialize intercept by sample mean direction.
    c = np.mean(np.cos(y))
    s = np.mean(np.sin(y))
    r = np.sqrt(c * c + s * s)

    # Approximate kappa from resultant length.
    # Standard circular statistics approximation.
    if r < 1e-6:
        kappa0 = 0.0
    elif r < 0.53:
        kappa0 = 2 * r + r**3 + 5 * r**5 / 6
    elif r < 0.85:
        kappa0 = -0.4 + 1.39 * r + 0.43 / (1 - r)
    else:
        kappa0 = 1 / (r**3 - 4 * r**2 + 3 * r)

    mu0 = np.arctan2(s, c)
    beta0[0] = kappa0 * np.cos(mu0)
    beta0[q] = kappa0 * np.sin(mu0)

    result = minimize(
        fun=lambda b: _neg_loglik_and_grad(b, Z, y, l2=l2),
        x0=beta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-6},
    )

    return result.x, result


def loglik_von_mises_regression(beta, Z, y):
    """Return per-sample log-likelihood values."""
    n, q = Z.shape
    beta_a = beta[:q]
    beta_b = beta[q:]

    a = Z @ beta_a
    b = Z @ beta_b
    kappa = np.sqrt(a * a + b * b)

    logp = (
        a * np.cos(y)
        + b * np.sin(y)
        - np.log(2 * np.pi)
        - _log_i0(kappa)
    )
    return logp


def pairwise_ar_von_mises_te_scores(
    X,
    lag=1,
    train_fraction=0.7,
    l2=1e-6,
    max_iter=500,
    exclude_self=True,
    verbose=False,
):
    """
    Compute pairwise AR-von-Mises TE scores for all directed pairs k -> j.

    Parameters
    ----------
    X : ndarray, shape (N, p)
        Circular time series in radians.
    lag : int
        Lag length. Currently uses only x_{t-lag}.
    train_fraction : float
        Fraction of usable time points used for training.
    l2 : float
        Small ridge penalty for numerical stability.
    max_iter : int
        Max optimizer iterations.
    exclude_self : bool
        If True, self edges j -> j are set to nan.
    verbose : bool
        Print progress.

    Returns
    -------
    scores : ndarray, shape (p, p)
        scores[j, k] is TE score for k -> j.
        Diagonal is nan if exclude_self=True.

    info : dict
        Optimization diagnostics.
    """
    X = _wrap_angle(np.asarray(X))
    N, p = X.shape

    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be in (0, 1).")

    usable_n = N - lag
    n_train = int(np.floor(train_fraction * usable_n))

    if n_train < 10 or usable_n - n_train < 10:
        raise ValueError("Train/test split too small. Increase N or adjust train_fraction.")

    scores = np.full((p, p), np.nan)
    info = {
        "reduced_success": np.zeros(p, dtype=bool),
        "full_success": np.zeros((p, p), dtype=bool),
        "reduced_fun": np.full(p, np.nan),
        "full_fun": np.full((p, p), np.nan),
    }

    # Fit reduced model once per target j.
    reduced_models = {}

    for j in range(p):
        Z_red, y = _make_features(X, target_j=j, source_k=None, lag=lag)

        Z_red_train = Z_red[:n_train]
        y_train = y[:n_train]

        beta_red, res_red = fit_von_mises_regression(
            Z_red_train, y_train, l2=l2, max_iter=max_iter
        )

        reduced_models[j] = beta_red
        info["reduced_success"][j] = res_red.success
        info["reduced_fun"][j] = res_red.fun

        if verbose:
            print(f"Target {j}: reduced success={res_red.success}, fun={res_red.fun:.4f}")

    # Fit full model for each directed pair k -> j.
    for j in range(p):
        Z_red, y = _make_features(X, target_j=j, source_k=None, lag=lag)
        y_test = y[n_train:]
        Z_red_test = Z_red[n_train:]

        beta_red = reduced_models[j]
        ll_red = loglik_von_mises_regression(beta_red, Z_red_test, y_test)

        for k in range(p):
            if exclude_self and j == k:
                scores[j, k] = np.nan
                continue

            Z_full, y_full = _make_features(X, target_j=j, source_k=k, lag=lag)

            Z_full_train = Z_full[:n_train]
            y_full_train = y_full[:n_train]
            Z_full_test = Z_full[n_train:]
            y_full_test = y_full[n_train:]

            beta_full, res_full = fit_von_mises_regression(
                Z_full_train, y_full_train, l2=l2, max_iter=max_iter
            )

            ll_full = loglik_von_mises_regression(beta_full, Z_full_test, y_full_test)

            # TE estimate as test log-likelihood improvement.
            te = np.mean(ll_full - ll_red)

            # Finite-sample estimates can become slightly negative.
            # For graph scoring, clipping is common, but you can remove this if desired.
            scores[j, k] = max(0.0, te)

            info["full_success"][j, k] = res_full.success
            info["full_fun"][j, k] = res_full.fun

            if verbose:
                print(
                    f"{k} -> {j}: TE={scores[j,k]:.6f}, "
                    f"success={res_full.success}"
                )

    return scores, info


def evaluate_directed_graph_scores(
    scores,
    A_true,
    threshold=None,
    top_m=None,
    exclude_self=True,
):
    """
    Evaluate directed graph recovery.

    Parameters
    ----------
    scores : ndarray, shape (p, p)
        scores[j, k] is edge score for k -> j.
    A_true : ndarray, shape (p, p)
        Ground truth adjacency with A_true[j, k] = 1 if k -> j exists.
    threshold : float or None
        Predict edge if score > threshold.
        If None and top_m is None, threshold is chosen so predicted edge count
        equals the number of true edges.
    top_m : int or None
        Predict exactly top_m edges by score.
        Overrides threshold if provided.
    exclude_self : bool
        Exclude diagonal entries.

    Returns
    -------
    metrics : dict
    y_true : ndarray
    y_score : ndarray
    y_pred : ndarray
    """
    scores = np.asarray(scores)
    A_true = np.asarray(A_true).astype(int)

    if scores.shape != A_true.shape:
        raise ValueError("scores and A_true must have the same shape.")

    p = scores.shape[0]
    mask = np.ones_like(A_true, dtype=bool)

    if exclude_self:
        np.fill_diagonal(mask, False)

    mask &= np.isfinite(scores)

    y_true = A_true[mask].astype(int)
    y_score = scores[mask].astype(float)

    if top_m is not None:
        top_m = int(top_m)
        if top_m < 0 or top_m > len(y_score):
            raise ValueError("top_m must be between 0 and number of candidate edges.")

        y_pred = np.zeros_like(y_true)
        if top_m > 0:
            idx = np.argsort(y_score)[::-1][:top_m]
            y_pred[idx] = 1

        used_threshold = None

    else:
        if threshold is None:
            # Match number of predicted edges to number of true edges.
            m_true = int(np.sum(y_true))
            y_pred = np.zeros_like(y_true)

            if m_true > 0:
                idx = np.argsort(y_score)[::-1][:m_true]
                y_pred[idx] = 1

            used_threshold = None
        else:
            used_threshold = float(threshold)
            y_pred = (y_score > used_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "num_true_edges": int(np.sum(y_true)),
        "num_pred_edges": int(np.sum(y_pred)),
        "threshold": used_threshold,
    }

    # Ranking metrics.
    if len(np.unique(y_true)) == 2:
        metrics["AUROC"] = float(roc_auc_score(y_true, y_score))
        metrics["AUPRC"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["AUROC"] = np.nan
        metrics["AUPRC"] = np.nan

    return metrics, y_true, y_score, y_pred


def run_pairwise_ar_von_mises_te_baseline(
    X,
    A_true,
    lag=1,
    train_fraction=0.7,
    l2=1e-6,
    max_iter=500,
    threshold=None,
    top_m=None,
    exclude_self=True,
    verbose=False,
):
    """
    End-to-end baseline.

    A_true[j, k] = 1 means directed edge k -> j.
    scores[j, k] is TE score for k -> j.
    """
    scores, info = pairwise_ar_von_mises_te_scores(
        X=X,
        lag=lag,
        train_fraction=train_fraction,
        l2=l2,
        max_iter=max_iter,
        exclude_self=exclude_self,
        verbose=verbose,
    )

    metrics, y_true, y_score, y_pred = evaluate_directed_graph_scores(
        scores=scores,
        A_true=A_true,
        threshold=threshold,
        top_m=top_m,
        exclude_self=exclude_self,
    )

    return {
        "scores": scores,
        "metrics": metrics,
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred,
        "optimization_info": info,
    }

if __name__ == "__main__":
    # X: shape (N, 25)
    # A_true: shape (25, 25)
    # A_true[j, k] = 1 if k -> j is a true directed edge.
    
    from data_sim import Kuramoto_Model
    raw, K_true = Kuramoto_Model(N=25)
    A_true = (np.abs(K_true) > 0).astype(int)
    np.fill_diagonal(A_true, 0)
    result = run_pairwise_ar_von_mises_te_baseline(
        X=raw,
        A_true=A_true,
        lag=1,
        train_fraction=0.7,
        l2=1e-6,
        max_iter=500,
        threshold=1e-2,   # Noneなら真のエッジ数と同じ数だけ上位をedgeにする
        top_m=None,
        exclude_self=True,
        verbose=False,
    )
    TE_scores = result["scores"]

    print(result["metrics"])