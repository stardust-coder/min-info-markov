"""
Monte Carlo optimism-bias check for the TIC-type correction of Besag BPLE.

This L-BFGS-B ridge-stabilized version is aligned with the notation

    I_d = {i : d < i < n-d},      N_d = |I_d|,
    K_d = binom(N_d, 2),

and with the proposed estimator

    B_hat_{n,d} = (K_d / N_d) tr(J_hat^{-1} I_hat),

where

    phi_i_hat = 1/(N_d-1) sum_{j in I_d, j != i} s_ij(theta_hat),
    I_hat     = 4/N_d sum_i phi_i_hat phi_i_hat^T,
    J_hat     = -1/K_d sum_{i<j} H_ij(theta_hat).

For diagnostics, the script also records a Newey--West variant based on the
endpoint sequence phi_i_hat.  The theory-scaled estimator and the NW variant
are both saved in the output CSV and summary.

The Monte Carlo target is

    B_{n,d} = E[S_{n,d}(theta_hat)] - K_d E[M_{n,d}(theta_hat)],

estimated by outer replications plus an inner independent Monte Carlo bank:

    B_true_rep ~= S_obs(theta_hat) - mean_m S_inner,m(theta_hat).

You must ensure sample_from_true() generates data from the target true model q.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from time import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.optimize import minimize

from sampling import sample_from_mininfo_markov


def torus_pair_feature(x_now, x_lag):
    """
    x_now, x_lag: shape (dim,)

    Feature order within one lag:
        for i in range(dim):
            for j in range(dim):
                cc, cs, sc, ss
    """
    c_now = np.cos(x_now)
    s_now = np.sin(x_now)
    c_lag = np.cos(x_lag)
    s_lag = np.sin(x_lag)

    dim = x_now.shape[0]
    out = np.empty(4 * dim * dim, dtype=np.float64)

    k = 0
    for i in range(dim):
        for j in range(dim):
            out[k:k + 4] = [
                c_now[i] * c_lag[j],
                c_now[i] * s_lag[j],
                s_now[i] * c_lag[j],
                s_now[i] * s_lag[j],
            ]
            k += 4

    return out


def valid_positions(n: int, order: int) -> np.ndarray:
    """0-indexed positions corresponding to I_d={i: d<i<n-d} in 1-indexing.

    If the paper's i is 1-indexed, Python position p=i-1 satisfies
        order <= p < n-order-1.
    Hence this returns range(order, n-order-1).

    If your implementation convention intentionally uses d<i<=n-d instead,
    change this one function to `np.arange(order, n-order)` and all N_d/K_d
    scaling below will update consistently.
    """
    pos = np.arange(order, n - order - 1, dtype=np.int64)
    if pos.size < 2:
        raise ValueError(
            f"Need at least two valid positions, but got N_d={pos.size}. "
            f"Increase n or decrease order."
        )
    return pos


def build_pair_index_arrays(n: int, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-aligned raw positions and local endpoint indices.

    The row order matches combinations(valid_positions(n, order), 2), which is
    also used by build_X_torus().
    """
    pos = valid_positions(n, order)
    raw_pairs = np.array(list(combinations(pos.tolist(), 2)), dtype=np.int64)

    local_index = {int(p): k for k, p in enumerate(pos)}
    local_pairs = np.array(
        [(local_index[int(p)], local_index[int(q)]) for p, q in raw_pairs],
        dtype=np.int64,
    )
    return pos, raw_pairs, local_pairs


def swap_delta_torus(raw, p, q, order):
    """Compute h(original) - h(swapped) after swapping raw[p] and raw[q]."""
    n, dim = raw.shape
    per_lag = 4 * dim * dim

    delta = np.zeros(order * per_lag, dtype=np.float64)

    def after_value(idx):
        if idx == p:
            return raw[q]
        if idx == q:
            return raw[p]
        return raw[idx]

    for lag in range(1, order + 1):
        affected = {p, q, p + lag, q + lag}
        affected = [t for t in affected if lag <= t < n]

        before = np.zeros(per_lag, dtype=np.float64)
        after = np.zeros(per_lag, dtype=np.float64)

        for t in affected:
            before += torus_pair_feature(raw[t], raw[t - lag])
            after += torus_pair_feature(after_value(t), after_value(t - lag))

        start = (lag - 1) * per_lag
        end = lag * per_lag
        delta[start:end] = before - after

    return delta


def build_X_torus(raw, order, dtype=np.float32, show_progress: bool = True):
    """Build the row matrix X_ij = h(original) - h(swapped).

    Rows are ordered by combinations(valid_positions(n, order), 2), i.e. the
    strict theoretical set d<i<j<n-d.
    """
    raw = np.asarray(raw)
    n, dim = raw.shape
    _, raw_pairs, _ = build_pair_index_arrays(n, order)

    n_pairs = raw_pairs.shape[0]
    n_features = order * 4 * dim * dim
    X = np.empty((n_pairs, n_features), dtype=dtype)

    iterator = enumerate(raw_pairs)
    if show_progress:
        iterator = tqdm(iterator, total=n_pairs)

    for row, (p, q) in iterator:
        X[row] = swap_delta_torus(raw, int(p), int(q), order)

    return X


def stable_sigmoid(eta: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    eta = np.asarray(eta)
    out = np.empty_like(eta, dtype=np.float64)
    positive = eta >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
    exp_eta = np.exp(eta[~positive])
    out[~positive] = exp_eta / (1.0 + exp_eta)
    return out


def besag_log_likelihood_from_X(theta: np.ndarray, X: np.ndarray) -> float:
    theta = np.asarray(theta).reshape(-1)
    eta = X @ theta
    return float(-np.sum(np.logaddexp(0.0, -eta)))


def make_groups(dim, order, design_matrix):
    n_features = design_matrix.shape[1]
    return np.arange(n_features).reshape(-1, 4 * order)


def tic_diagnostics_from_X(
    theta: np.ndarray,
    X: np.ndarray,
    n: int,
    order: int,
    ridge: float = 1e-8,
    ridge_refit_curvature: float = 0.0,
    nw_bandwidth: int | None = None,
    center_nw: bool = True,
) -> dict[str, Any]:
    """Compute theory-scaled TIC diagnostics and a Newey--West variant.

    Theory-scaled quantities:
        J_hat = (1/K_d) sum_{i<j} pi_ij(1-pi_ij) X_ij X_ij^T
        phi_i = (1/(N_d-1)) sum_{j != i} (1-pi_ij) X_ij
        I_hat = 4/N_d sum_i phi_i phi_i^T
        B_hat = (K_d/N_d) tr(J_hat^{-1} I_hat)

    NW variant:
        Replace I_hat by a Bartlett HAC estimator computed from the sequence
        {phi_i}_{i in I_d}:
        I_hat_NW = 4 * [Gamma_0 + sum_h w_h(Gamma_h + Gamma_h^T)].
    """
    theta = np.asarray(theta).reshape(-1)
    X64 = np.asarray(X, dtype=np.float64)

    pos, _, local_pairs = build_pair_index_arrays(n, order)
    N_d = int(pos.size)
    K_d = int(local_pairs.shape[0])

    if X64.shape[0] != K_d:
        raise ValueError(
            f"X has {X64.shape[0]} rows, but K_d={K_d} from strict I_d. "
            "Check build_X_torus() and pair indexing."
        )

    eta = X64 @ theta
    pi = stable_sigmoid(eta)
    one_minus_pi = 1.0 - pi
    weight = pi * one_minus_pi

    log_likelihood = float(-np.sum(np.logaddexp(0.0, -eta)))

    # J_hat = - K_d^{-1} sum H_ij = K_d^{-1} X^T W X.
    J_hat = (X64.T @ (weight[:, None] * X64)) / K_d

    # Pairwise score s_ij(theta) = (1-pi_ij(theta)) X_ij.
    score_rows = one_minus_pi[:, None] * X64

    # Endpoint aggregation for phi_i.  Each unordered pair contributes the
    # same row-score to both endpoints, matching sum_{j != i} s_ij.
    p_dim = X64.shape[1]
    endpoint_score_sum = np.zeros((N_d, p_dim), dtype=np.float64)
    np.add.at(endpoint_score_sum, local_pairs[:, 0], score_rows)
    np.add.at(endpoint_score_sum, local_pairs[:, 1], score_rows)

    phi = endpoint_score_sum / (N_d - 1)
    I_hat_theory = 4.0 * (phi.T @ phi) / N_d

    # Newey--West/HAC variant retained as a diagnostic, not the main theory formula.
    if nw_bandwidth is None:
        q_n = int(np.floor(4.0 * (N_d / 100.0) ** (2.0 / 9.0)))
        q_n = max(1, min(q_n, N_d - 1))
    else:
        q_n = int(max(0, min(nw_bandwidth, N_d - 1)))

    phi_for_nw = phi - phi.mean(axis=0, keepdims=True) if center_nw else phi
    Gamma0 = (phi_for_nw.T @ phi_for_nw) / N_d
    I_hat_nw = Gamma0.copy()

    for h in range(1, q_n + 1):
        Gamma_h = (phi_for_nw[h:].T @ phi_for_nw[:-h]) / N_d
        bartlett_weight = 1.0 - h / (q_n + 1.0)
        I_hat_nw += bartlett_weight * (Gamma_h + Gamma_h.T)

    I_hat_nw *= 4.0

    # Numerical/estimation curvature used in TIC.
    # If theta was obtained by minimizing mean loss + 0.5 * ridge_refit * ||theta||^2,
    # the corresponding average-scale curvature is J_hat + ridge_refit * I.
    # The small `ridge` remains only a numerical stabilizer.
    J_reg = J_hat + (ridge + ridge_refit_curvature) * np.eye(p_dim)
    trace_theory = float(np.trace(np.linalg.solve(J_reg, I_hat_theory)))
    trace_nw = float(np.trace(np.linalg.solve(J_reg, I_hat_nw)))

    B_hat_theory = float((K_d / N_d) * trace_theory)
    B_hat_nw = float((K_d / N_d) * trace_nw)

    ic_theory = float(-2.0 * log_likelihood + 2.0 * B_hat_theory)
    ic_nw = float(-2.0 * log_likelihood + 2.0 * B_hat_nw)

    # Algebraic check for K_d=N_d(N_d-1)/2:
    # -2S + 2(K/N)tr = -2S + (N-1)tr.
    ic_theory_alt = float(-2.0 * log_likelihood + (N_d - 1) * trace_theory)
    ic_nw_alt = float(-2.0 * log_likelihood + (N_d - 1) * trace_nw)

    return {
        "log_likelihood": log_likelihood,
        "N_d": N_d,
        "K_d": K_d,
        "num_params": int(p_dim),
        "trace_JinvI_theory": trace_theory,
        "trace_JinvI_nw": trace_nw,
        "B_hat_theory": B_hat_theory,
        "B_hat_nw": B_hat_nw,
        "ic_theory": ic_theory,
        "ic_nw": ic_nw,
        "ic_theory_alt": ic_theory_alt,
        "ic_nw_alt": ic_nw_alt,
        "nw_bandwidth": int(q_n),
        "nw_centered": bool(center_nw),
        "ridge": float(ridge),
        "ridge_refit_curvature": float(ridge_refit_curvature),
        "score_norm": float(np.linalg.norm(score_rows.sum(axis=0))),
        "phi_mean_norm": float(np.linalg.norm(phi.mean(axis=0))),
    }



def besag_PMLE_lbfgs_ridge(
    theta_init: np.ndarray | None,
    X: np.ndarray,
    ridge_refit: float = 1e-5,
    maxiter: int = 5000,
    gtol: float = 1e-8,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Fit the unpenalized/ridge-stabilized Besag pseudo-logistic model.

    The optimization objective is on mean-loss scale:

        mean_{i<j} log(1 + exp(-X_ij^T theta))
        + 0.5 * ridge_refit * ||theta||^2.

    Using mean-loss scaling makes `ridge_refit` directly comparable to the
    average Hessian J_hat used in TIC diagnostics.  Therefore the TIC curvature
    should use J_hat + ridge_refit * I.
    """
    X64 = np.asarray(X, dtype=np.float64)
    p_dim = int(X64.shape[1])

    if theta_init is None:
        theta0 = np.zeros(p_dim, dtype=np.float64)
    else:
        theta0 = np.asarray(theta_init, dtype=np.float64).reshape(-1)
        if theta0.size != p_dim:
            raise ValueError(
                f"theta_init has size {theta0.size}, but X has {p_dim} columns."
            )

    if p_dim == 0:
        return np.zeros(0, dtype=np.float64), "C", {
            "optimizer": "L-BFGS-B-ridge-mean",
            "success": True,
            "message": "null model",
            "nit": 0,
            "fun": float(np.log(2.0)),
            "grad_norm": 0.0,
            "ridge_refit": float(ridge_refit),
            "theta_norm": 0.0,
            "theta_maxabs": 0.0,
        }

    def obj(theta: np.ndarray) -> float:
        eta = X64 @ theta
        loss = float(np.mean(np.logaddexp(0.0, -eta)))
        penalty = 0.5 * float(ridge_refit) * float(theta @ theta)
        return loss + penalty

    def grad(theta: np.ndarray) -> np.ndarray:
        eta = X64 @ theta
        pi = stable_sigmoid(eta)
        grad_loss = -(X64.T @ (1.0 - pi)) / X64.shape[0]
        grad_penalty = float(ridge_refit) * theta
        return grad_loss + grad_penalty

    res = minimize(
        obj,
        theta0,
        jac=grad,
        method="L-BFGS-B",
        options={
            "maxiter": int(maxiter),
            "gtol": float(gtol),
            "ftol": 1e-12,
            "maxls": 50,
        },
    )

    theta_hat = np.asarray(res.x, dtype=np.float64)
    grad_norm = float(np.linalg.norm(grad(theta_hat)))
    info: dict[str, Any] = {
        "optimizer": "L-BFGS-B-ridge-mean",
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit),
        "fun": float(res.fun),
        "grad_norm": grad_norm,
        "ridge_refit": float(ridge_refit),
        "theta_norm": float(np.linalg.norm(theta_hat)),
        "theta_maxabs": float(np.max(np.abs(theta_hat))) if theta_hat.size else 0.0,
    }
    is_converged = "C" if bool(res.success) else "F"
    return theta_hat, is_converged, info


def besag_PMLE_fista(
    raw,
    order,
    group_lasso=False,
    n_iter=1000,
    L1_=0,
    X=None,
    ic=False,
    init=None,
    ridge: float = 1e-8,
    ridge_refit: float = 1e-4,
    nw_bandwidth: int | None = None,
    center_nw: bool = True,
):
    """Fit Besag BPLE as logistic regression with y=1 and no intercept."""
    raw = np.asarray(raw)
    n, dim = raw.shape

    assert n <= 1500

    if X is None:
        print("Building X ...")
        X = build_X_torus(raw, order=order, dtype=np.float32)

    if group_lasso or float(L1_) != 0.0:
        raise NotImplementedError(
            "This bias-check script now uses L-BFGS-B ridge fitting for the "
            "fixed unpenalized model only. group_lasso/L1 paths should use the "
            "separate regularization-path script."
        )

    print("Start Fitting with L-BFGS-B ridge-stabilized mean loss ...")
    start_fit = time()
    theta_hat, is_converged, opt_info = besag_PMLE_lbfgs_ridge(
        theta_init=init,
        X=X,
        ridge_refit=ridge_refit,
        maxiter=n_iter,
        gtol=1e-8,
    )
    print(f"Optimization took {time() - start_fit:.2f} sec")
    print(
        "L-BFGS-B info:",
        f"status={is_converged}",
        f"nit={opt_info['nit']}",
        f"grad_norm={opt_info['grad_norm']:.3e}",
        f"theta_norm={opt_info['theta_norm']:.3e}",
        f"ridge_refit={ridge_refit:.3e}",
    )

    if not ic:
        return theta_hat, is_converged, L1_

    diagnostics = tic_diagnostics_from_X(
        theta=theta_hat,
        X=X,
        n=n,
        order=order,
        ridge=ridge,
        ridge_refit_curvature=ridge_refit,
        nw_bandwidth=nw_bandwidth,
        center_nw=center_nw,
    )
    diagnostics.update({
        "optimizer": opt_info["optimizer"],
        "optimizer_success": bool(opt_info["success"]),
        "optimizer_message": opt_info["message"],
        "optimizer_nit": int(opt_info["nit"]),
        "optimizer_fun": float(opt_info["fun"]),
        "optimizer_grad_norm": float(opt_info["grad_norm"]),
        "theta_norm": float(opt_info["theta_norm"]),
        "theta_maxabs": float(opt_info["theta_maxabs"]),
        "ridge_refit": float(ridge_refit),
    })
    return theta_hat, is_converged, L1_, diagnostics["ic_nw"], diagnostics


def sample_from_true(n: int, dim: int, seed: int):
    return sample_from_mininfo_markov(n, dim, seed, marginal="vonmises")


def build_inner_mc_X_bank(
    dim: int,
    order: int,
    sample_size: int,
    n_inner_mc: int,
    seed_base: int,
    dtype=np.float32,
) -> np.ndarray:
    """Precompute independent Monte Carlo design matrices once."""
    raw0, _ = sample_from_true(
        n=sample_size,
        dim=dim,
        seed=seed_base + 10_000_000,
    )
    X0 = build_X_torus(raw0, order=order, dtype=dtype)

    X_bank = np.empty((n_inner_mc, X0.shape[0], X0.shape[1]), dtype=dtype)
    X_bank[0] = X0

    for m in range(1, n_inner_mc):
        inner_seed = seed_base + 10_000_000 + m
        raw_mc, _ = sample_from_true(n=sample_size, dim=dim, seed=inner_seed)
        X_bank[m] = build_X_torus(raw_mc, order=order, dtype=dtype)

    return X_bank


def one_outer_rep_fixed_model(
    rep_id: int,
    dim: int,
    order: int,
    sample_size: int = 1000,
    n_inner_mc: int = 100,
    n_iter: int = 1000,
    seed_base: int = 12345,
    X_mc_bank: np.ndarray | None = None,
    x_dtype=np.float32,
    ridge: float = 1e-8,
    ridge_refit: float = 1e-4,
    nw_bandwidth: int | None = None,
    center_nw: bool = True,
) -> dict[str, Any]:
    """One outer replication for fixed-model optimism checking."""
    if X_mc_bank is None:
        raise ValueError("X_mc_bank must be precomputed and passed in.")

    seed = seed_base + rep_id
    raw, _ = sample_from_true(n=sample_size, dim=dim, seed=seed)
    X = build_X_torus(raw, order=order, dtype=x_dtype)
    n = raw.shape[0]

    # Fixed model: all parameters. Replace by active_mask if needed.
    active_mask = np.arange(X.shape[1])
    X_active = X[:, active_mask]

    theta_fit, is_converged_fit, _, _, res = besag_PMLE_fista(
        raw=raw,
        order=order,
        L1_=0.0,
        group_lasso=False,
        X=X_active,
        n_iter=n_iter,
        ic=True,
        init=None,
        ridge=ridge,
        ridge_refit=ridge_refit,
        nw_bandwidth=nw_bandwidth,
        center_nw=center_nw,
    )

    empirical_ll = float(res["log_likelihood"])
    B_hat_theory = float(res["B_hat_theory"])
    B_hat_nw = float(res["B_hat_nw"])

    mc_log_likelihoods = np.empty(X_mc_bank.shape[0], dtype=np.float64)
    for m in range(X_mc_bank.shape[0]):
        mc_log_likelihoods[m] = besag_log_likelihood_from_X(theta_fit, X_mc_bank[m])

    mc_expected_ll = float(np.mean(mc_log_likelihoods))
    mc_expected_ll_se = float(
        np.std(mc_log_likelihoods, ddof=1) / np.sqrt(n_inner_mc)
        if n_inner_mc > 1 else np.nan
    )

    # Replication-level MC approximation to
    # S_obs(theta_hat) - K_d M_{n,d}(theta_hat).
    B_true = empirical_ll - mc_expected_ll

    # ------------------------------------------------------------
    # Additional diagnostics:
    # Use only the first g inner MC samples to see how B_true changes
    # as the number of inner samples increases.
    # This avoids running n_inner_mc=5000 for all outer replications.
    # ------------------------------------------------------------
    inner_grid = [50, 100, 200, 500, 1000]
    inner_grid = [g for g in inner_grid if g <= X_mc_bank.shape[0]]

    extra = {}
    for g in inner_grid:
        mc_mean_g = float(np.mean(mc_log_likelihoods[:g]))
        mc_se_g = float(
            np.std(mc_log_likelihoods[:g], ddof=1) / np.sqrt(g)
            if g > 1 else np.nan
        )
        B_true_g = float(empirical_ll - mc_mean_g)

        extra[f"mc_expected_ll_inner{g}"] = mc_mean_g
        extra[f"B_true_inner{g}"] = B_true_g
        extra[f"B_true_mc_se_inner{g}"] = mc_se_g
        extra[f"gap_B_hat_theory_minus_B_true_inner{g}"] = float(B_hat_theory - B_true_g)
        extra[f"gap_B_hat_nw_minus_B_true_inner{g}"] = float(B_hat_nw - B_true_g)

    out = {
        "ok": True,
        "rep_id": rep_id,
        "n": int(n),
        "N_d": int(res["N_d"]),
        "K_d": int(res["K_d"]),
        "num_pseudo_terms": int(X.shape[0]),
        "num_params": int(X_active.shape[1]),
        "empirical_ll": empirical_ll,
        "mc_expected_ll": mc_expected_ll,
        "mc_expected_ll_se": mc_expected_ll_se,
        "B_true_mc_se": mc_expected_ll_se,
        "B_true": float(B_true),
        "B_hat_theory": B_hat_theory,
        "B_hat_nw": B_hat_nw,
        "gap_B_hat_theory_minus_B_true": float(B_hat_theory - B_true),
        "gap_B_hat_nw_minus_B_true": float(B_hat_nw - B_true),
        "ratio_B_hat_theory_to_B_true": (
            float(B_hat_theory / B_true) if B_true != 0 else float("nan")
        ),
        "ratio_B_hat_nw_to_B_true": (
            float(B_hat_nw / B_true) if B_true != 0 else float("nan")
        ),
        "trace_JinvI_theory": float(res["trace_JinvI_theory"]),
        "trace_JinvI_nw": float(res["trace_JinvI_nw"]),
        "ic_theory": float(res["ic_theory"]),
        "ic_nw": float(res["ic_nw"]),
        "nw_bandwidth": int(res["nw_bandwidth"]),
        "nw_centered": bool(res["nw_centered"]),
        "ridge": float(res["ridge"]),
        "ridge_refit": float(res.get("ridge_refit", ridge_refit)),
        "ridge_refit_curvature": float(res.get("ridge_refit_curvature", ridge_refit)),
        "optimizer": res.get("optimizer", ""),
        "optimizer_nit": int(res.get("optimizer_nit", -1)),
        "optimizer_grad_norm": float(res.get("optimizer_grad_norm", np.nan)),
        "theta_norm": float(res.get("theta_norm", np.nan)),
        "theta_maxabs": float(res.get("theta_maxabs", np.nan)),
        "score_norm": float(res["score_norm"]),
        "phi_mean_norm": float(res["phi_mean_norm"]),
        "refit_converged": is_converged_fit,
    }

    out.update(extra)
    return out


def _mean_se(df: pd.DataFrame, col: str) -> dict[str, float]:
    return {
        f"mean_{col}": float(df[col].mean()),
        f"se_{col}": (
            float(df[col].std(ddof=1) / np.sqrt(len(df)))
            if len(df) > 1 else float("nan")
        ),
        f"sd_{col}": (
            float(df[col].std(ddof=1)) if len(df) > 1 else float("nan")
        ),
    }


def monte_carlo_bias_check_parallel(
    dim: int,
    order: int,
    sample_size: int = 1000,
    n_outer_rep: int = 100,
    n_inner_mc: int = 100,
    n_iter: int = 1000,
    seed_base: int = 12345,
    n_jobs: int = 10,
    verbose: int = 10,
    save_csv_path: str | None = None,
    x_dtype: str = "float32",
    ridge: float = 1e-8,
    ridge_refit: float = 1e-4,
    nw_bandwidth: int | None = None,
    center_nw: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run fixed-model outer Monte Carlo replications in parallel."""
    dtype = np.float32 if x_dtype == "float32" else np.float64

    print("Precomputing inner Monte Carlo X bank ...")
    start_bank = time()
    X_mc_bank = build_inner_mc_X_bank(
        dim=dim,
        order=order,
        sample_size=sample_size,
        n_inner_mc=n_inner_mc,
        seed_base=seed_base,
        dtype=dtype,
    )
    print(
        f"Inner MC bank shape={X_mc_bank.shape}, "
        f"size={X_mc_bank.nbytes / 1024**3:.2f} GiB, "
        f"built in {time() - start_bank:.2f} sec"
    )

    results = Parallel(
        n_jobs=n_jobs,
        verbose=verbose,
        max_nbytes="100M",
        mmap_mode="r",
    )(
        delayed(one_outer_rep_fixed_model)(
            rep_id=r,
            dim=dim,
            order=order,
            sample_size=sample_size,
            n_inner_mc=n_inner_mc,
            n_iter=n_iter,
            seed_base=seed_base,
            X_mc_bank=X_mc_bank,
            x_dtype=dtype,
            ridge=ridge,
            ridge_refit=ridge_refit,
            nw_bandwidth=nw_bandwidth,
            center_nw=center_nw,
        )
        for r in range(n_outer_rep)
    )

    df = pd.DataFrame(results)
    df_ok = df[df["ok"]].copy()

    if len(df_ok) == 0:
        summary = {
            "n_outer_rep": n_outer_rep,
            "n_ok": 0,
            "message": "No successful replications.",
        }
        return df, summary

    summary: dict[str, Any] = {
        "n_outer_rep": int(n_outer_rep),
        "n_ok": int(len(df_ok)),
        "n_failed": int(n_outer_rep - len(df_ok)),
        "dim": int(dim),
        "order": int(order),
        "sample_size": int(sample_size),
        "n_inner_mc": int(n_inner_mc),
        "x_dtype": x_dtype,
        "ridge": float(ridge),
        "ridge_refit": float(ridge_refit),
        "nw_bandwidth_arg": nw_bandwidth,
        "center_nw": bool(center_nw),
        "mean_n": float(df_ok["n"].mean()),
        "mean_N_d": float(df_ok["N_d"].mean()),
        "mean_K_d": float(df_ok["K_d"].mean()),
        "mean_num_pseudo_terms": float(df_ok["num_pseudo_terms"].mean()),
        "mean_num_params": float(df_ok["num_params"].mean()),
        "refit_converged_rate": float(np.mean(df_ok["refit_converged"] == "C")),
    }

    for col in [
        "empirical_ll",
        "mc_expected_ll",
        "mc_expected_ll_se",
        "B_true_mc_se",
        "B_true",
        "B_hat_theory",
        "B_hat_nw",
        "gap_B_hat_theory_minus_B_true",
        "gap_B_hat_nw_minus_B_true",
        "ratio_B_hat_theory_to_B_true",
        "ratio_B_hat_nw_to_B_true",
        "trace_JinvI_theory",
        "trace_JinvI_nw",
        "ic_theory",
        "ic_nw",
        "nw_bandwidth",
        "ridge_refit",
        "ridge_refit_curvature",
        "optimizer_nit",
        "optimizer_grad_norm",
        "theta_norm",
        "theta_maxabs",
        "score_norm",
        "phi_mean_norm",
    ]:
        summary.update(_mean_se(df_ok, col))

    for g in [50, 100, 200, 500, 1000]:
        for col in [
            f"mc_expected_ll_inner{g}",
            f"B_true_inner{g}",
            f"B_true_mc_se_inner{g}",
            f"gap_B_hat_theory_minus_B_true_inner{g}",
            f"gap_B_hat_nw_minus_B_true_inner{g}",
        ]:
            if col in df_ok.columns:
                summary.update(_mean_se(df_ok, col))

    # Useful aggregate checks: unbiasedness is about means across outer reps.
    se = summary["se_gap_B_hat_theory_minus_B_true"]
    summary["mean_gap_theory_over_se"] = (
        summary["mean_gap_B_hat_theory_minus_B_true"] / se
        if np.isfinite(se) and se != 0.0
        else float("nan")
    )
    se = summary["se_gap_B_hat_nw_minus_B_true"]
    summary["mean_gap_nw_over_se"] = (
        summary["mean_gap_B_hat_nw_minus_B_true"] / se
        if np.isfinite(se) and se != 0.0
        else float("nan")
    )

    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=False)

    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Besag PLE TIC optimism check"
    )

    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--n-outer-rep", type=int, default=1000)
    parser.add_argument("--n-inner-mc", type=int, default=5000)
    parser.add_argument("--n-iter", type=int, default=5000)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--save-csv-path",
        type=str,
        default="tic_bple_bias_check-inner5000-vM-(1,-1,0,0).csv",
    )
    parser.add_argument(
        "--x-dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
    )
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument(
        "--ridge-refit",
        type=float,
        default=1e-5,
        help=(
            "Ridge coefficient for L-BFGS-B mean-loss fitting. "
            "Use 0.0 for unregularized L-BFGS-B, but this may separate."
        ),
    )
    parser.add_argument(
        "--nw-bandwidth",
        type=int,
        default=None,
        help="Newey--West bandwidth. If omitted, uses floor(4*(N_d/100)^(2/9)).",
    )
    parser.add_argument(
        "--no-center-nw",
        action="store_true",
        help="Do not center phi_i before the Newey--West covariance calculation.",
    )

    args = parser.parse_args()

    _, summary = monte_carlo_bias_check_parallel(
        dim=args.dim,
        order=args.order,
        sample_size=args.sample_size,
        n_outer_rep=args.n_outer_rep,
        n_inner_mc=args.n_inner_mc,
        n_iter=args.n_iter,
        seed_base=args.seed_base,
        n_jobs=args.n_jobs,
        verbose=10,
        save_csv_path=args.save_csv_path,
        x_dtype=args.x_dtype,
        ridge=args.ridge,
        ridge_refit=args.ridge_refit,
        nw_bandwidth=args.nw_bandwidth,
        center_nw=not args.no_center_nw,
    )

    print(summary)


if __name__ == "__main__":
    main()
