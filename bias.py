"""
Monte Carlo bias check for Besag PLIC on a fixed model.

This version uses only the time-scale normalization.

It checks the fixed-model sandwich correction returned by besag_PMLE_fista(ic=True).

You must ensure sample_from_true() generates data from the target true model q.
"""

from __future__ import annotations

import argparse
from typing import Any
from time import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sampling import sample_from_mininfo_markov
from itertools import combinations
from tqdm import tqdm

def torus_pair_feature(x_now, x_lag):
    """
    x_now, x_lag: shape (dim,)

    並び:
    (i=1,j=1): cc, cs, sc, ss
    (i=1,j=2): cc, cs, sc, ss
    ...
    (i=2,j=1): cc, cs, sc, ss
    ...

    つまり同じ (i,j) の4パラメタが連続。
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
                c_now[i] * c_lag[j],  # cc
                c_now[i] * s_lag[j],  # cs
                s_now[i] * c_lag[j],  # sc
                s_now[i] * s_lag[j],  # ss
            ]
            k += 4

    return out


# ============================================================
# Swap delta
# ============================================================

def swap_delta_torus(raw, p, q, order):
    """
    raw[p] と raw[q] を swap したときの

        h(original) - h(swapped)

    を差分で計算する。

    raw: shape (n, dim)
    p, q: 0-indexed
    order: lag order
    """
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
            after += torus_pair_feature(
                after_value(t),
                after_value(t - lag),
            )

        start = (lag - 1) * per_lag
        end = lag * per_lag
        delta[start:end] = before - after

    return delta

def build_X_torus(raw, order, dtype=np.float32):
    """
    Besag PLE 用の design matrix X を構築。

    各行:
        h(original) - h(swapped)

    特徴量の並び:
        lag=1 の全特徴量
        lag=2 の全特徴量
        ...
        lag=order の全特徴量

    各 lag 内では:
        (i,j): cc, cs, sc, ss
    """
    raw = np.asarray(raw)
    n, dim = raw.shape

    n_pairs = (n - 2 * order) * (n - 2 * order - 1) // 2
    n_features = order * 4 * dim * dim

    X = np.empty((n_pairs, n_features), dtype=dtype)
    pairs = combinations(range(order, n - order), 2)

    for row, (p, q) in enumerate(tqdm(pairs, total=n_pairs)):
        X[row] = swap_delta_torus(raw, p, q, order)

    return X

def besag_log_likelihood_from_X(theta: np.ndarray, X: np.ndarray) -> float:
    theta = np.asarray(theta).reshape(-1)
    eta = X @ theta
    return float(-np.sum(np.logaddexp(0.0, -eta)))

def make_groups(dim, order, design_matrix):
    # n_features = order * dim * dim * 4
    n_features = design_matrix.shape[1]
    return np.arange(n_features).reshape(-1, 4*order)


def besag_PMLE_fista(raw, order, group_lasso=False, n_iter=1000, L1_=0, X=None, ic=False, init=None):
    """
    Besag PMLE を logistic regression として解く FISTA 版。

    論文の形:
        X = h(original) - h(swapped)
        y = 1
        fit_intercept = False

    ただし L1_ > 0 や group_lasso=True の場合は regularized 版。
    """
    raw = np.asarray(raw)
    n, dim = raw.shape

    assert n <= 1500

    if X is None:
        print("Building X ...")
        X = build_X_torus(raw, order=order, dtype=np.float32)

    n_pairs, n_features = X.shape
    y = np.ones(n_pairs, dtype=np.float32)

    from fista import LogisticRegressionFISTA
    clf = LogisticRegressionFISTA(
        eta=None,
        n_iter=n_iter,
        tol=1e-6,
        l1=L1_,
        fit_intercept=False,
        line_search=True,
        verbose=True,
        init_w=init
    )

    if group_lasso:
        clf.groups = make_groups(dim, order, X)
        print("Groups shape:", clf.groups.shape)

    print("Start Fitting with FISTA ...")
    start_fit = time()
    clf.fit(X, y, is_group=group_lasso)
    is_converged = "C" if clf.converged_ else "F"
    end_fit = time()
    print(f"Optimization took {end_fit - start_fit:.2f} sec")

    if ic:
        w = clf.w

        # ------------------------------------------------------------
        # Unpenalized Besag log-likelihood
        # ell_B(theta) = - sum_i log(1 + exp(- x_i^T theta))
        # ------------------------------------------------------------
        eta = X @ w

        # stable computation of log(1 + exp(-eta))
        loss_terms = np.logaddexp(0.0, -eta)
        log_likelihood = -np.sum(loss_terms)

        # ------------------------------------------------------------
        # J estimator:
        # J = (1/n) X^T W X
        # where W_i = sigma(eta_i) sigma(-eta_i)
        # ------------------------------------------------------------
        prob = 1.0 / (1.0 + np.exp(-eta))      # sigma(eta)
        weight = prob * (1.0 - prob)           # sigma(eta) sigma(-eta)

        J = (X.T @ (weight[:, None] * X)) / n

        # ------------------------------------------------------------
        # I estimator: Newey--West estimator for the aggregated score
        #
        # pairwise score:
        # psi_{s,t}(theta)
        # = - sigma(-theta^T X_{s,t}) X_{s,t}
        #
        # aggregate by second index t:
        # g_t(theta) = sum_{s<t} psi_{s,t}(theta)
        # ------------------------------------------------------------
        score_rows = -((1.0 - prob)[:, None] * X)

        pairs = np.array(
            [(s-1, t-1) for s in range(order + 1, n - order)
                    for t in range(s + 1, n - order+1)],
            dtype=np.int64,
        )

        if len(pairs) != n_pairs:
            raise ValueError(
                "The number of constructed pairs does not match X.shape[0]. "
                "Please modify the pair-index construction to match build_X_torus."
            )

        second_index = pairs[:, 1]

        g = np.zeros((n, n_features), dtype=np.float64)
        np.add.at(g, second_index, score_rows)

        # Newey--West bandwidth.
        q_n = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        q_n = max(1, min(q_n, n - 1))

        # Center the score process
        valid_t = np.arange(order + 1, n - order)

        g_bar = g[valid_t].sum(axis=0) / n
        g_centered = g - g_bar

        Gamma0 = (g_centered[valid_t].T @ g_centered[valid_t]) / n
        I = Gamma0.copy()

        for h in range(1, q_n + 1):
            t_idx = np.arange(order + 1 + h, n - order)

            Gamma_h = (
                g_centered[t_idx].T
                @ g_centered[t_idx - h]
            ) / n

            bartlett_weight = 1.0 - h / (q_n + 1.0)
            I += bartlett_weight * (Gamma_h + Gamma_h.T)

        # ------------------------------------------------------------
        # Sandwich penalty
        # ------------------------------------------------------------
        ridge = 1e-8
        J_reg = J + ridge * np.eye(n_features) #singular回避
        penalty = np.trace(np.linalg.solve(J_reg, I))
        plic = -2.0 * log_likelihood + 2.0 * penalty
        return clf.w, is_converged, L1_, plic, (log_likelihood,penalty)
    else:
        return clf.w, is_converged, L1_
    

def sample_from_true(n: int, dim: int, seed: int) -> np.ndarray:
    return sample_from_mininfo_markov(n, dim)


def one_outer_rep_fixed_model(
    rep_id: int,
    dim: int,
    order: int,
    sample_size: int = 1000,
    n_inner_mc: int = 100,
    n_iter: int = 1000,
    seed_base: int = 12345,
) -> dict[str, Any]:
    """One outer replication for fixed-model bias checking.

    All risks and bias quantities are normalized by the original time length n.
    """
    seed = seed_base + rep_id

    raw, _ = sample_from_true(n=sample_size, dim=dim, seed=seed)
    X = build_X_torus(raw, order=order, dtype=np.float64)

    n = raw.shape[0]

    # Fixed model: all parameters. Replace this by a true_mask if desired.
    active_mask = np.arange(X.shape[1])
    X_active = X[:, active_mask]

    theta_fit, is_converged_fit, _, plic, res = besag_PMLE_fista(
        raw=raw,
        order=order,
        L1_=0.0,
        group_lasso=False,
        X=X_active,
        n_iter=n_iter,
        ic=True,
        init=None,
    )

    empirical_ll, B_hat = res

    # Time-scale empirical risk.
    empirical_risk = -empirical_ll / n

    mc_risks: list[float] = []

    for m in range(n_inner_mc):
        inner_seed = seed_base + 10_000_000 + rep_id * n_inner_mc + m

        raw_mc, _ = sample_from_true(n=sample_size, dim=dim, seed=inner_seed)
        X_mc = build_X_torus(raw_mc, order=order, dtype=np.float64)
        X_mc_active = X_mc[:, active_mask]

        ll_mc = besag_log_likelihood_from_X(theta_fit, X_mc_active)

        # Time-scale Monte Carlo risk.
        mc_risks.append(-ll_mc / raw_mc.shape[0])

    mc_expected_risk = float(np.mean(mc_risks))
    mc_expected_risk_se = float(np.std(mc_risks, ddof=1) / np.sqrt(n_inner_mc))

    # Time-scale target bias:
    #
    #     B_true = n * { E_q[-ell(theta_hat; Z) / n] - [-ell(theta_hat; z) / n] }
    #
    # where n is the original time length.
    B_true = n * (mc_expected_risk - empirical_risk)

    return {
        "ok": True,
        "rep_id": rep_id,
        "n": int(n),
        "num_pseudo_terms": int(X.shape[0]),
        "num_params": int(X_active.shape[1]),
        "empirical_ll": float(empirical_ll),
        "empirical_risk": float(empirical_risk),
        "mc_expected_risk": mc_expected_risk,
        "mc_expected_risk_se": mc_expected_risk_se,
        "B_hat_plus": float(B_hat),
        "B_true": float(B_true),
        "refit_converged": is_converged_fit,
        "plic": float(plic),
    }


def _mean_se(df: pd.DataFrame, col: str) -> dict[str, float]:
    return {
        f"mean_{col}": float(df[col].mean()),
        f"se_{col}": float(df[col].std(ddof=1) / np.sqrt(len(df)))
        if len(df) > 1
        else float("nan"),
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run fixed-model outer Monte Carlo replications in parallel.

    This version reports only time-scale risks and time-scale bias estimates.
    """
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(one_outer_rep_fixed_model)(
            rep_id=r,
            dim=dim,
            order=order,
            sample_size=sample_size,
            n_inner_mc=n_inner_mc,
            n_iter=n_iter,
            seed_base=seed_base,
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
        "n_outer_rep": n_outer_rep,
        "n_ok": int(len(df_ok)),
        "n_failed": int(n_outer_rep - len(df_ok)),
        "dim": dim,
        "order": order,
        "sample_size": sample_size,
        "n_inner_mc": n_inner_mc,
        "mean_n": float(df_ok["n"].mean()),
        "mean_num_pseudo_terms": float(df_ok["num_pseudo_terms"].mean()),
        "mean_num_params": float(df_ok["num_params"].mean()),
        "refit_converged_rate": float(np.mean(df_ok["refit_converged"] == "C")),
    }

    for col in [
        "empirical_risk",
        "mc_expected_risk",
        "mc_expected_risk_se",
        "B_hat_plus",
        "B_true",
        "plic",
    ]:
        summary.update(_mean_se(df_ok, col))

    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=False)

    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-model Besag PLIC bias check using time-scale normalization"
    )

    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--n-outer-rep", type=int, default=200)
    parser.add_argument("--n-inner-mc", type=int, default=100)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--save-csv-path", type=str, default="bias_check.csv")

    args = parser.parse_args()

    df_bias, summary = monte_carlo_bias_check_parallel(
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
    )

    print(summary)


if __name__ == "__main__":
    main()