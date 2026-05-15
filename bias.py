"""Monte Carlo bias check for Besag PLIC on a fixed model.

It checks the fixed-model sandwich correction returned by besag_PMLE_fista(ic=True).

You must ensure sample_from_true() generates data from the target true model q.
"""

from __future__ import annotations

import argparse
import random
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from run_PPC import besag_PMLE_fista, build_X_torus, besag_log_likelihood_from_X


def sample_from_true(n: int, dim: int, seed: int) -> np.ndarray:
    return 

def one_outer_rep_fixed_model(
    rep_id: int,
    dim: int,
    order: int,
    sample_size: int = 1000,
    n_inner_mc: int = 100,
    n_iter: int = 1000,
    seed_base: int = 12345,
) -> dict[str, Any]:
    """One outer replication for fixed-model bias checking."""
    seed = seed_base + rep_id

    raw = sample_from_true(n=sample_size, dim=dim, seed=seed)
    X = build_X_torus(raw, order=order, dtype=np.float64)

    n_time = raw.shape[0]
    n_pairs = X.shape[0]

    # Fixed model: all parameters. Replace this by a true_mask if desired.
    active_mask = np.arange(X.shape[1])
    X_active = X[:, active_mask]

    theta_refit, is_converged_refit, _, plic, res = besag_PMLE_fista(
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

    empirical_risk_pair = -empirical_ll / n_pairs
    empirical_risk_time = -empirical_ll / n_time

    mc_risks_pair: list[float] = []
    mc_risks_time: list[float] = []

    for m in range(n_inner_mc):
        inner_seed = seed_base + 10_000_000 + rep_id * n_inner_mc + m
        raw_mc = sample_from_true(n=sample_size, dim=dim, seed=inner_seed)
        X_mc = build_X_torus(raw_mc, order=order, dtype=np.float64)
        X_mc_active = X_mc[:, active_mask]

        ll_mc = besag_log_likelihood_from_X(theta_refit, X_mc_active)
        mc_risks_pair.append(-ll_mc / X_mc_active.shape[0])
        mc_risks_time.append(-ll_mc / raw_mc.shape[0])

    mc_expected_risk_pair = float(np.mean(mc_risks_pair))
    mc_expected_risk_time = float(np.mean(mc_risks_time))
    mc_expected_risk_pair_se = float(np.std(mc_risks_pair, ddof=1) / np.sqrt(n_inner_mc))
    mc_expected_risk_time_se = float(np.std(mc_risks_time, ddof=1) / np.sqrt(n_inner_mc))

    # Four scale conventions, to diagnose which one matches the current theory.
    B_true_pair_pair = n_pairs * (mc_expected_risk_pair - empirical_risk_pair)
    B_true_time_pair = n_time * (mc_expected_risk_pair - empirical_risk_pair)
    B_true_pair_time = n_pairs * (mc_expected_risk_time - empirical_risk_time)
    B_true_time_time = n_time * (mc_expected_risk_time - empirical_risk_time)

    return {
        "ok": True,
        "rep_id": rep_id,
        "n_time": n_time,
        "n_pairs": n_pairs,
        "num_params": int(X_active.shape[1]),
        "empirical_ll": empirical_ll,
        "empirical_risk_pair": empirical_risk_pair,
        "empirical_risk_time": empirical_risk_time,
        "mc_expected_risk_pair": mc_expected_risk_pair,
        "mc_expected_risk_time": mc_expected_risk_time,
        "mc_expected_risk_pair_se": mc_expected_risk_pair_se,
        "mc_expected_risk_time_se": mc_expected_risk_time_se,
        "B_hat_plus": B_hat,
        "B_hat_minus": -B_hat,
        "B_true_pair_pair": B_true_pair_pair,
        "B_true_time_pair": B_true_time_pair,
        "B_true_pair_time": B_true_pair_time,
        "B_true_time_time": B_true_time_time,
        "diff_pair_pair_plus": B_hat - B_true_pair_pair,
        "diff_time_pair_plus": B_hat - B_true_time_pair,
        "diff_pair_time_plus": B_hat - B_true_pair_time,
        "diff_time_time_plus": B_hat - B_true_time_time,
        "refit_converged": is_converged_refit,
        "plic": plic,
    }


def _mean_se(df: pd.DataFrame, col: str) -> dict[str, float]:
    return {
        f"mean_{col}": float(df[col].mean()),
        f"se_{col}": float(df[col].std(ddof=1) / np.sqrt(len(df))) if len(df) > 1 else float("nan"),
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
    """Run fixed-model outer Monte Carlo replications in parallel."""
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
        "mean_n_time": float(df_ok["n_time"].mean()),
        "mean_n_pairs": float(df_ok["n_pairs"].mean()),
        "mean_num_params": float(df_ok["num_params"].mean()),
        "refit_converged_rate": float(np.mean(df_ok["refit_converged"] == "C")),
    }

    for col in [
        "B_hat_plus",
        "B_hat_minus",
        "B_true_pair_pair",
        "B_true_time_pair",
        "B_true_pair_time",
        "B_true_time_time",
        "diff_pair_pair_plus",
        "diff_time_pair_plus",
        "diff_pair_time_plus",
        "diff_time_time_plus",
        "mc_expected_risk_pair_se",
        "mc_expected_risk_time_se",
        "plic",
    ]:
        summary.update(_mean_se(df_ok, col))

    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=False)

    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-model Besag PLIC bias check")
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
