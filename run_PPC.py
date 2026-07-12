from __future__ import annotations

import argparse
from itertools import combinations
from time import time
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm

from data_sim import Kuramoto_Model, generate_5d_phase_timeseries_data


def sample_plot(data, save_dir):
    """Input: array with shape (steps, dim)."""
    df = pd.DataFrame(data)
    df.plot(figsize=(15, 5))
    plt.legend().remove()
    plt.savefig(save_dir+"/"+"sample_plot.png")


def MLE(Y, order):
    from statsmodels.tsa.api import VAR, ARIMA

    if Y.shape[1] == 1:
        model = ARIMA(Y, order=(order, 0, 0), trend="n")  # AR(d)
        results = model.fit()
    else:
        model = VAR(Y)
        results = model.fit(maxlags=order, ic=None, trend="n")

    print(results.summary())
    return results

def ecog_case1():
    from data_real import load_marmoset_ecog, extract_feature_matrix, FeatureSpec
    dataset = load_marmoset_ecog(animal="Ji", session_index=1, window=slice(0, 1500))
    gamma_phase = extract_feature_matrix(
        dataset,
        FeatureSpec(name="gamma_phase", feature="phase", band=(25, 40)),
        trials=[0],
    )
    return gamma_phase

def ecog_case2(mode="pre"):
    from data_real import (
        load_marmoset_ecog_epoched,
        split_marmoset_pre_post_1500ms,
        extract_feature_matrix,
        FeatureSpec,
        NeuralDataset,
    )

    # ------------------------------------------------------------
    # 1. Event.mat に基づいて epoch 化
    # ds_epoch.data shape:
    #     (n_epochs, n_channels, 3000)
    # ------------------------------------------------------------
    ds_epoch = load_marmoset_ecog_epoched(
        animal="Ji",
        session_index=1,
        epoch_window=(-1.5, 1.5),
        samplerate=1000.0,
        event_key="cntEvent",
        event_sample_column=5,
        event_time_unit="samples_matlab",
    )

    # ------------------------------------------------------------
    # 2. pre/post に分割
    # pre  shape: (n_epochs, n_channels, 1500)
    # post shape: (n_epochs, n_channels, 1500)
    # ------------------------------------------------------------
    pre, post = split_marmoset_pre_post_1500ms(
        ds_epoch.data,
        samplerate=ds_epoch.samplerate,
    )
    if mode == "post":
        raw = post
    else:
        raw = pre
    # ------------------------------------------------------------
    # 3. pre を NeuralDataset として包み直す
    # extract_feature_matrix は NeuralDataset を要求するため
    # ------------------------------------------------------------
    epoched_dataset = NeuralDataset(
        name=ds_epoch.name + "_epoched",
        data=raw,
        channel_names=ds_epoch.channel_names,
        samplerate=ds_epoch.samplerate,
        mne_epochs=None,
    )

    # ------------------------------------------------------------
    # 4. phase feature を抽出
    #
    # 出力 shape:
    #     (1500, n_channels)
    # ------------------------------------------------------------
    phase = extract_feature_matrix(
        dataset=epoched_dataset,
        spec=FeatureSpec(
            name="phase",
            feature="phase",
            band=(12, 25), #(8,15)/(12,25)/(25,40)
        ),
        trials=[0],
    )

    return phase

def ecog_case3():
    return 



def run(dim, order, method, ridge_refit: float = 1e-3, refit_maxiter: int = 5000, refit_gtol: float = 1e-8, save_dir: str = "./outputs"):
    # raw, _ = Kuramoto_Model(N=dim, directed_K=True, base_k=0.1)
    # raw = generate_5d_phase_timeseries_data(
    #     n_steps=1500,
    #     graph=[(0, 1), (0, 2), (0, 3), (0, 4)],
    # )

    selected_20_with_pfc = [7, 8, 9, 19, 20, 27, 28, 29, 35, 36, 37,
                        1, 2, 5, 6, 16, 21,
                        55, 61, 63]
    raw = ecog_case2(mode="post")[:, [x-1 for x in selected_20_with_pfc]] #electrodes selection.
    # import pdb; pdb.set_trace()
    print("raw.shape = ", raw.shape)
    
    # sample_plot(raw, save_dir)
    # import pdb; pdb.set_trace()

    # ============================================================
    # Feature construction
    # ============================================================
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

        dim_local = x_now.shape[0]
        out = np.empty(4 * dim_local * dim_local, dtype=np.float64)

        k = 0
        for i in range(dim_local):
            for j in range(dim_local):
                out[k:k + 4] = [
                    c_now[i] * c_lag[j],
                    c_now[i] * s_lag[j],
                    s_now[i] * c_lag[j],
                    s_now[i] * s_lag[j],
                ]
                k += 4

        return out

    def valid_positions(n: int, order: int) -> np.ndarray:
        """0-indexed positions corresponding to I_d={i: d<i<n-d} in 1-indexing."""
        pos = np.arange(order, n - order - 1, dtype=np.int64)
        if pos.size < 2:
            raise ValueError(
                f"Need at least two valid positions, but got N_d={pos.size}. "
                f"Increase n or decrease order."
            )
        return pos

    def build_pair_index_arrays(n: int, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return row-aligned raw positions and local endpoint indices."""
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
        n, dim_local = raw.shape
        per_lag = 4 * dim_local * dim_local

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
        """Build the row matrix X_ij = h(original) - h(swapped)."""
        raw = np.asarray(raw)
        n, dim_local = raw.shape
        _, raw_pairs, _ = build_pair_index_arrays(n, order)

        n_pairs = raw_pairs.shape[0]
        n_features = order * 4 * dim_local * dim_local
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

    def refit_logistic_lbfgs_ridge(
        X_sub: np.ndarray,
        init: np.ndarray | None = None,
        ridge_refit: float = 1e-3,
        maxiter: int = 5000,
        gtol: float = 1e-8,
        ftol: float = 1e-12,
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        """
        Ridge-stabilized refit for fixed support using L-BFGS-B.

        This minimizes the mean-loss scaled objective

            mean_r log(1 + exp(-X_r^T theta))
            + 0.5 * ridge_refit * ||theta||^2.

        The mean-loss scaling makes ridge_refit independent of the number of
        pseudo-pairs K_d.  For TIC diagnostics, use the corresponding curvature
        adjustment J_lambda = J_hat + ridge_refit * I.
        """
        X64 = np.asarray(X_sub, dtype=np.float64)
        n_rows = int(X64.shape[0])
        p_dim = int(X64.shape[1])

        if p_dim == 0:
            return np.zeros(0, dtype=np.float64), "C", {
                "success": True,
                "message": "null model",
                "nit": 0,
                "fun": float(np.log(2.0)),
                "grad_norm": 0.0,
                "ridge_refit": float(ridge_refit),
                "objective_scale": "mean_loss",
            }

        if init is None:
            theta0 = np.zeros(p_dim, dtype=np.float64)
        else:
            theta0 = np.asarray(init, dtype=np.float64).reshape(-1)
            if theta0.size != p_dim:
                theta0 = np.zeros(p_dim, dtype=np.float64)

        def obj(theta: np.ndarray) -> float:
            eta = X64 @ theta
            mean_loss = float(np.mean(np.logaddexp(0.0, -eta)))
            penalty = 0.5 * float(ridge_refit) * float(theta @ theta)
            return mean_loss + penalty

        def grad(theta: np.ndarray) -> np.ndarray:
            eta = X64 @ theta
            pi = stable_sigmoid(eta)
            grad_loss = -(X64.T @ (1.0 - pi)) / n_rows
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
                "ftol": float(ftol),
                "maxls": 50,
            },
        )

        theta = np.asarray(res.x, dtype=np.float64)
        grad_norm = float(np.linalg.norm(grad(theta)))
        info = {
            "success": bool(res.success),
            "message": str(res.message),
            "nit": int(res.nit),
            "fun": float(res.fun),
            "grad_norm": grad_norm,
            "ridge_refit": float(ridge_refit),
            "objective_scale": "mean_loss",
        }
        is_converged = "C" if res.success else "F"
        return theta, is_converged, info

    # ============================================================
    # Group lasso groups
    # ============================================================
    def make_edge_lag_groups(dim: int, order: int) -> np.ndarray:
        """
        One group corresponds to one directed edge (i, j).
        For each edge, collect its 4 torus features across all lags.

        Feature layout assumed by build_X_torus:
            lag 1 block: dim*dim edges, each with 4 features
            lag 2 block: dim*dim edges, each with 4 features
            ...
        """
        groups = []
        per_lag = 4 * dim * dim

        for i in range(dim):
            for j in range(dim):
                edge_id = i * dim + j
                idx = []
                for lag in range(order):
                    start = lag * per_lag + edge_id * 4
                    idx.extend(range(start, start + 4))
                groups.append(idx)

        return np.asarray(groups, dtype=np.int64)

    def get_nonzero_mask_from_groups(
        theta_hat: np.ndarray,
        groups: np.ndarray,
        tol: float = 1e-8,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Extract active feature indices using the exact group definition used by group lasso."""
        theta_hat = np.asarray(theta_hat)
        keep_features: list[int] = []
        keep_groups: list[int] = []

        for g, idx in enumerate(groups):
            block = theta_hat[idx]
            if np.linalg.norm(block) > tol:
                keep_features.extend(idx.tolist())
                keep_groups.append(g)

        return tuple(keep_features), tuple(keep_groups)

    # ============================================================
    # Main: Besag PMLE with FISTA
    # ============================================================
    def tic_diagnostics_from_X(
        theta: np.ndarray,
        X: np.ndarray,
        n: int,
        order: int,
        ridge: float = 1e-8,
        nw_bandwidth: int | None = None,
        center_nw: bool = True,
    ) -> dict[str, Any]:
        """Compute TIC diagnostics with both iid-type and Newey-West/HAC I-hat."""
        theta = np.asarray(theta).reshape(-1)
        X64 = np.asarray(X, dtype=np.float64)

        pos, _, local_pairs = build_pair_index_arrays(n, order)
        N_d = int(pos.size)
        K_d = int(local_pairs.shape[0])
        p_dim = int(X64.shape[1])

        if X64.shape[0] != K_d:
            raise ValueError(
                f"X has {X64.shape[0]} rows, but K_d={K_d} from strict I_d. "
                "Check build_X_torus() and pair indexing."
            )

        # Null model / empty support.
        if p_dim == 0:
            log_likelihood = float(-K_d * np.log(2.0))
            ic = float(-2.0 * log_likelihood)
            return {
                "log_likelihood": log_likelihood,
                "N_d": N_d,
                "K_d": K_d,
                "num_params": 0,
                "trace_JinvI_theory": 0.0,
                "trace_JinvI_nw": 0.0,
                "B_hat_theory": 0.0,
                "B_hat_nw": 0.0,
                "ic_theory": ic,
                "ic_nw": ic,
                "ic_theory_alt": ic,
                "ic_nw_alt": ic,
                "nw_bandwidth": 0,
                "nw_centered": bool(center_nw),
                "ridge": float(ridge),
                "score_norm": 0.0,
                "phi_mean_norm": 0.0,
                "eigval_min_J": np.nan,
                "eigval_max_J": np.nan,
                "cond_J": np.nan,
            }

        eta = X64 @ theta
        pi = stable_sigmoid(eta)
        one_minus_pi = 1.0 - pi
        weight = pi * one_minus_pi

        log_likelihood = float(-np.sum(np.logaddexp(0.0, -eta)))

        # J_hat = - K_d^{-1} sum H_ij = K_d^{-1} X^T W X.
        J_hat = (X64.T @ (weight[:, None] * X64)) / K_d

        eigvals_J = np.linalg.eigvalsh(J_hat) if J_hat.size > 0 else np.array([])
        eigval_min_J = float(np.min(eigvals_J)) if eigvals_J.size > 0 else np.nan
        eigval_max_J = float(np.max(eigvals_J)) if eigvals_J.size > 0 else np.nan
        cond_J = (
            float(eigval_max_J / eigval_min_J)
            if eigvals_J.size > 0 and eigval_min_J > 0
            else float("inf")
        )

        # Pairwise score s_ij(theta) = (1-pi_ij(theta)) X_ij.
        score_rows = one_minus_pi[:, None] * X64

        # Endpoint aggregation for phi_i.
        endpoint_score_sum = np.zeros((N_d, p_dim), dtype=np.float64)
        np.add.at(endpoint_score_sum, local_pairs[:, 0], score_rows)
        np.add.at(endpoint_score_sum, local_pairs[:, 1], score_rows)

        phi = endpoint_score_sum / (N_d - 1)
        I_hat_theory = 4.0 * (phi.T @ phi) / N_d

        # Newey-West/HAC variant.
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

        J_reg = J_hat + (float(ridge)) * np.eye(p_dim)
        trace_theory = float(np.trace(np.linalg.solve(J_reg, I_hat_theory)))
        trace_nw = float(np.trace(np.linalg.solve(J_reg, I_hat_nw)))

        B_hat_theory = float((K_d / N_d) * trace_theory)
        B_hat_nw = float((K_d / N_d) * trace_nw)

        ic_theory = float(-2.0 * log_likelihood + 2.0 * B_hat_theory)
        ic_nw = float(-2.0 * log_likelihood + 2.0 * B_hat_nw)

        # Algebraically equivalent forms because K_d=N_d(N_d-1)/2.
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
            "score_norm": float(np.linalg.norm(score_rows.sum(axis=0))),
            "phi_mean_norm": float(np.linalg.norm(phi.mean(axis=0))),
            "eigval_min_J": eigval_min_J,
            "eigval_max_J": eigval_max_J,
            "cond_J": cond_J,
        }

    def besag_PMLE_fista(
        raw,
        order,
        group_lasso=False,
        n_iter=5000,
        L1_=0,
        X=None,
        ic=False,
        init=None,
        ridge: float = 1e-8,
        nw_bandwidth: int | None = None,
        center_nw: bool = True,
        groups: np.ndarray | None = None,
    ):
        """Fit Besag BPLE as logistic regression with y=1 and no intercept."""
        raw = np.asarray(raw)
        n, dim_local = raw.shape

        assert n <= 1500

        if X is None:
            print("Building X ...")
            X = build_X_torus(raw, order=order, dtype=np.float32)

        n_pairs, _ = X.shape
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
            init_w=init,
        )

        if group_lasso:
            if groups is None:
                groups = make_edge_lag_groups(dim_local, order)
            clf.groups = np.asarray(groups, dtype=np.int64)
            print("Groups shape:", clf.groups.shape)

        print("Start Fitting with FISTA ...")
        start_fit = time()
        clf.fit(X, y, is_group=group_lasso)
        is_converged = "C" if clf.converged_ else "F"
        print(f"Optimization took {time() - start_fit:.2f} sec")

        if not ic:
            return clf.w, is_converged, L1_

        diagnostics = tic_diagnostics_from_X(
            theta=clf.w,
            X=X,
            n=n,
            order=order,
            ridge=ridge,
            nw_bandwidth=nw_bandwidth,
            center_nw=center_nw,
        )
        return clf.w, is_converged, L1_, diagnostics["ic_nw"], diagnostics

    def print_result(theta_hat, save_dir, npy_name):
        np.save(save_dir+"/"+npy_name, theta_hat)
        print("#Non zero entries:", np.count_nonzero(theta_hat))

    if method == "mle":
        start_time = time()
        res_mle = MLE(raw, order=order)
        theta_hat = res_mle.params.T @ np.linalg.inv(res_mle.sigma_u)  # VAR(1) case
        theta_hat = theta_hat.flatten()
        end_time = time()

    elif "pmle" in method:
        start_time = time()

        if method == "pmle_grouplasso":
            print("Building X...")
            X = build_X_torus(raw, order=order, dtype=np.float64)
            print("Built X!")

            groups = make_edge_lag_groups(dim, order)
            tol = 1e-8

            def fit_lasso_path(i, lam):
                theta_hat, is_converged, L1_ = besag_PMLE_fista(
                    raw=raw,
                    order=order,
                    L1_=lam,
                    group_lasso=True,
                    X=X,
                    init=None,
                    groups=groups,
                )

                nonzero_mask, support_groups = get_nonzero_mask_from_groups(
                    theta_hat,
                    groups=groups,
                    tol=tol,
                )

                return {
                    "i": i,
                    "lambda": lam,
                    "L1_": L1_,
                    "theta_hat_lasso": theta_hat,
                    "is_converged": is_converged,
                    "nonzero_mask": nonzero_mask,
                    "support_groups": support_groups,
                    "support_string": ",".join(map(str, support_groups)),
                    "num_edge": len(support_groups),
                    "theta_norm_lasso": float(np.linalg.norm(theta_hat)),
                    "theta_maxabs_lasso": (
                        float(np.max(np.abs(theta_hat))) if theta_hat.size > 0 else 0.0
                    ),
                }

            def unique_by_support(candidates):
                unique = {}
                for cand in candidates:
                    key = cand["support_groups"]
                    if key not in unique:
                        unique[key] = cand
                    else:
                        # Prefer the less-regularized estimate as refit initializer.
                        if cand["lambda"] < unique[key]["lambda"]:
                            unique[key] = cand
                return list(unique.values())

            def refit_and_score(candidate, j):
                nonzero_mask = list(candidate["nonzero_mask"])
                num_edge = candidate["num_edge"]

                # Null model special case.
                if len(nonzero_mask) == 0:
                    _, _, local_pairs = build_pair_index_arrays(raw.shape[0], order)
                    K_d = int(local_pairs.shape[0])
                    log_likelihood = float(-K_d * np.log(2.0))
                    minus2S = float(-2.0 * log_likelihood)
                    pen_nw = 0.0
                    plic = minus2S

                    theta_refit = np.zeros(0, dtype=np.float64)
                    is_converged_refit = "C"

                    base_name = (
                        f"{j:02d}_"
                        f"l1={candidate['L1_']:.10f}_"
                        f"edge=0_"
                        f"PLIC={plic:.2f}_"
                    )
                    lasso_npy_name = f"{base_name}_lasso_{candidate['is_converged']}.npy"
                    refit_npy_name = f"{base_name}_refit_{is_converged_refit}.npy"

                    print_result(candidate["theta_hat_lasso"], save_dir, lasso_npy_name)
                    print_result(theta_refit, save_dir, refit_npy_name)

                    return {
                        **candidate,
                        "theta_refit": theta_refit,
                        "is_converged_refit": is_converged_refit,
                        "plic": plic,
                        "minus2S": minus2S,
                        "pen_nw": pen_nw,
                        "B_hat_nw": 0.0,
                        "trace_JinvI_nw": 0.0,
                        "log_likelihood_refit": log_likelihood,
                        "theta_norm_refit": 0.0,
                        "theta_maxabs_refit": 0.0,
                        "eigval_min_J": np.nan,
                        "eigval_max_J": np.nan,
                        "cond_J": np.nan,
                        "nw_bandwidth": np.nan,
                        "refit_optimizer": "null",
                        "ridge_refit": float(ridge_refit),
                        "refit_nit": 0,
                        "refit_grad_norm": 0.0,
                        "refit_message": "null model",
                        "lasso_npy_name": lasso_npy_name,
                        "refit_npy_name": refit_npy_name,
                    }

                X_sub = X[:, nonzero_mask]

                # Ridge-stabilized refit with L-BFGS-B.
                # We use zero initialization by default so that each support model
                # is refit independently of other models and independent of the
                # parallel lasso path computation.
                theta_refit, is_converged_refit, refit_info = refit_logistic_lbfgs_ridge(
                    X_sub=X_sub,
                    init=None,
                    ridge_refit=ridge_refit,
                    maxiter=refit_maxiter,
                    gtol=refit_gtol,
                )

                diagnostics = tic_diagnostics_from_X(
                    theta=theta_refit,
                    X=X_sub,
                    n=raw.shape[0],
                    order=order,
                    ridge=ridge_refit,
                    nw_bandwidth=None,
                    center_nw=True,
                )
                plic = float(diagnostics["ic_nw"])

                minus2S = float(-2.0 * diagnostics["log_likelihood"])
                pen_nw = float(2.0 * diagnostics["B_hat_nw"])
                ic_check = minus2S + pen_nw

                print(
                    "IC sanity:",
                    f"edge={num_edge}",
                    f"minus2S={minus2S:.2f}",
                    f"pen_nw={pen_nw:.2f}",
                    f"plic={plic:.2f}",
                    f"diff={plic - ic_check:.6e}",
                    f"cond_J={diagnostics.get('cond_J', np.nan):.3e}",
                    f"refit_nit={refit_info.get('nit', np.nan)}",
                    f"grad_norm={refit_info.get('grad_norm', np.nan):.3e}",
                )

                base_name = (
                    f"{j:02d}_"
                    f"l1={candidate['L1_']:.10f}_"
                    f"edge={num_edge}_"
                    f"PLIC={plic:.2f}_"
                )
                lasso_npy_name = f"{base_name}_lasso_{candidate['is_converged']}.npy"
                refit_npy_name = f"{base_name}_refit_{is_converged_refit}.npy"

                print("L1 regularization with λ=", candidate["lambda"])
                print("Saving LASSO estimate:", lasso_npy_name)
                print_result(candidate["theta_hat_lasso"], save_dir, lasso_npy_name)

                print("Saving ridge-stabilized refit estimate:", refit_npy_name)
                print_result(theta_refit, save_dir, refit_npy_name)

                return {
                    **candidate,
                    "theta_refit": theta_refit,
                    "is_converged_refit": is_converged_refit,
                    "plic": float(plic),
                    "minus2S": minus2S,
                    "pen_nw": pen_nw,
                    "B_hat_nw": float(diagnostics["B_hat_nw"]),
                    "trace_JinvI_nw": float(diagnostics["trace_JinvI_nw"]),
                    "log_likelihood_refit": float(diagnostics["log_likelihood"]),
                    "theta_norm_refit": float(np.linalg.norm(theta_refit)),
                    "theta_maxabs_refit": (
                        float(np.max(np.abs(theta_refit))) if theta_refit.size > 0 else 0.0
                    ),
                    "eigval_min_J": float(diagnostics.get("eigval_min_J", np.nan)),
                    "eigval_max_J": float(diagnostics.get("eigval_max_J", np.nan)),
                    "cond_J": float(diagnostics.get("cond_J", np.nan)),
                    "nw_bandwidth": int(diagnostics["nw_bandwidth"]),
                    "refit_optimizer": "L-BFGS-B-ridge",
                    "ridge_refit": float(refit_info.get("ridge_refit", np.nan)),
                    "refit_objective_scale": str(refit_info.get("objective_scale", "")),
                    "refit_nit": int(refit_info.get("nit", -1)),
                    "refit_grad_norm": float(refit_info.get("grad_norm", np.nan)),
                    "refit_message": str(refit_info.get("message", "")),
                    "lasso_npy_name": lasso_npy_name,
                    "refit_npy_name": refit_npy_name,
                }

            def run_grid(ls, offset=0, use_parallel=True):
                if use_parallel:
                    path_results = Parallel(n_jobs=5, backend="loky", max_nbytes="100M")(
                        delayed(fit_lasso_path)(offset + i, lam)
                        for i, lam in enumerate(ls)
                    )
                else:
                    path_results = [
                        fit_lasso_path(offset + i, lam)
                        for i, lam in enumerate(ls)
                    ]

                unique_candidates = unique_by_support(path_results)

                print(
                    f"lambda grid size: {len(ls)}, "
                    f"unique supports: {len(unique_candidates)}"
                )

                # Refit and save only unique supports. Sequential is safer for file names and print order.
                scored = [
                    refit_and_score(cand, offset + j)
                    for j, cand in enumerate(unique_candidates)
                ]

                return scored

            use_parallel = False

            # 1st stage: coarse search.
            ls1 = np.logspace(4.0, 0, 30) / X.shape[0]
            scored1 = run_grid(ls1, offset=0, use_parallel=use_parallel)

            best1 = min(scored1, key=lambda d: d["plic"])
            best_lam = best1["lambda"]

            print("\nBest result in coarse search")
            print("lambda =", best_lam)
            print("edge   =", best1["num_edge"])
            print("PLIC   =", best1["plic"])
            print("refit  =", best1["refit_npy_name"])

            # Optional 2nd stage around best lambda.
            log_best = np.log10(best_lam)
            ls2 = np.logspace(log_best + 0.5, log_best - 0.5, 10)
            # scored2 = run_grid(ls2, offset=len(scored1), use_parallel=use_parallel)
            # all_scored = scored1 + scored2
            all_scored = scored1

            best_by_support = {}
            for cand in all_scored:
                key = cand["support_groups"]
                if key not in best_by_support:
                    best_by_support[key] = cand
                elif cand["plic"] < best_by_support[key]["plic"]:
                    best_by_support[key] = cand

            final_results = list(best_by_support.values())
            final_results.sort(key=lambda d: d["plic"])
            best = final_results[0]

            path_df = pd.DataFrame([
                {
                    "lambda": d["lambda"],
                    "L1_": d["L1_"],
                    "num_edge": d["num_edge"],
                    "support_string": d.get("support_string", ""),
                    "plic": d["plic"],
                    "minus2S": d.get("minus2S", np.nan),
                    "pen_nw": d.get("pen_nw", np.nan),
                    "B_hat_nw": d.get("B_hat_nw", np.nan),
                    "trace_JinvI_nw": d.get("trace_JinvI_nw", np.nan),
                    "log_likelihood_refit": d.get("log_likelihood_refit", np.nan),
                    "theta_norm_lasso": d.get("theta_norm_lasso", np.nan),
                    "theta_maxabs_lasso": d.get("theta_maxabs_lasso", np.nan),
                    "theta_norm_refit": d.get("theta_norm_refit", np.nan),
                    "theta_maxabs_refit": d.get("theta_maxabs_refit", np.nan),
                    "eigval_min_J": d.get("eigval_min_J", np.nan),
                    "eigval_max_J": d.get("eigval_max_J", np.nan),
                    "cond_J": d.get("cond_J", np.nan),
                    "nw_bandwidth": d.get("nw_bandwidth", np.nan),
                    "refit_optimizer": d.get("refit_optimizer", ""),
                    "ridge_refit": d.get("ridge_refit", np.nan),
                    "refit_objective_scale": d.get("refit_objective_scale", ""),
                    "refit_nit": d.get("refit_nit", np.nan),
                    "refit_grad_norm": d.get("refit_grad_norm", np.nan),
                    "refit_message": d.get("refit_message", ""),
                    "lasso_converged": d["is_converged"],
                    "refit_converged": d["is_converged_refit"],
                    "lasso_npy_name": d["lasso_npy_name"],
                    "refit_npy_name": d["refit_npy_name"],
                }
                for d in final_results
            ])
            path_df = path_df.sort_values(["num_edge", "lambda"])
            path_df.to_csv(save_dir+"/"+"regularization_path_ic_debug.csv", index=False)

            print("\nRegularization path debug table")
            print(path_df[[
                "lambda",
                "num_edge",
                "plic",
                "minus2S",
                "pen_nw",
                "trace_JinvI_nw",
                "cond_J",
                "theta_norm_refit",
                "refit_optimizer",
                "ridge_refit",
                "refit_nit",
                "refit_grad_norm",
                "lasso_converged",
                "refit_converged",
            ]])

            def _parse_support_string(support_string):
                if pd.isna(support_string) or str(support_string).strip() == "":
                    return set()
                return set(int(x) for x in str(support_string).split(","))

            print("\nNested support sanity check")
            nested_warning_count = 0
            tmp_nested = path_df.sort_values("num_edge").reset_index(drop=True)
            for a in range(len(tmp_nested)):
                row_a = tmp_nested.iloc[a]
                set_a = _parse_support_string(row_a["support_string"])
                for b in range(a + 1, len(tmp_nested)):
                    row_b = tmp_nested.iloc[b]
                    set_b = _parse_support_string(row_b["support_string"])
                    if set_a.issubset(set_b) and row_b["minus2S"] > row_a["minus2S"] + 1e-5:
                        nested_warning_count += 1
                        print(
                            "WARNING: nested model has worse fit:",
                            f"edge {row_a['num_edge']} -> {row_b['num_edge']},",
                            f"minus2S {row_a['minus2S']:.6f} -> {row_b['minus2S']:.6f}",
                        )
            if nested_warning_count == 0:
                print("OK: no nested-support likelihood monotonicity violations found.")

            print("\nFinal best result")
            print("lambda =", best["lambda"])
            print("edge   =", best["num_edge"])
            print("PLIC   =", best["plic"])
            print("lasso  =", best["lasso_npy_name"])
            print("refit  =", best["refit_npy_name"])

            print("\nTop candidates")
            for k, cand in enumerate(final_results[:10]):
                print(
                    f"{k:02d}: "
                    f"lambda={cand['lambda']:.6g}, "
                    f"edge={cand['num_edge']}, "
                    f"PLIC={cand['plic']:.2f}, "
                    f"minus2S={cand.get('minus2S', np.nan):.2f}, "
                    f"pen={cand.get('pen_nw', np.nan):.2f}, "
                    f"cond_J={cand.get('cond_J', np.nan):.3e}, "
                    f"ridge_refit={cand.get('ridge_refit', np.nan):.1e}, "
                    f"refit_nit={cand.get('refit_nit', np.nan)}, "
                    f"grad_norm={cand.get('refit_grad_norm', np.nan):.3e}, "
                    f"lasso_converged={cand['is_converged']}, "
                    f"refit_converged={cand['is_converged_refit']}, "
                    f"refit_file={cand['refit_npy_name']}"
                )

        elif method == "pmle_lasso":
            print("Building X...")
            X = build_X_torus(raw, order=order, dtype=np.float64)
            print("Built X!")
            for l in np.logspace(4.0, 0, 25) / X.shape[0]:
                theta_hat, is_converged, L1_ = besag_PMLE_fista(
                    raw=raw,
                    order=order,
                    L1_=l,
                    X=X,
                )
                npy_name = f"theta_hat_l1={L1_:.4f}_edge={np.count_nonzero(theta_hat)}_{is_converged}.npy"
                print("L1 regularization with λ=", l)
                print_result(theta_hat, save_dir, npy_name)

        end_time = time()
    else:
        raise ValueError("methodが指定されていません!")

    print("Computational time = ", end_time - start_time, " seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Besag PMLE / group lasso with TIC-HAC diagnostics")
    parser.add_argument("--dim", type=int, required=True, help="データの次元")
    parser.add_argument("--order", type=int, required=True, help="マルコフモデルの次数")
    parser.add_argument(
        "--ridge-refit",
        type=float,
        default=1e-5,
        help="Mean-loss scale ridge coefficient used only for post-selection refit.",
    )
    parser.add_argument(
        "--refit-maxiter",
        type=int,
        default=5000,
        help="Maximum L-BFGS-B iterations for ridge-stabilized refit.",
    )
    parser.add_argument(
        "--refit-gtol",
        type=float,
        default=1e-8,
        help="Projected-gradient tolerance for ridge-stabilized refit.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mle", "pmle", "pmle_sgd", "pmle_lasso", "pmle_grouplasso"],
        required=True,
        help="推定手法",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs"
    )
    
    args = parser.parse_args()
    import os; os.makedirs(args.save_dir, exist_ok=True)

    run(
        args.dim,
        args.order,
        args.method,
        ridge_refit=args.ridge_refit,
        refit_maxiter=args.refit_maxiter,
        refit_gtol=args.refit_gtol,
        save_dir=args.save_dir
    )
