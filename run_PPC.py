import numpy as np
from numpy import cos, atan, sin
import matplotlib.pyplot as plt
import statsmodels.graphics.api as smg
from statsmodels.graphics.tsaplots import plot_acf
from time import time
from tqdm import tqdm
import pandas as pd
import random
from random import sample
import pdb
from itertools import combinations
import copy
from joblib import Parallel, delayed
from data import Kuramoto_Model

def sample_plot(data):
    '''
    Input: (steps,dim)
    '''
    df = pd.DataFrame(data)
    df.plot(figsize=(15,5))
    plt.savefig("sample_plot.png")

def MLE(Y, order):
    from statsmodels.tsa.api import VAR, ARIMA
    if Y.shape[1] == 1:
        model = ARIMA(Y, order=(order, 0, 0), trend="n") #AR(d)
        results = model.fit()
    
    else:
        model = VAR(Y)
        results = model.fit(maxlags=order, ic=None, trend="n")
    print(results.summary())
    return results
    

def run(dim, order, method):
    raw = Kuramoto_Model(N=dim)
    sample_plot(raw)

    # ============================================================
    # Feature construction
    # ============================================================
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


    # ============================================================
    # Design matrix
    # ============================================================

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


    # ============================================================
    # Group lasso groups
    # ============================================================

    def make_groups(dim, order, design_matrix):
        # n_features = order * dim * dim * 4
        n_features = design_matrix.shape[1]
        return np.arange(n_features).reshape(-1, 4*order)


    # ============================================================
    # Main: Besag PMLE with FISTA
    # ============================================================

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
    
    def print_result(theta_hat, npy_name):
        # print("--- 推定するパラメタ数 --- ")
        # print(len(theta_hat))
        # print("--- 推定値 --- ")
        # print(theta_hat.T)
        np.save(npy_name,theta_hat)
        print("#Non zero entries:", np.count_nonzero(theta_hat))

    if method == "mle":
        ### MLE for AR or VAR
        start_time = time()
        res_mle = MLE(raw, order=order)
        # theta_hat = res_mle.params[0] / res_mle.params[1] #AR(1) case
        # theta_hat = np.array([res_mle.params[k]/res_mle.params[-1] for k in range(res_mle.params.shape[0]-1)]) #AR(d) case
        theta_hat = res_mle.params.T @ np.linalg.inv(res_mle.sigma_u) #VAR(1) case
        theta_hat = theta_hat.flatten() #VAR(1) case
        optimization_time = None
        end_time = time()

    elif "pmle" in method:
        # Besag's PMLE for any model
        start_time = time()

        if method == "pmle_grouplasso":

            def print_result(theta_hat, npy_name):
                np.save(npy_name, theta_hat)
                print("#Non zero entries:", np.count_nonzero(theta_hat))

            print("Building X...")
            X = build_X_torus(raw, order=order, dtype=np.float64)
            print("Built X!")

            group_size = 4 * order
            tol = 1e-8

            def get_nonzero_mask(theta_hat, group_size, tol=1e-8):
                theta_hat = np.asarray(theta_hat)
                n_groups = len(theta_hat) // group_size

                keep = []
                for g in range(n_groups):
                    start = g * group_size
                    end = (g + 1) * group_size
                    block = theta_hat[start:end]

                    if np.linalg.norm(block) > tol:
                        keep.extend(range(start, end))

                return tuple(keep)

            def fit_lasso_path(i, lam):
                theta_hat, is_converged, L1_ = besag_PMLE_fista(
                    raw=raw,
                    order=order,
                    L1_=lam,
                    group_lasso=True,
                    X=X,
                    init=None,
                )

                nonzero_mask = get_nonzero_mask(
                    theta_hat,
                    group_size=group_size,
                    tol=tol,
                )

                return {
                    "i": i,
                    "lambda": lam,
                    "L1_": L1_,
                    "theta_hat_lasso": theta_hat,
                    "is_converged": is_converged,
                    "nonzero_mask": nonzero_mask,
                    "num_edge": len(nonzero_mask) // group_size,
                }

            def unique_by_support(candidates):
                unique = {}

                for cand in candidates:
                    key = cand["nonzero_mask"]

                    if key not in unique:
                        unique[key] = cand

                return list(unique.values())

            def refit_and_score(candidate, j):
                nonzero_mask = list(candidate["nonzero_mask"])
                num_edge = candidate["num_edge"]

                X_sub = X[:, nonzero_mask] if len(nonzero_mask) > 0 else X[:, []]
                init_refit = np.asarray(candidate["theta_hat_lasso"])[nonzero_mask]
                theta_refit, is_converged_refit, _, plic, res = besag_PMLE_fista(
                    raw=raw,
                    order=order,
                    L1_=0.0,
                    group_lasso=False,
                    X=X_sub,
                    ic=True,
                    init=init_refit,
                )

                ll = res[0]
                pen = res[1]

                base_name = (
                    f"{j:02d}_"
                    f"l1={candidate['L1_']:.10f}_"
                    f"edge={num_edge}_"
                    f"PLIC={plic:.2f}_"
                    f"ll={ll:.2f}_"
                    f"pen={pen:.2f}"
                )

                lasso_npy_name = (
                    f"{base_name}_lasso_"
                    f"{candidate['is_converged']}.npy"
                )

                refit_npy_name = (
                    f"{base_name}_refit_"
                    f"{is_converged_refit}.npy"
                )

                print("L1 regularization with λ=", candidate["lambda"])
                print("Saving LASSO estimate:", lasso_npy_name)
                print_result(candidate["theta_hat_lasso"], lasso_npy_name)

                print("Saving unregularized refit estimate:", refit_npy_name)
                print_result(theta_refit, refit_npy_name)

                return {
                    **candidate,
                    "theta_refit": theta_refit,
                    "is_converged_refit": is_converged_refit,
                    "plic": plic,
                    "ll": ll,
                    "pen": pen,
                    "lasso_npy_name": lasso_npy_name,
                    "refit_npy_name": refit_npy_name,
                }

            def run_grid(ls, offset=0, use_parallel=True):
                if use_parallel:
                    path_results = Parallel(n_jobs=10, backend="loky", max_nbytes="100M")(
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

                # refit + save は unique support のみ実行
                # 保存ファイル名の衝突やprint順を避けるため、ここは逐次が安全
                scored = [
                    refit_and_score(cand, offset + j)
                    for j, cand in enumerate(unique_candidates)
                ]

                return scored

            use_parallel = True

            # 1st stage: coarse search
            ls1 = np.logspace(3.5, 0, 10)/ X.shape[0]
            scored1 = run_grid(ls1, offset=0, use_parallel=use_parallel)

            best1 = min(scored1, key=lambda d: d["plic"])
            best_lam = best1["lambda"]

            print("\nBest result in coarse search")
            print("lambda =", best_lam)
            print("edge   =", best1["num_edge"])
            print("PLIC   =", best1["plic"])
            print("ll     =", best1["ll"])
            print("pen    =", best1["pen"])
            print("refit  =", best1["refit_npy_name"])

            # 2nd stage: search around best lambda
            log_best = np.log10(best_lam)

            ls2 = np.logspace(
                log_best + 0.5,
                log_best - 0.5,
                10,
            )

            scored2 = run_grid(ls2, offset=len(scored1), use_parallel=use_parallel)

            # 全体で重複 support をもう一度まとめる
            best_by_support = {}

            for cand in scored1 + scored2:
                key = cand["nonzero_mask"]

                if key not in best_by_support:
                    best_by_support[key] = cand
                elif cand["plic"] < best_by_support[key]["plic"]:
                    best_by_support[key] = cand

            final_results = list(best_by_support.values())
            final_results.sort(key=lambda d: d["plic"])

            best = final_results[0]

            print("\nFinal best result")
            print("lambda =", best["lambda"])
            print("edge   =", best["num_edge"])
            print("PLIC   =", best["plic"])
            print("ll     =", best["ll"])
            print("pen    =", best["pen"])
            print("lasso  =", best["lasso_npy_name"])
            print("refit  =", best["refit_npy_name"])

            print("\nTop candidates")
            for k, cand in enumerate(final_results[:10]):
                print(
                    f"{k:02d}: "
                    f"lambda={cand['lambda']:.6g}, "
                    f"edge={cand['num_edge']}, "
                    f"PLIC={cand['plic']:.2f}, "
                    f"ll={cand['ll']:.2f}, "
                    f"pen={cand['pen']:.2f}, "
                    f"lasso_converged={cand['is_converged']}, "
                    f"refit_converged={cand['is_converged_refit']}, "
                    f"refit_file={cand['refit_npy_name']}"
                )

        elif method == "pmle_lasso":
            for l in np.logspace(3.5, 0, 25)/ X.shape[0]:
                theta_hat, is_converged, L1_ = besag_PMLE_fista(raw=raw, order=order, L1_=l)  
                npy_name = f"theta_hat_l1={L1_:.4f}_edge={np.count_nonzero(theta_hat)}_{is_converged}"
                print("L1 regularization with λ=",l)
                print_result(theta_hat, npy_name) 
        end_time = time()
    else:
        raise ValueError("methodが指定されていません!")

    print("Computational time = ", end_time-start_time, " seconds.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="サンプルCLI")
    parser.add_argument("--dim", type=int, help="データの次元")
    parser.add_argument("--order", type=int, help="マルコフモデルの次数")
    parser.add_argument("--method", type=str, choices=["mle", "pmle", "pmle_sgd", "pmle_lasso", "pmle_grouplasso"], help="推定手法")

    args = parser.parse_args()

    run(args.dim, args.order, args.method)