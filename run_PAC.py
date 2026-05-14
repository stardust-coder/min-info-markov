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
from data import Kuramoto_Model, simulated_data, extract_phase_and_amplitude

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
    eeg_signal, meta = simulated_data(mode="nadalin")
    eeg_signal = eeg_signal[:1500]
    raw = extract_phase_and_amplitude(
        eeg_signal,
        fs=meta["fs"],
        amp_band=meta["amp_band"],
        phase_band=meta["phase_band"],
    )
    
    # sample_plot(raw)
    # print(raw)

    # ============================================================
    # Feature construction
    # ============================================================

    def pair_feature(x_now, x_lag):
        """
        x_now, x_lag: shape (dim,)
            dim = 2 * num_channel
            [phase_0, amp_0, phase_1, amp_1, ...]

        return:
            shape (9 * num_channel * num_channel,)
        """
        x_now = np.asarray(x_now)
        x_lag = np.asarray(x_lag)

        if x_now.shape != x_lag.shape:
            raise ValueError(
                f"x_now and x_lag must have the same shape, got {x_now.shape} and {x_lag.shape}"
            )

        if x_now.ndim != 1:
            raise ValueError(f"x_now and x_lag must have shape (dim,), got {x_now.shape}")

        dim = x_now.shape[0]

        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got dim={dim}")

        phase_now = x_now[0::2]
        amp_now = x_now[1::2]

        phase_lag = x_lag[0::2]
        amp_lag = x_lag[1::2]

        now = np.column_stack([
            np.cos(phase_now),
            np.sin(phase_now),
            amp_now,
        ])

        lag = np.column_stack([
            np.cos(phase_lag),
            np.sin(phase_lag),
            amp_lag,
        ])

        out = now[:, None, :, None] * lag[None, :, None, :]

        return out.reshape(-1).astype(np.float64, copy=False)


    # ============================================================
    # Swap delta
    # ============================================================

    def swap_delta(state, p, q, order):
        """
        state[p] と state[q] を swap したときの

            h(original) - h(swapped)

        を差分で計算する。

        state: shape (n, dim)
            dim = 2 * num_channel
            [phase_0, amp_0, phase_1, amp_1, ...]
        p, q: 0-indexed
        order: lag order
        """
        state = np.asarray(state)

        if state.ndim != 2:
            raise ValueError(f"state must have shape (n, dim), got {state.shape}")

        n, dim = state.shape

        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got dim={dim}")

        if not (0 <= p < n and 0 <= q < n):
            raise IndexError(f"p and q must be in [0, {n}), got p={p}, q={q}")

        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")

        num_channel = dim // 2
        per_lag = 9 * num_channel * num_channel

        if p == q:
            return np.zeros(order * per_lag, dtype=np.float64)

        delta = np.zeros(order * per_lag, dtype=np.float64)

        def after_value(idx):
            if idx == p:
                return state[q]
            if idx == q:
                return state[p]
            return state[idx]

        for lag in range(1, order + 1):
            affected = {p, q, p + lag, q + lag}
            affected = [t for t in affected if lag <= t < n]

            before = np.zeros(per_lag, dtype=np.float64)
            after = np.zeros(per_lag, dtype=np.float64)

            for t in affected:
                before += pair_feature(state[t], state[t - lag])
                after += pair_feature(after_value(t), after_value(t - lag))

            start = (lag - 1) * per_lag
            end = lag * per_lag
            delta[start:end] = before - after

        return delta


    # ============================================================
    # Design matrix
    # ============================================================

    def build_X(raw, order, dtype=np.float32):
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

        num_channel = dim // 2
        n_pairs = (n - 2 * order) * (n - 2 * order - 1) // 2
        n_features = order * 9 * num_channel * num_channel

        X = np.empty((n_pairs, n_features), dtype=dtype)
        pairs = combinations(range(order, n - order), 2)

        for row, (p, q) in enumerate(tqdm(pairs, total=n_pairs)):
            X[row] = swap_delta(raw, p, q, order)

        return X


    # ============================================================
    # Group lasso groups
    # ============================================================

    def make_groups(dim, order):
        n_features = order * 9 * dim * dim 
        return np.arange(n_features).reshape(-1, 9*order)


    # ============================================================
    # Main: Besag PMLE with FISTA
    # ============================================================

    def besag_PMLE_fista(raw, order, group_lasso=False, L1_=1):
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

        assert n <= 2000

        print("Building X ...")
        X = build_X(raw, order=order, dtype=np.float32)

        n_pairs, n_features = X.shape
        y = np.ones(n_pairs, dtype=np.float32)

        print("Start Fitting with FISTA ...")
        start_fit = time()

        from fista import LogisticRegressionFISTA

        clf = LogisticRegressionFISTA(
            eta=1.0,
            n_iter=1000,
            tol=1e-10,
            grad_tol=1e-7,
            l1=L1_,
            fit_intercept=False,
            line_search=True,
            verbose=True,
        )

        if group_lasso:
            clf.groups = make_groups(dim, order)
            print("Groups shape:", clf.groups.shape)

        clf.fit(X, y, is_group=group_lasso)

        end_fit = time()
        print(f"Optimization took {end_fit - start_fit:.2f} sec")

        return clf.w, end_fit - start_fit, L1_
    
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
  
            def worker(i, l):
                theta_hat, optimization_time, L1_ = besag_PMLE_fista(
                    raw=raw, order=order, L1_=l, group_lasso=True
                )

                edge = int(np.count_nonzero(theta_hat) / (4 * order))
                npy_name = f"{i:02d}_theta_hat_l1={L1_:.10f}_edge={edge}"

                print("L1 regularization with λ=", l)
                print_result(theta_hat, npy_name)

            ls = np.logspace(-4, -3, 30)

            use_parallel = True
            if use_parallel:
                Parallel(n_jobs=-1)(
                    delayed(worker)(i, l) for i, l in enumerate(ls)
                )
            else:
                for i, l in enumerate(ls):
                    worker(i, l)

        elif method == "pmle_lasso":
            for l in np.logspace(-3, 1, 5):
                theta_hat, optimization_time, L1_ = besag_PMLE_fista(raw=raw, order=order, L1_=l)  
                npy_name = f"theta_hat_l1={L1_:.4f}_edge={np.count_nonzero(theta_hat)}"
                print("L1 regularization with λ=",l)
                print_result(theta_hat, npy_name) 
        
        elif method == "pmle_fista":
            theta_hat, optimization_time, L1_ = besag_PMLE_fista(raw=raw, order=order, L1_=0)  
            npy_name = f"theta_hat_l1={L1_:.4f}_nonzero={np.count_nonzero(theta_hat)}"
            print("L1 regularization with λ=",0)
            print_result(theta_hat, npy_name) 
        end_time = time()
    else:
        raise ValueError("methodが指定されていません！")

    
    print("--- 所要時間 うち 勾配法 ---")
    comp_time = (end_time-start_time, " seconds / ", optimization_time, " seconds")
    print(comp_time)
    return theta_hat, None, comp_time


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="hoge")
    parser.add_argument("--dim", type=int, help="データの次元")
    parser.add_argument("--order", type=int, help="マルコフモデルの次数")
    parser.add_argument("--method", type=str, choices=["pmle_fista"], help="推定手法")

    args = parser.parse_args()

    loss_list = []
    time_list = []
    for r in range(1):
        print(f"Run {r}")
        _, loss_, time_ = run(args.dim, args.order, args.method)
        loss_list.append(loss_)
        time_list.append(time_)

    from scipy import stats
    print("Average L2 error for 30 runs:", sum(loss_list)/len(loss_list))
    print("Standard error for 30 runs", stats.sem(loss_list))
    print("Average whole estimation time for 30 runs:",sum([t[0] for t in time_list])/len(time_list))
    if args.method != "mle":
        print("Average time consumed for gradient descent for 30 runs:",sum([t[1] for t in time_list])/len(time_list))