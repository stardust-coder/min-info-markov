import random
import numpy as np
from scipy.stats import multivariate_t, multivariate_normal
import pandas as pd

# -----------------------------
# Utilities
# -----------------------------
def flatten_outer(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """vec(xa xb^T) を縦ベクトルで返す: shape (dim*dim, 1)"""
    return np.outer(xa, xb).reshape(-1, 1)


def compute_h_from_series(x: np.ndarray, order: int) -> np.ndarray:
    """
    x: shape (N, dim)
    return: shape (dim*dim*order, 1)
    """
    N, dim_ = x.shape
    h = np.zeros((dim_ * dim_ * order, 1))

    for j in range(1, order + 1):
        # x[j:] : (N-j, dim), x[:-j] : (N-j, dim)
        block = x[j:].T @ x[:-j]   # shape (dim, dim)
        h[(j - 1) * dim_ * dim_ : j * dim_ * dim_, 0] = block.ravel()

    return h


def affected_pairs(N: int, order: int, s: int, t: int):
    """
    swap(s, t) によって影響を受ける pair (a, b, j) を列挙する。
    h_j は pair (a, b) = (u, u-j) の総和。
    """
    pairs = set()

    for j in range(1, order + 1):
        # index s が first index 側に現れる: (s, s-j)
        if s - j >= 0:
            pairs.add((s, s - j, j))
        # index s が second index 側に現れる: (s+j, s)
        if s + j < N:
            pairs.add((s + j, s, j))

        # index t が first index 側に現れる: (t, t-j)
        if t - j >= 0:
            pairs.add((t, t - j, j))
        # index t が second index 側に現れる: (t+j, t)
        if t + j < N:
            pairs.add((t + j, t, j))

    return pairs


def swap_delta_h(x: np.ndarray, s: int, t: int, order: int) -> np.ndarray:
    """
    x[s] と x[t] を swap したときの h の差分:
        delta = h_after - h_before
    """
    N, dim_ = x.shape
    delta = np.zeros((dim_ * dim_ * order, 1))
    pairs = affected_pairs(N, order, s, t)

    # swap 後の参照を関数で表現
    def x_after(idx: int) -> np.ndarray:
        if idx == s:
            return x[t]
        elif idx == t:
            return x[s]
        return x[idx]

    for a, b, j in pairs:
        before = flatten_outer(x[a], x[b])
        after = flatten_outer(x_after(a), x_after(b))
        sl = slice((j - 1) * dim_ * dim_, j * dim_ * dim_)
        delta[sl] += (after - before)

    return delta


def apply_swap_inplace(x: np.ndarray, s: int, t: int) -> None:
    """x[s], x[t] を in-place swap"""
    x[[s, t]] = x[[t, s]]


# -----------------------------
# Exchange algorithm
# -----------------------------
def exchange(rawdata: np.ndarray, theta: np.ndarray, order: int, L: int, burnin: int = 100):
    x = rawdata.copy()
    N, dim_ = x.shape

    h_current = compute_h_from_series(x, order)

    samples = []
    hstar_list = []

    accepted = 0
    trial = 0

    while accepted < L:
        s, t = sorted(random.sample(range(order, N - order), 2))

        delta = swap_delta_h(x, s, t, order)
        log_rho = float(theta.T @ delta)
        rho = np.exp(min(log_rho, 700))

        u = random.uniform(0.0, 1.0)
        if u <= min(1.0, rho):
            apply_swap_inplace(x, s, t)
            h_current = h_current + delta
            accepted += 1

            if accepted > burnin:
                samples.append(x.copy())
                hstar_list.append(h_current.copy())

            if accepted % 1000 == 0:
                print(f"{accepted} samples accepted ... continue")

        trial += 1

    print(f"Acceptance rate: {accepted}/{trial} = {accepted / trial:.4f}")
    return samples, hstar_list, x.copy()

def sample_from_mininfo_markov(N):
    _mu = [0]
    _Sigma = [[0.5]]
    _model_param = np.array([[1.0]]).T
    # _mu = [0, 0]
    # _Sigma = [[1, 0.5],
    #          [0.5, 1]]
    # _model_param = np.array([[1.0,1.0,1.0,1.0]]).T

    rv = multivariate_t(loc=_mu, shape=_Sigma, df=4)
    _samples = rv.rvs(size=N)
    if _samples.ndim == 1:
        _samples = _samples[:, None]

    samples, hstar_list, x_final = exchange(
        rawdata=_samples,
        theta=_model_param,
        order=1,
        L=10000,
        burnin=1000,
    )

    return samples[-1], _model_param
   

if __name__ == "__main__":
    # -----------------------------
    # Sampling
    # -----------------------------

    mu = [0]
    Sigma = [[0.5]]

    # mu = [0, 0]
    # Sigma = [[1, 0.5],
    #          [0.5, 1]]


    df_t = 4
    rv = multivariate_t(loc=mu, shape=Sigma, df=df_t)
    samples = rv.rvs(size=1000)
    if samples.ndim == 1:
        samples = samples[:, None]

    # Model
    order = 1   # マルコフカーネルの次数 d
    dim = 1    # 各時点の次元 p

    model_param = np.array([[1.0]])  # shape (1, 1) for dim=1, order=1
    # model_param = np.array([[1.0,1.0,1.0,1.0]]).T  # shape (4, 1) for dim=4, order=1

    print("dim:", dim, "order:", order)
    yy, hstar_list, x_final = exchange(
        rawdata=samples,
        theta=model_param,
        order=order,
        L=1000,
        burnin=100,
    )
    y = yy[-1]
    print("final shape:", y.shape)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    if y.shape[1] == 1:
        x = np.arange(y.shape[0])
        yy = np.asarray(y[:, 0]).ravel()

        fig = plt.figure(figsize=(15, 4))
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)

        ax = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1], sharey=ax)

        ax.plot(x, yy)
        ax.set_xlabel("time")
        ax.set_ylabel("value")

        ax_hist.set_xscale("log")

        ax_hist.hist(yy, bins=100, orientation="horizontal")
        ax_hist.set_xlabel("log(freq)")
        plt.setp(ax_hist.get_yticklabels(), visible=False)

        plt.tight_layout()
        plt.savefig("sample_path.png", dpi=150)
        plt.close()

    if y.shape[1] == 2:
        x = np.arange(y.shape[0])
        yy1 = np.asarray(y[:, 0]).ravel()
        yy2 = np.asarray(y[:, 1]).ravel()

        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(x, yy1, label="dim1")
        ax.plot(x, yy2, label="dim2")
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        plt.savefig("sample_path.png", dpi=150)
        plt.close()


        # Marginal (normal scale)
        plt.figure(figsize=(8, 8))
        df = pd.DataFrame(y, columns=["dim1","dim2"])  
        sns.jointplot(data=df, x="dim1", y="dim2", kind="scatter")
        plt.savefig("marginal.png", dpi=150)
        plt.close()

        # Marginal (log scale)
        plt.figure(figsize=(8, 8))
        log_df = df.apply(np.log)
        sns.jointplot(data=log_df, x="dim1", y="dim2", kind="scatter")
        plt.savefig("marginal(log).png", dpi=150)
        plt.close()
        