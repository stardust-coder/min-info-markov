import random
from typing import Callable

import numpy as np

# ============================================================
# Compute h from full series
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

def compute_h_from_series(
    x: np.ndarray,
    order: int,
    pair_feature: Callable[[np.ndarray, np.ndarray], np.ndarray],
    feature_dim: int,
) -> np.ndarray:
    """
    時系列 x から十分統計量 h を計算する。

    x: shape (N, p)
    order: Markov order
    pair_feature: h(x_t, x_{t-j})
    feature_dim: pair_feature の次元

    return:
        shape (feature_dim * order, 1)

    数式では、

        H_j(x) = sum_{t=j}^{N-1} h(x_t, x_{t-j})

    を j = 1, ..., order について縦に並べたもの。

    order = 1 かつ h(x_t, x_{t-1}) = x_t ⊗ x_{t-1} なら、
        return shape = (p^2, 1)

    order = q なら、
        return shape = (q * p^2, 1)
    """
    N, p = x.shape
    H = np.zeros((feature_dim * order, 1))

    for j in range(1, order + 1):
        block = np.zeros((feature_dim, 1))

        for t in range(j, N):
            block += pair_feature(x[t], x[t - j])

        start = (j - 1) * feature_dim
        end = j * feature_dim
        H[start:end] = block

    return H


# ============================================================
# Affected pairs under swap
# ============================================================

def affected_pairs(N: int, order: int, s: int, t: int):
    """
    swap(s, t) によって影響を受ける pair (a, b, j) を列挙する。

    H_j は pair

        (a, b) = (u, u-j)

    の和として定義される。

    つまり、

        H_j = sum_u h(x_u, x_{u-j})

    である。
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


# ============================================================
# Delta h under swap
# ============================================================

def compute_h_from_series(
    x: np.ndarray,
    order: int,
    pair_feature: Callable[[np.ndarray, np.ndarray], np.ndarray],
    feature_dim: int,
) -> np.ndarray:
    N, p = x.shape
    H = np.zeros((feature_dim * order, 1), dtype=np.float64)

    for j in range(1, order + 1):
        block = np.zeros((feature_dim, 1), dtype=np.float64)

        for tt in range(j, N):
            feat = pair_feature(x[tt], x[tt - j]).reshape(feature_dim, 1)
            block += feat

        start = (j - 1) * feature_dim
        end = j * feature_dim
        H[start:end] = block

    return H


def apply_swap_inplace(x: np.ndarray, s: int, t: int) -> None:
    """
    x[s], x[t] を in-place swap する。
    """
    x[[s, t]] = x[[t, s]]


def swap_delta_h(
    x: np.ndarray,
    s: int,
    t: int,
    order: int,
    pair_feature: Callable[[np.ndarray, np.ndarray], np.ndarray],
    feature_dim: int,
) -> np.ndarray:
    N, p = x.shape
    delta = np.zeros((feature_dim * order, 1), dtype=np.float64)

    pairs = affected_pairs(N, order, s, t)

    def x_after(idx: int) -> np.ndarray:
        if idx == s:
            return x[t]
        elif idx == t:
            return x[s]
        else:
            return x[idx]

    for a, b, j in pairs:
        before = pair_feature(x[a], x[b]).reshape(feature_dim, 1)
        after = pair_feature(x_after(a), x_after(b)).reshape(feature_dim, 1)

        start = (j - 1) * feature_dim
        end = j * feature_dim

        delta[start:end] += after - before

    return delta

# ============================================================
# Exchange algorithm
# ============================================================

def exchange(
    rawdata: np.ndarray,
    theta: np.ndarray,
    order: int,
    L: int,
    pair_feature: Callable[[np.ndarray, np.ndarray], np.ndarray],
    feature_dim: int | None = None,
    burnin: int = 100,
):
    """
    Exchange algorithm.

    rawdata:
        shape (N, p)

    theta:
        shape (feature_dim * order, 1)

    order:
        Markov order

    pair_feature:
        h(x_t, x_{t-j})

    feature_dim:
        h(x_t, x_{t-j}) の次元。
        h(x_t, x_{t-j}) = x_t ⊗ x_{t-j} の場合は p^2。

    L:
        accepted swap の総数

    burnin:
        accepted count が burnin を超えた後のサンプルを保存する。
    """
    x = rawdata.copy()
    N, p = x.shape

    if feature_dim is None:
        feature_dim = p * p

    expected_theta_shape = (feature_dim * order, 1)
    if theta.shape != expected_theta_shape:
        raise ValueError(
            f"theta.shape must be {expected_theta_shape}, "
            f"but got {theta.shape}."
        )

    h_current = compute_h_from_series(
        x=x,
        order=order,
        pair_feature=pair_feature,
        feature_dim=feature_dim,
    )

    samples = []
    hstar_list = []

    accepted = 0
    trial = 0

    # swap の候補が空にならないか確認
    if N <= 2 * order:
        raise ValueError(
            f"N must be larger than 2 * order. "
            f"Got N={N}, order={order}."
        )

    while accepted < L:
        s, t = sorted(random.sample(range(order, N - order), 2))

        delta = swap_delta_h(
            x=x,
            s=s,
            t=t,
            order=order,
            pair_feature=pair_feature,
            feature_dim=feature_dim,
        )

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
                print(f"{accepted} swaps accepted ... continue")

        trial += 1

    print(f"Acceptance rate: {accepted}/{trial} = {accepted / trial:.4f}")

    return samples, hstar_list, x.copy()


# ============================================================
# Sampling from minimum information Markov model
# ============================================================
def sample_from_mininfo_markov(N, dim):
    p = dim
    order = 1

    # torus_pair_feature は cc, cs, sc, ss の4種類を
    # 各 (i, j) ペアごとに返すので 4 * p * p
    feature_dim = 4 * p * p

    _model_param = np.ones((feature_dim * order, 1), dtype=np.float64)

    _samples = np.random.uniform(0.0, 2.0 * np.pi, size=(N, p))

    samples, hstar_list, x_final = exchange(
        rawdata=_samples,
        theta=_model_param,
        order=order,
        L=10000,
        pair_feature=torus_pair_feature,
        feature_dim=feature_dim,
        burnin=1000,
    )

    return samples[-1], _model_param