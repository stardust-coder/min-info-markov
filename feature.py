import numpy as np
from tqdm import tqdm

# ============================================================
# Feature construction
# ============================================================
def valid_positions(n: int, order: int) -> np.ndarray:
    """0-indexed positions corresponding to I_d={i: d<i<n-d} in 1-indexing."""
    pos = np.arange(order, n - order - 1, dtype=np.int64)
    if pos.size < 2:
        raise ValueError(
            f"Need at least two valid positions, but got N_d={pos.size}. "
            f"Increase n or decrease order."
        )
    return pos

def build_pair_index_arrays(
    n: int,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-aligned raw positions and local endpoint indices."""
    pos = valid_positions(n, order)
    i, j = np.triu_indices(pos.size, k=1)
    local_pairs = np.column_stack((i, j)).astype(np.int64, copy=False)
    raw_pairs = pos[local_pairs]
    return pos, raw_pairs, local_pairs

def precompute_trig(raw: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Precompute cosine/sine once for all observations.

    trig[t, i, 0] = cos(raw[t, i])
    trig[t, i, 1] = sin(raw[t, i])
    """
    raw = np.asarray(raw)
    trig = np.empty((raw.shape[0], raw.shape[1], 2), dtype=dtype)
    trig[..., 0] = np.cos(raw)
    trig[..., 1] = np.sin(raw)
    return trig

def torus_pair_feature_from_trig(
    trig_now: np.ndarray,
    trig_lag: np.ndarray,
) -> np.ndarray:
    """
    Vectorized torus-pair feature with the original feature order:
        for i in range(dim):
            for j in range(dim):
                cc, cs, sc, ss
    """
    
    return (
        trig_now[:, None, :, None]
        * trig_lag[None, :, None, :]
    ).reshape(-1)

def swap_delta_torus_from_trig(
    trig: np.ndarray,
    p: int,
    q: int,
    order: int,
) -> np.ndarray:
    """Compute h(original) - h(swapped) using precomputed trig values."""
    n, dim_local, _ = trig.shape
    per_lag = 4 * dim_local * dim_local
    delta = np.zeros(order * per_lag, dtype=trig.dtype)

    def after_value(idx: int) -> np.ndarray:
        if idx == p:
            return trig[q]
        if idx == q:
            return trig[p]
        return trig[idx]

    for lag in range(1, order + 1):
        candidates = (p, q, p + lag, q + lag)
        affected = []
        for t in candidates:
            if lag <= t < n and t not in affected:
                affected.append(t)

        before = np.zeros(per_lag, dtype=trig.dtype)
        after = np.zeros(per_lag, dtype=trig.dtype)

        for t in affected:
            before += torus_pair_feature_from_trig(trig[t], trig[t - lag])
            after += torus_pair_feature_from_trig(
                after_value(t),
                after_value(t - lag),
            )

        start = (lag - 1) * per_lag
        delta[start:start + per_lag] = before - after

    return delta

def build_X_torus(
    raw,
    order,
    dtype=np.float32,
    show_progress: bool = True,
):
    """Build X_ij = h(original) - h(swapped)."""
    raw = np.asarray(raw)
    n, dim_local = raw.shape
    _, raw_pairs, _ = build_pair_index_arrays(n, order)

    # Major quick optimization: compute sin/cos once instead of per pair.
    trig = precompute_trig(raw, dtype=dtype)

    n_pairs = raw_pairs.shape[0]
    n_features = order * 4 * dim_local * dim_local
    X = np.empty((n_pairs, n_features), dtype=dtype)

    iterator = enumerate(raw_pairs)
    if show_progress:
        iterator = tqdm(iterator, total=n_pairs)

    for row, (p, q) in iterator:
        X[row] = swap_delta_torus_from_trig(
            trig,
            int(p),
            int(q),
            order,
        )

    return X