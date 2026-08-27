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
    local_pairs = np.column_stack((i, j)).astype(
        np.int64,
        copy=False,
    )
    raw_pairs = pos[local_pairs]

    return pos, raw_pairs, local_pairs


def precompute_trig(
    raw: np.ndarray,
    dtype=np.float32,
) -> np.ndarray:
    """
    Precompute cosine and sine once for all observations.

    trig[t, i, 0] = cos(raw[t, i])
    trig[t, i, 1] = sin(raw[t, i])
    """
    raw = np.asarray(raw)

    trig = np.empty(
        (raw.shape[0], raw.shape[1], 2),
        dtype=dtype,
    )
    trig[..., 0] = np.cos(raw)
    trig[..., 1] = np.sin(raw)

    return trig


def multiplicative_pair_feature_from_trig(
    trig_now: np.ndarray,
    trig_lag: np.ndarray,
) -> np.ndarray:
    """
    Multiplicative feature for every ordered dimension pair (i, j):

        cos(x_now[i]) cos(x_lag[j])
        cos(x_now[i]) sin(x_lag[j])
        sin(x_now[i]) cos(x_lag[j])
        sin(x_now[i]) sin(x_lag[j])

    Feature order:
        for i in range(dim):
            for j in range(dim):
                cc, cs, sc, ss
    """
    return (
        trig_now[:, None, :, None]
        * trig_lag[None, :, None, :]
    ).reshape(-1)


def rotational_pair_feature_from_trig(
    trig_now: np.ndarray,
    trig_lag: np.ndarray,
) -> np.ndarray:
    """
    Rotational feature for every ordered dimension pair (i, j):

        cos(x_now[i] - x_lag[j])
        sin(x_now[i] - x_lag[j])

    Feature order:
        for i in range(dim):
            for j in range(dim):
                cos_diff, sin_diff
    """
    cos_now = trig_now[:, 0]
    sin_now = trig_now[:, 1]

    cos_lag = trig_lag[:, 0]
    sin_lag = trig_lag[:, 1]

    cos_diff = (
        cos_now[:, None] * cos_lag[None, :]
        + sin_now[:, None] * sin_lag[None, :]
    )

    sin_diff = (
        sin_now[:, None] * cos_lag[None, :]
        - cos_now[:, None] * sin_lag[None, :]
    )

    return np.stack(
        (cos_diff, sin_diff),
        axis=-1,
    ).reshape(-1)


def get_feature_spec(feature_type: str):
    """
    Return the pair-feature function and number of features
    per ordered dimension pair.
    """
    feature_type = feature_type.lower()

    if feature_type == "multiplicative":
        return multiplicative_pair_feature_from_trig, 4

    if feature_type == "rotational":
        return rotational_pair_feature_from_trig, 2

    raise ValueError(
        "feature_type must be either "
        "'multiplicative' or 'rotational', "
        f"but got {feature_type!r}."
    )


def swap_delta_torus_from_trig(
    trig: np.ndarray,
    p: int,
    q: int,
    order: int,
    feature_type: str = "multiplicative",
) -> np.ndarray:
    """
    Compute h(original) - h(swapped) using precomputed trig values.

    Parameters
    ----------
    feature_type : {"multiplicative", "rotational"}
        multiplicative:
            [cos cos, cos sin, sin cos, sin sin]

        rotational:
            [cos(x_now - x_lag), sin(x_now - x_lag)]
    """
    n, dim_local, _ = trig.shape

    pair_feature, features_per_pair = get_feature_spec(feature_type)

    per_lag = features_per_pair * dim_local * dim_local
    delta = np.zeros(
        order * per_lag,
        dtype=trig.dtype,
    )

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
            before += pair_feature(
                trig[t],
                trig[t - lag],
            )

            after += pair_feature(
                after_value(t),
                after_value(t - lag),
            )

        start = (lag - 1) * per_lag
        delta[start:start + per_lag] = before - after

    return delta


def build_X_torus(
    raw,
    order,
    feature_type: str = "multiplicative",
    dtype=np.float32,
    show_progress: bool = True,
):
    """
    Build X_ij = h(original) - h(swapped).

    Parameters
    ----------
    raw : array-like, shape (n, dim_local)
        Angular observations.

    order : int
        Maximum lag.

    feature_type : {"multiplicative", "rotational"}
        multiplicative:
            For each (i, j), use
            [cos cos, cos sin, sin cos, sin sin].

        rotational:
            For each (i, j), use
            [cos(x_t,i - x_t-lag,j),
             sin(x_t,i - x_t-lag,j)].

    dtype : numpy dtype
        Output dtype.

    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    X : ndarray
        If feature_type == "multiplicative":
            shape = (
                n_pairs,
                order * 4 * dim_local**2,
            )

        If feature_type == "rotational":
            shape = (
                n_pairs,
                order * 2 * dim_local**2,
            )
    """
    raw = np.asarray(raw)
    n, dim_local = raw.shape

    _, raw_pairs, _ = build_pair_index_arrays(n, order)

    trig = precompute_trig(raw, dtype=dtype)

    _, features_per_pair = get_feature_spec(feature_type)

    n_pairs = raw_pairs.shape[0]
    n_features = (
        order
        * features_per_pair
        * dim_local
        * dim_local
    )

    X = np.empty(
        (n_pairs, n_features),
        dtype=dtype,
    )

    iterator = enumerate(raw_pairs)

    if show_progress:
        iterator = tqdm(iterator, total=n_pairs)

    for row, (p, q) in iterator:
        X[row] = swap_delta_torus_from_trig(
            trig=trig,
            p=int(p),
            q=int(q),
            order=order,
            feature_type=feature_type,
        )

    return X