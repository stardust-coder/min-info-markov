import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy.special import logsumexp, i0


def von_mises_density(x, r, mu):
    """
    von Mises density on [0, 2π):
        π(x) = exp(r cos(x - μ)) / (2π I_0(r))
    """
    return np.exp(r * np.cos(x - mu)) / (2 * np.pi * i0(r))


def sinkhorn_circular_markov(
    theta,
    r,
    mu,
    N=1024,
    max_iter=10_000,
    tol=1e-12,
    verbose=False,
):
    """
    Construct a circular Markov kernel with von Mises stationary density.

    Parameters
    ----------
    theta : array-like, shape (4,)
        Parameters θ = (θ1, θ2, θ3, θ4), corresponding to

            [[θ1, θ2],
             [θ3, θ4]]

        in u(y)^T A u(x).

    r : float
        Concentration parameter of the target von Mises distribution.

    mu : float
        Mean direction of the target von Mises distribution.

    N : int
        Number of grid points on [0, 2π).

    max_iter : int
        Maximum number of Sinkhorn iterations.

    tol : float
        Convergence tolerance.

    verbose : bool
        If True, prints progress.

    Returns
    -------
    result : dict
        Dictionary containing:
            x        : grid points
            w        : quadrature weight
            pi       : target stationary density on grid
            P        : Markov kernel matrix, P[i, j] ≈ p(x_j | x_i)
            kappa    : numerical κ(x_i)
            delta    : numerical δ(x_i)
            log_a    : row scaling log a_i
            log_b    : column scaling log b_j
            err_row  : final row normalization error
            err_stat : final stationarity error
    """

    theta = np.asarray(theta, dtype=float)
    if theta.shape != (4,):
        raise ValueError("theta must have shape (4,)")

    A = np.array(
        [
            [theta[0], theta[1]],
            [theta[2], theta[3]],
        ],
        dtype=float,
    )

    # Grid and quadrature weight
    x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    w = 2 * np.pi / N

    # u(x) = (cos x, sin x)
    U = np.column_stack([np.cos(x), np.sin(x)])

    # Target stationary density
    pi = von_mises_density(x, r, mu)

    # Normalize numerically, just in case
    pi = pi / (np.sum(w * pi))

    # log K[i, j] = u(x_j)^T A u(x_i)
    #
    # Row i = previous state x_i
    # Col j = next state x_j
    logK = U @ A @ U.T
    logK = logK.T
    # Equivalent:
    # logK[i, j] = U[j]^T A U[i]

    # Initialize log b_j = 0
    log_b = np.zeros(N)
    log_a = np.zeros(N)

    log_w = np.log(w)
    log_pi = np.log(pi)

    for it in range(max_iter):
        old_log_b = log_b.copy()

        # a_i = 1 / sum_j w K_ij b_j
        log_a = -logsumexp(log_w + logK + log_b[None, :], axis=1)

        # b_j = pi_j / sum_i w pi_i a_i K_ij
        log_b = log_pi - logsumexp(
            log_w + log_pi[:, None] + log_a[:, None] + logK,
            axis=0,
        )

        diff = np.max(np.abs(log_b - old_log_b))

        if verbose and (it % 100 == 0 or diff < tol):
            print(f"iter={it}, max |Δ log_b|={diff:.3e}")

        if diff < tol:
            break
    else:
        print("Warning: Sinkhorn iteration did not converge within max_iter.")

    # Construct P[i, j] = a_i K_ij b_j
    logP = log_a[:, None] + logK + log_b[None, :]
    P = np.exp(logP)

    # Check row normalization:
    # ∫ p(y | x_i) dy ≈ sum_j w P[i, j] = 1
    row_sums = np.sum(w * P, axis=1)
    err_row = np.max(np.abs(row_sums - 1.0))

    # Check stationarity:
    # π_j ≈ sum_i w π_i P[i, j]
    stat = np.sum(w * pi[:, None] * P, axis=0)
    err_stat = np.max(np.abs(stat - pi))

    # Recover kappa and delta.
    #
    # P_ij = exp{u_j^T A u_i + kappa_j - kappa_i - delta_j}
    #
    # Also P_ij = a_i K_ij b_j.
    #
    # Hence:
    #   a_i = exp{-kappa_i}
    #   b_j = exp{kappa_j - delta_j}
    #
    # Therefore:
    #   kappa_i = -log_a_i + constant
    #   delta_i = kappa_i - log_b_i
    #
    # Fix gauge by setting mean kappa = 0.
    kappa = -log_a
    kappa = kappa - np.sum(w * kappa) / (2 * np.pi)

    delta = kappa - log_b

    # delta is also determined only up to the same gauge convention.
    # With the above kappa normalization, delta is fixed accordingly.

    return {
        "x": x,
        "w": w,
        "pi": pi,
        "P": P,
        "kappa": kappa,
        "delta": delta,
        "log_a": log_a,
        "log_b": log_b,
        "err_row": err_row,
        "err_stat": err_stat,
        "n_iter": it + 1,
    }

def simulate_circular_markov(res, T=5000, x0=None, burnin=0, random_state=None):
    """
    Simulate the circular Markov chain using the Sinkhorn-corrected kernel.

    Parameters
    ----------
    res : dict
        Output of sinkhorn_circular_markov.

    T : int
        Number of samples after burn-in.

    x0 : float or None
        Initial angle. If None, starts from a random grid point.

    burnin : int
        Number of burn-in steps.

    random_state : int or None
        Random seed.

    Returns
    -------
    samples : ndarray, shape (T,)
        Simulated angles in [0, 2π).

    indices : ndarray, shape (T,)
        Grid indices corresponding to samples.
    """

    rng = np.random.default_rng(random_state)

    x_grid = res["x"]
    w = res["w"]
    P = res["P"]
    N = len(x_grid)

    # P is a density matrix, so transition probabilities are w * P[i, :]
    Q = w * P

    # Numerical row renormalization
    Q = Q / Q.sum(axis=1, keepdims=True)

    total_steps = T + burnin
    indices = np.empty(total_steps, dtype=int)

    if x0 is None:
        indices[0] = rng.integers(N)
    else:
        # Choose nearest grid point to x0 modulo 2π
        x0 = x0 % (2 * np.pi)
        indices[0] = np.argmin(np.abs(np.angle(np.exp(1j * (x_grid - x0)))))

    for t in range(1, total_steps):
        i = indices[t - 1]
        indices[t] = rng.choice(N, p=Q[i])

    indices = indices[burnin:]
    samples = x_grid[indices]

    return samples, indices

def plot_timeseries_with_stationary_marginal(
    samples,
    res,
    max_points=None,
    bins=80,
    show_empirical=True,
):
    x_grid = res["x"]
    pi_grid = res["pi"]

    samples_plot = samples[:max_points] if max_points is not None else samples
    t = np.arange(len(samples_plot))

    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[5, 1.0],
        wspace=0.03,
    )

    ax_ts = fig.add_subplot(gs[0, 0])
    ax_marg = fig.add_subplot(gs[0, 1], sharey=ax_ts)

    # 左: 時系列。sns.lineplot は使わない
    ax_ts.plot(
        t,
        samples_plot,
        linewidth=0.8,
        label=r"$x_t$",
    )

    ax_ts.set_xlabel("time")
    ax_ts.set_ylabel(r"$x_t$")
    ax_ts.set_title("Simulated univariate PPC Markov process")
    ax_ts.set_ylim(0, 2 * np.pi)

    # Right: empirical marginal distribution, horizontal histogram
    if show_empirical:
        sns.histplot(
            y=samples_plot,
            bins=bins,
            stat="density",
            alpha=0.35,
            ax=ax_marg,
            label="empirical",
        )

    # 右: 定常分布。横軸 density, 縦軸 x
    ax_marg.plot(
        pi_grid,
        x_grid,
        linewidth=2.0,
        label="theoretical",
    )

    ax_marg.set_xlabel("density")
    ax_marg.set_ylabel("")
    ax_marg.set_title("marginal")
    ax_marg.tick_params(axis="y", labelleft=False)
    ax_marg.set_ylim(0, 2 * np.pi)
    ax_marg.set_xlim(left=0)
    ax_marg.legend()

    plt.tight_layout()
    plt.savefig("sinkhorn.png")


if __name__ == "__main__":
    theta = np.array([50,-15,15,500])

    r = 1e-5       # von Mises concentration
    mu = 0.0      # von Mises mean direction

    res = sinkhorn_circular_markov(
        theta=theta,
        r=r,
        mu=mu,
        N=1024,
        max_iter=10_000,
        tol=1e-12,
        verbose=True,
    )

    print("iterations:", res["n_iter"])
    print("row error:", res["err_row"])
    print("stationarity error:", res["err_stat"])

    samples, indices = simulate_circular_markov(
        res,
        T=200000,
        burnin=50000,
        random_state=123,
    )

    plot_timeseries_with_stationary_marginal(
        samples,
        res,
        max_points=1000,
        bins=80,
        show_empirical=True,
    )