from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import i0e, i1e


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
KappaScoreMethod = Literal["mean_log", "geometric", "minimum", "mean"]


# ============================================================
# 基本関数
# ============================================================

def wrap_angle(x: FloatArray) -> FloatArray:
    """角度を [-pi, pi) に折り返す。"""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def log_i0(kappa: FloatArray) -> FloatArray:
    """
    log(I_0(kappa)) を数値的に安定に計算する。

    i0e(kappa) = exp(-kappa) I_0(kappa), kappa >= 0
    """
    return np.log(i0e(kappa)) + kappa


def a1_ratio(kappa: FloatArray) -> FloatArray:
    """A_1(kappa) = I_1(kappa) / I_0(kappa)。"""
    return i1e(kappa) / i0e(kappa)


def resultant_length_to_kappa(r: float) -> float:
    """
    平均合成ベクトル長 r からvon Mises集中度kappaを近似する。
    """
    r = float(np.clip(r, 0.0, 0.999999))

    if r < 0.53:
        return 2.0 * r + r**3 + 5.0 * r**5 / 6.0

    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1.0 - r)

    return 1.0 / (r**3 - 4.0 * r**2 + 3.0 * r)


# ============================================================
# AR-TG用デザイン行列
# ============================================================

def make_lagged_design(
    phases: FloatArray,
    n_lags: int,
    include_intercept: bool = True,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """
    位相系列からAR-TG用ラグ特徴量を作る。

    Parameters
    ----------
    phases
        shape (T, d) の位相行列。単位はラジアン。
    n_lags
        使用するラグ数。
    include_intercept
        TrueならXの先頭列に1を追加。

    Returns
    -------
    X
        shape (T - n_lags, p)

        列順:
            intercept,
            lag 1: ch1 cos, ch1 sin, ..., chd cos, chd sin,
            lag 2: ch1 cos, ch1 sin, ..., chd cos, chd sin,
            ...

    Y
        shape (T - n_lags, d) の現在位相。

    time_indices
        Yの各行に対応する元データ上の時点。
    """
    phases = np.asarray(phases, dtype=np.float64)

    if phases.ndim != 2:
        raise ValueError("phases must have shape (T, d).")

    if n_lags < 1:
        raise ValueError("n_lags must be at least 1.")

    if phases.shape[0] <= n_lags:
        raise ValueError("Not enough observations for requested lags.")

    if not np.all(np.isfinite(phases)):
        raise ValueError("phases contains NaN or infinite values.")

    phases = wrap_angle(phases)

    n_time, n_channels = phases.shape
    n_samples = n_time - n_lags

    blocks: list[FloatArray] = []

    if include_intercept:
        blocks.append(np.ones((n_samples, 1), dtype=np.float64))

    for lag in range(1, n_lags + 1):
        past = phases[n_lags - lag : n_time - lag]

        lag_block = np.empty(
            (n_samples, 2 * n_channels),
            dtype=np.float64,
        )
        lag_block[:, 0::2] = np.cos(past)
        lag_block[:, 1::2] = np.sin(past)

        blocks.append(lag_block)

    X = np.concatenate(blocks, axis=1)
    Y = phases[n_lags:]
    time_indices = np.arange(n_lags, n_time, dtype=np.int64)

    return X, Y, time_indices


# ============================================================
# 1チャネルAR-TG
# ============================================================

@dataclass
class ChannelARTG:
    beta_cos: FloatArray
    beta_sin: FloatArray
    success: bool
    message: str
    n_iter: int
    final_mean_nll: float
    gradient_inf_norm: float

    def natural_parameters(
        self,
        X: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        eta_cos = X @ self.beta_cos
        eta_sin = X @ self.beta_sin
        return eta_cos, eta_sin

    def predict_parameters(
        self,
        X: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """
        条件付きvon Mises分布の平均方向muと集中度kappaを返す。
        """
        eta_cos, eta_sin = self.natural_parameters(X)

        mu = np.arctan2(eta_sin, eta_cos)
        kappa = np.hypot(eta_cos, eta_sin)

        return mu, kappa

    def coefficient_norm(self) -> float:
        return float(
            np.sqrt(
                np.sum(self.beta_cos**2)
                + np.sum(self.beta_sin**2)
            )
        )


def fit_artg_channel(
    X: FloatArray,
    y: FloatArray,
    l2_penalty: float = 1e-3,
    penalize_intercept: bool = False,
    max_iter: int = 5000,
    ftol: float = 1e-10,
    gtol: float = 1e-6,
) -> ChannelARTG:
    """
    1つのターゲットチャネルにAR-TGを適合する。

    条件付き分布:
        p(y_t | H)
        ∝ exp{
            eta_cos,t cos(y_t)
            + eta_sin,t sin(y_t)
          }

    eta_cos,t = X_t beta_cos
    eta_sin,t = X_t beta_sin

    Xには各入力位相のcosとsinが入るため、指数部を展開すると、
    各ターゲット・入力・ラグについて次の4特徴量が含まれる。

        cos(y_t) cos(x_{t-lag})
        cos(y_t) sin(x_{t-lag})
        sin(y_t) cos(x_{t-lag})
        sin(y_t) sin(x_{t-lag})

    l2_penalty=0なら無正則化最尤推定。
    正値ならpenalized maximum likelihood。
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have inconsistent sample sizes.")

    if l2_penalty < 0:
        raise ValueError("l2_penalty must be nonnegative.")

    n_samples, n_features = X.shape

    cos_y = np.cos(y)
    sin_y = np.sin(y)

    penalty_mask = np.ones(n_features, dtype=np.float64)
    if not penalize_intercept:
        penalty_mask[0] = 0.0

    def unpack(
        theta: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        return theta[:n_features], theta[n_features:]

    def objective_and_gradient(
        theta: FloatArray,
    ) -> tuple[float, FloatArray]:
        beta_cos, beta_sin = unpack(theta)

        eta_cos = X @ beta_cos
        eta_sin = X @ beta_sin
        kappa = np.hypot(eta_cos, eta_sin)

        nll_terms = (
            np.log(2.0 * np.pi)
            + log_i0(kappa)
            - eta_cos * cos_y
            - eta_sin * sin_y
        )

        # サンプル数への依存を弱くするため平均NLLを最適化
        mean_nll = float(np.mean(nll_terms))

        if l2_penalty > 0:
            penalty = 0.5 * l2_penalty * (
                np.sum(penalty_mask * beta_cos**2)
                + np.sum(penalty_mask * beta_sin**2)
            )
            mean_nll += float(penalty)

        # d log I0(kappa) / d eta
        # = A1(kappa) eta / kappa
        ratio = np.empty_like(kappa)

        nonzero = kappa > 1e-10
        ratio[nonzero] = (
            a1_ratio(kappa[nonzero]) / kappa[nonzero]
        )
        ratio[~nonzero] = 0.5

        grad_eta_cos = ratio * eta_cos - cos_y
        grad_eta_sin = ratio * eta_sin - sin_y

        grad_beta_cos = (X.T @ grad_eta_cos) / n_samples
        grad_beta_sin = (X.T @ grad_eta_sin) / n_samples

        if l2_penalty > 0:
            grad_beta_cos += (
                l2_penalty * penalty_mask * beta_cos
            )
            grad_beta_sin += (
                l2_penalty * penalty_mask * beta_sin
            )

        gradient = np.concatenate(
            [grad_beta_cos, grad_beta_sin]
        )

        return mean_nll, gradient

    # 周辺von Mises分布に基づく切片初期値
    mean_vector = np.mean(np.exp(1j * y))
    mu0 = float(np.angle(mean_vector))
    r0 = float(np.abs(mean_vector))
    kappa0 = resultant_length_to_kappa(r0)

    initial = np.zeros(2 * n_features, dtype=np.float64)
    initial[0] = kappa0 * np.cos(mu0)
    initial[n_features] = kappa0 * np.sin(mu0)

    result = minimize(
        fun=objective_and_gradient,
        x0=initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iter,
            "ftol": ftol,
            "gtol": gtol,
            "maxls": 50,
        },
    )

    beta_cos, beta_sin = unpack(result.x)

    gradient_inf_norm = (
        np.nan
        if result.jac is None
        else float(np.max(np.abs(result.jac)))
    )

    return ChannelARTG(
        beta_cos=beta_cos,
        beta_sin=beta_sin,
        success=bool(result.success),
        message=str(result.message),
        n_iter=int(result.nit),
        final_mean_nll=float(result.fun),
        gradient_inf_norm=gradient_inf_norm,
    )


# ============================================================
# 多チャネルAR-TG
# ============================================================

@dataclass
class MultivariateARTG:
    channel_models: list[ChannelARTG]
    n_lags: int
    n_channels: int

    def predict_parameters(
        self,
        X: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        n_samples = X.shape[0]

        mu = np.empty(
            (n_samples, self.n_channels),
            dtype=np.float64,
        )
        kappa = np.empty_like(mu)

        for j, model in enumerate(self.channel_models):
            mu[:, j], kappa[:, j] = model.predict_parameters(X)

        return mu, kappa


def fit_multivariate_artg(
    X: FloatArray,
    Y: FloatArray,
    n_lags: int,
    l2_penalty: float = 1e-3,
    max_iter: int = 5000,
) -> MultivariateARTG:
    """
    各ターゲットチャネルの条件付きvon Mises回帰を個別に推定する。
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if Y.ndim != 2:
        raise ValueError(
            "Y must have shape (n_samples, n_channels)."
        )

    n_channels = Y.shape[1]
    models: list[ChannelARTG] = []

    for j in range(n_channels):
        print(f"Fitting channel {j + 1}/{n_channels} ...")

        model = fit_artg_channel(
            X=X,
            y=Y[:, j],
            l2_penalty=l2_penalty,
            max_iter=max_iter,
        )

        models.append(model)

        print(
            f"  success={model.success}, "
            f"iterations={model.n_iter}, "
            f"mean NLL={model.final_mean_nll:.6f}, "
            f"|grad|_inf={model.gradient_inf_norm:.3e}, "
            f"|beta|={model.coefficient_norm():.3e}"
        )

        if not model.success:
            print(f"  optimizer message: {model.message}")

    return MultivariateARTG(
        channel_models=models,
        n_lags=n_lags,
        n_channels=n_channels,
    )


# ============================================================
# 角度残差
# ============================================================

def compute_angular_residuals(
    observed_phases: FloatArray,
    predicted_mu: FloatArray,
) -> FloatArray:
    """
    AR-TGの条件付き平均方向に対する角度残差。

        residual = wrap(observed - predicted_mu)

    戻り値は [-pi, pi)。
    """
    observed_phases = np.asarray(
        observed_phases,
        dtype=np.float64,
    )
    predicted_mu = np.asarray(
        predicted_mu,
        dtype=np.float64,
    )

    if observed_phases.shape != predicted_mu.shape:
        raise ValueError(
            "observed_phases and predicted_mu "
            "must have the same shape."
        )

    return wrap_angle(observed_phases - predicted_mu)


# ============================================================
# Expanding-window cross-fitting
# ============================================================

@dataclass
class CrossFitResult:
    residuals: FloatArray
    mu: FloatArray
    kappa: FloatArray
    observed: FloatArray
    time_indices: IntArray
    fold_index: IntArray


def expanding_window_cross_fit(
    X: FloatArray,
    Y: FloatArray,
    time_indices: IntArray,
    n_lags: int,
    initial_train_fraction: float = 0.4,
    n_folds: int = 4,
    l2_penalty: float = 1e-3,
    max_iter: int = 5000,
) -> CrossFitResult:
    """
    Expanding-window方式でout-of-sample残差を作る。

    最初のinitial_train_fractionは学習専用。
    それ以降をn_folds個の連続検証ブロックに分割する。

    各foldでは、その検証ブロックより前の全観測だけを学習に使う。
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    time_indices = np.asarray(time_indices, dtype=np.int64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional.")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same sample size.")

    if time_indices.shape != (X.shape[0],):
        raise ValueError(
            "time_indices must have shape (n_samples,)."
        )

    if not 0.0 < initial_train_fraction < 1.0:
        raise ValueError(
            "initial_train_fraction must lie between 0 and 1."
        )

    if n_folds < 1:
        raise ValueError("n_folds must be positive.")

    n_samples = X.shape[0]
    initial_end = int(
        np.floor(n_samples * initial_train_fraction)
    )

    if initial_end < max(10, X.shape[1] // 2):
        print(
            "Warning: initial training interval may be small "
            "relative to the number of predictors."
        )

    validation_indices = np.arange(initial_end, n_samples)
    validation_blocks = np.array_split(
        validation_indices,
        n_folds,
    )

    all_residuals: list[FloatArray] = []
    all_mu: list[FloatArray] = []
    all_kappa: list[FloatArray] = []
    all_observed: list[FloatArray] = []
    all_times: list[IntArray] = []
    all_folds: list[IntArray] = []

    for fold, validation_index in enumerate(validation_blocks):
        if validation_index.size == 0:
            continue

        validation_start = int(validation_index[0])
        train_index = np.arange(validation_start)

        print("\n" + "=" * 70)
        print(
            f"Cross-fit fold {fold + 1}/{n_folds}: "
            f"train={train_index.size}, "
            f"validation={validation_index.size}"
        )
        print("=" * 70)

        fold_model = fit_multivariate_artg(
            X=X[train_index],
            Y=Y[train_index],
            n_lags=n_lags,
            l2_penalty=l2_penalty,
            max_iter=max_iter,
        )

        mu_fold, kappa_fold = fold_model.predict_parameters(
            X[validation_index]
        )

        observed_fold = Y[validation_index]

        residual_fold = compute_angular_residuals(
            observed_phases=observed_fold,
            predicted_mu=mu_fold,
        )

        all_residuals.append(residual_fold)
        all_mu.append(mu_fold)
        all_kappa.append(kappa_fold)
        all_observed.append(observed_fold)
        all_times.append(time_indices[validation_index])
        all_folds.append(
            np.full(
                validation_index.size,
                fold,
                dtype=np.int64,
            )
        )

    if not all_residuals:
        raise RuntimeError(
            "No cross-fitted observations were generated."
        )

    return CrossFitResult(
        residuals=np.concatenate(all_residuals, axis=0),
        mu=np.concatenate(all_mu, axis=0),
        kappa=np.concatenate(all_kappa, axis=0),
        observed=np.concatenate(all_observed, axis=0),
        time_indices=np.concatenate(all_times, axis=0),
        fold_index=np.concatenate(all_folds, axis=0),
    )


# ============================================================
# κ診断
# ============================================================

def compare_predicted_and_empirical_kappa(
    residuals: FloatArray,
    predicted_kappa: FloatArray,
    channel_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    予測kappaと、全残差から推定した経験的kappaを比較する。

    経験的kappaは時間変動するkappaを1値に要約しているため、
    完全一致する必要はない。
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    predicted_kappa = np.asarray(
        predicted_kappa,
        dtype=np.float64,
    )

    if residuals.shape != predicted_kappa.shape:
        raise ValueError(
            "residuals and predicted_kappa must have the same shape."
        )

    n_channels = residuals.shape[1]

    if channel_names is None:
        channel_names = [
            f"Ch {j + 1}" for j in range(n_channels)
        ]

    rows: list[dict[str, float | str]] = []

    print("\nKappa diagnostics")

    for j in range(n_channels):
        rj = residuals[:, j]
        kj = predicted_kappa[:, j]

        resultant = float(
            np.abs(np.mean(np.exp(1j * rj)))
        )
        empirical_kappa = resultant_length_to_kappa(resultant)

        circular_rmse = float(np.sqrt(np.mean(rj**2)))
        mean_abs_error = float(np.mean(np.abs(rj)))

        standardized = np.sqrt(kj) * rj

        row = {
            "channel": channel_names[j],
            "kappa_min": float(np.min(kj)),
            "kappa_q1": float(np.quantile(kj, 0.25)),
            "kappa_median": float(np.median(kj)),
            "kappa_q3": float(np.quantile(kj, 0.75)),
            "kappa_max": float(np.max(kj)),
            "kappa_empirical": float(empirical_kappa),
            "residual_rmse_rad": circular_rmse,
            "residual_rmse_deg": float(
                np.degrees(circular_rmse)
            ),
            "residual_mae_rad": mean_abs_error,
            "residual_mae_deg": float(
                np.degrees(mean_abs_error)
            ),
            "standardized_mean": float(
                np.mean(standardized)
            ),
            "standardized_std": float(
                np.std(standardized, ddof=1)
            ),
        }
        rows.append(row)

        print(
            f"{channel_names[j]}: "
            f"kappa median={row['kappa_median']:.2f}, "
            f"range=[{row['kappa_min']:.2f}, "
            f"{row['kappa_max']:.2f}], "
            f"kappa empirical={empirical_kappa:.2f}, "
            f"RMSE={row['residual_rmse_deg']:.2f} deg, "
            f"std(sqrt(kappa)*r)="
            f"{row['standardized_std']:.3f}"
        )

    return pd.DataFrame(rows)


# ============================================================
# κ分位層
# ============================================================

def assign_equal_count_bins(
    values: FloatArray,
    n_bins: int = 3,
) -> IntArray:
    """
    順位に基づき、観測数がほぼ等しくなるビンへ分ける。
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    if not np.all(np.isfinite(values)):
        raise ValueError("values contains non-finite entries.")

    n = values.size

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)

    bin_index = np.floor(
        n_bins * ranks / n
    ).astype(np.int64)

    return np.minimum(bin_index, n_bins - 1)


def pairwise_kappa_score(
    kappa_j: FloatArray,
    kappa_k: FloatArray,
    method: KappaScoreMethod = "mean_log",
) -> FloatArray:
    """
    2チャネルのkappaから共同集中度スコアを作る。

    mean_log:
        0.5 * [log(1+kappa_j) + log(1+kappa_k)]

    geometric:
        sqrt(kappa_j * kappa_k)

    minimum:
        min(kappa_j, kappa_k)

    mean:
        0.5 * (kappa_j + kappa_k)
    """
    kappa_j = np.asarray(kappa_j, dtype=np.float64)
    kappa_k = np.asarray(kappa_k, dtype=np.float64)

    if method == "mean_log":
        return (
            np.log1p(kappa_j)
            + np.log1p(kappa_k)
        ) / 2.0

    if method == "geometric":
        return np.sqrt(
            np.maximum(kappa_j, 0.0)
            * np.maximum(kappa_k, 0.0)
        )

    if method == "minimum":
        return np.minimum(kappa_j, kappa_k)

    if method == "mean":
        return (kappa_j + kappa_k) / 2.0

    raise ValueError(f"Unknown kappa score method: {method}")


def quantile_labels(n_quantiles: int) -> list[str]:
    if n_quantiles == 2:
        return ["low", "high"]

    if n_quantiles == 3:
        return ["low", "middle", "high"]

    return [f"Q{q + 1}" for q in range(n_quantiles)]


# ============================================================
# プロット用共通設定
# ============================================================

ANGLE_TICKS = [
    -np.pi,
    -np.pi / 2.0,
    0.0,
    np.pi / 2.0,
    np.pi,
]

ANGLE_TICK_LABELS = [
    r"$-\pi$",
    r"$-\pi/2$",
    r"$0$",
    r"$\pi/2$",
    r"$\pi$",
]


def configure_joint_angle_axes(
    grid: sns.axisgrid.JointGrid,
    x_label: str,
    y_label: str,
) -> None:
    grid.ax_joint.set_xlim(-np.pi, np.pi)
    grid.ax_joint.set_ylim(-np.pi, np.pi)
    grid.ax_joint.set_aspect("equal", adjustable="box")

    grid.ax_joint.set_xticks(ANGLE_TICKS)
    grid.ax_joint.set_yticks(ANGLE_TICKS)
    grid.ax_joint.set_xticklabels(ANGLE_TICK_LABELS)
    grid.ax_joint.set_yticklabels(ANGLE_TICK_LABELS)

    grid.ax_joint.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
    )

    grid.set_axis_labels(x_label, y_label)


# ============================================================
# 層別なしのペアワイズ残差図
# ============================================================

def plot_pairwise_residuals_unstratified(
    residuals: FloatArray,
    pairs: Iterable[tuple[int, int]] | None = None,
    channel_names: list[str] | None = None,
    max_pairs: int | None = 12,
    marginal_bins: int = 30,
    point_size: float = 18.0,
    alpha: float = 0.50,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    全cross-fitted角度残差を使ったscatter + marginal histogram。
    """
    residuals = np.asarray(residuals, dtype=np.float64)

    if residuals.ndim != 2:
        raise ValueError(
            "residuals must have shape (n_samples, n_channels)."
        )

    finite_rows = np.all(np.isfinite(residuals), axis=1)
    residuals = residuals[finite_rows]

    n_samples, n_channels = residuals.shape

    if channel_names is None:
        channel_names = [
            f"Ch {j + 1}" for j in range(n_channels)
        ]

    if pairs is None:
        pair_list = list(
            combinations(range(n_channels), 2)
        )
        if max_pairs is not None:
            pair_list = pair_list[:max_pairs]
    else:
        pair_list = list(pairs)

    save_dir = None
    if output_dir is not None:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white")

    for j, k in pair_list:
        grid = sns.jointplot(
            x=residuals[:, j],
            y=residuals[:, k],
            kind="scatter",
            height=6.2,
            ratio=4,
            space=0.08,
            joint_kws={
                "s": point_size,
                "alpha": alpha,
                "linewidth": 0,
                "rasterized": True,
            },
            marginal_kws={
                "bins": marginal_bins,
                "binrange": (-np.pi, np.pi),
                "stat": "density",
                "fill": True,
            },
        )

        configure_joint_angle_axes(
            grid,
            x_label=f"Angular residual: {channel_names[j]}",
            y_label=f"Angular residual: {channel_names[k]}",
        )

        grid.figure.suptitle(
            (
                "Cross-fitted AR-TG angular residuals\n"
                f"{channel_names[j]} vs {channel_names[k]}, "
                f"n={n_samples:,}"
            ),
            y=1.03,
        )
        grid.figure.tight_layout()

        if save_dir is not None:
            grid.figure.savefig(
                save_dir
                / (
                    f"residual_unstratified_"
                    f"ch{j + 1:02d}_ch{k + 1:02d}.png"
                ),
                dpi=180,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close(grid.figure)


# ============================================================
# 共同κによる層別ペアワイズ図
# ============================================================

def plot_pairwise_residuals_by_joint_kappa(
    residuals: FloatArray,
    kappa: FloatArray,
    pairs: Iterable[tuple[int, int]] | None = None,
    channel_names: list[str] | None = None,
    n_quantiles: int = 3,
    kappa_score_method: KappaScoreMethod = "mean_log",
    max_pairs: int | None = 12,
    minimum_points: int = 50,
    marginal_bins: int = 25,
    point_size: float = 18.0,
    alpha: float = 0.50,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    ペアごとの共同kappaスコアを等数分位層へ分け、
    各層で角度残差のjointplotを描く。

    個別チャネルのlow-low交差ではなく、
    ペアごとに1つの共同kappaスコアを作るため、
    各層の観測数はほぼ等しくなる。
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)

    if residuals.ndim != 2:
        raise ValueError(
            "residuals must have shape (n_samples, n_channels)."
        )

    if residuals.shape != kappa.shape:
        raise ValueError(
            "residuals and kappa must have the same shape."
        )

    finite_rows = (
        np.all(np.isfinite(residuals), axis=1)
        & np.all(np.isfinite(kappa), axis=1)
    )

    residuals = residuals[finite_rows]
    kappa = kappa[finite_rows]

    _, n_channels = residuals.shape

    if channel_names is None:
        channel_names = [
            f"Ch {j + 1}" for j in range(n_channels)
        ]

    if pairs is None:
        pair_list = list(
            combinations(range(n_channels), 2)
        )
        if max_pairs is not None:
            pair_list = pair_list[:max_pairs]
    else:
        pair_list = list(pairs)

    labels = quantile_labels(n_quantiles)

    save_dir = None
    if output_dir is not None:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white")

    for j, k in pair_list:
        joint_score = pairwise_kappa_score(
            kappa_j=kappa[:, j],
            kappa_k=kappa[:, k],
            method=kappa_score_method,
        )

        joint_bin = assign_equal_count_bins(
            joint_score,
            n_bins=n_quantiles,
        )

        score_edges = np.quantile(
            joint_score,
            np.linspace(0.0, 1.0, n_quantiles + 1),
        )

        for q in range(n_quantiles):
            selected_indices = np.flatnonzero(
                joint_bin == q
            )
            n_selected = selected_indices.size

            if n_selected < minimum_points:
                print(
                    f"Skipping {channel_names[j]} vs "
                    f"{channel_names[k]}, {labels[q]}: "
                    f"only {n_selected} points."
                )
                continue

            x = residuals[selected_indices, j]
            y = residuals[selected_indices, k]

            selected_kappa_j = kappa[selected_indices, j]
            selected_kappa_k = kappa[selected_indices, k]

            grid = sns.jointplot(
                x=x,
                y=y,
                kind="scatter",
                height=6.2,
                ratio=4,
                space=0.08,
                joint_kws={
                    "s": point_size,
                    "alpha": alpha,
                    "linewidth": 0,
                    "rasterized": True,
                },
                marginal_kws={
                    "bins": marginal_bins,
                    "binrange": (-np.pi, np.pi),
                    "stat": "density",
                    "fill": True,
                },
            )

            configure_joint_angle_axes(
                grid,
                x_label=(
                    f"Angular residual: {channel_names[j]}"
                ),
                y_label=(
                    f"Angular residual: {channel_names[k]}"
                ),
            )

            grid.figure.suptitle(
                (
                    "Cross-fitted AR-TG angular residuals\n"
                    f"{channel_names[j]} vs {channel_names[k]}\n"
                    f"joint kappa={labels[q]}, "
                    f"n={n_selected:,}, "
                    f"score=[{score_edges[q]:.3g}, "
                    f"{score_edges[q + 1]:.3g}]\n"
                    f"median kappa="
                    f"({np.median(selected_kappa_j):.2f}, "
                    f"{np.median(selected_kappa_k):.2f})"
                ),
                y=1.09,
            )
            grid.figure.tight_layout()

            if save_dir is not None:
                grid.figure.savefig(
                    save_dir
                    / (
                        f"residual_joint_kappa_"
                        f"ch{j + 1:02d}_ch{k + 1:02d}_"
                        f"{labels[q]}.png"
                    ),
                    dpi=180,
                    bbox_inches="tight",
                )

            if show:
                plt.show()
            else:
                plt.close(grid.figure)


# ============================================================
# κ層を同一図に重ねる
# ============================================================
def plot_residual_pairs_with_kappa_hue(
    residuals: FloatArray,
    kappa: FloatArray,
    pairs: Iterable[tuple[int, int]],
    channel_names: list[str],
    n_quantiles: int = 3,
    kappa_score_method: KappaScoreMethod = "mean_log",
    marginal_bins: int = 25,
    point_size: float = 18.0,
    alpha: float = 0.50,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    全観測を同じ図に描き、共同kappa層をhueで区別する。

    seaborn.jointplotでは、hue使用時に周辺分布がkdeplotへ
    切り替わるバージョンがあり、marginal_kwsのbinsが
    Line2Dへ誤って渡されることがある。

    そのためJointGrid上へscatterplotとhistplotを
    明示的に描画する。
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)

    if residuals.ndim != 2:
        raise ValueError(
            "residuals must have shape (n_samples, n_channels)."
        )

    if residuals.shape != kappa.shape:
        raise ValueError(
            "residuals and kappa must have the same shape."
        )

    finite_rows = (
        np.all(np.isfinite(residuals), axis=1)
        & np.all(np.isfinite(kappa), axis=1)
    )

    residuals = residuals[finite_rows]
    kappa = kappa[finite_rows]

    if len(channel_names) != residuals.shape[1]:
        raise ValueError(
            "channel_names must match the number of channels."
        )

    labels = np.asarray(
        quantile_labels(n_quantiles),
        dtype=object,
    )

    save_dir: Path | None = None

    if output_dir is not None:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white")

    for j, k in pairs:
        if not (
            0 <= j < residuals.shape[1]
            and 0 <= k < residuals.shape[1]
            and j != k
        ):
            raise ValueError(f"Invalid channel pair: {(j, k)}")

        joint_score = pairwise_kappa_score(
            kappa[:, j],
            kappa[:, k],
            method=kappa_score_method,
        )

        bin_index = assign_equal_count_bins(
            joint_score,
            n_bins=n_quantiles,
        )

        stratum = pd.Categorical(
            labels[bin_index],
            categories=list(labels),
            ordered=True,
        )

        frame = pd.DataFrame(
            {
                "residual_x": residuals[:, j],
                "residual_y": residuals[:, k],
                "kappa_stratum": stratum,
            }
        )

        grid = sns.JointGrid(
            data=frame,
            x="residual_x",
            y="residual_y",
            height=6.5,
            ratio=4,
            space=0.08,
            xlim=(-np.pi, np.pi),
            ylim=(-np.pi, np.pi),
        )

        # 中央散布図
        sns.scatterplot(
            data=frame,
            x="residual_x",
            y="residual_y",
            hue="kappa_stratum",
            hue_order=list(labels),
            s=point_size,
            alpha=alpha,
            linewidth=0,
            ax=grid.ax_joint,
        )

        # 上側の周辺ヒストグラム
        sns.histplot(
            data=frame,
            x="residual_x",
            hue="kappa_stratum",
            hue_order=list(labels),
            bins=marginal_bins,
            binrange=(-np.pi, np.pi),
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            legend=False,
            ax=grid.ax_marg_x,
        )

        # 右側の周辺ヒストグラム
        sns.histplot(
            data=frame,
            y="residual_y",
            hue="kappa_stratum",
            hue_order=list(labels),
            bins=marginal_bins,
            binrange=(-np.pi, np.pi),
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            legend=False,
            ax=grid.ax_marg_y,
        )

        configure_joint_angle_axes(
            grid,
            x_label=f"Angular residual: {channel_names[j]}",
            y_label=f"Angular residual: {channel_names[k]}",
        )

        # # joint軸の凡例を図の外側へ移動
        # legend = grid.ax_joint.get_legend()

        # if legend is not None:
        #     legend.set_title("Joint kappa")
        #     legend.set_bbox_to_anchor((1.03, 1.0))
        #     legend.set_loc("upper left")

        grid.figure.suptitle(
            (
                "Cross-fitted AR-TG angular residuals\n"
                f"{channel_names[j]} vs {channel_names[k]}, "
                f"n={residuals.shape[0]:,}"
            ),
            y=1.03,
        )

        grid.figure.tight_layout()

        if save_dir is not None:
            grid.figure.savefig(
                save_dir
                / (
                    f"residual_kappa_hue_"
                    f"ch{j + 1:02d}_ch{k + 1:02d}.png"
                ),
                dpi=180,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close(grid.figure)

# ============================================================
# 標準化残差診断
# ============================================================

def plot_standardized_residual_histograms(
    residuals: FloatArray,
    kappa: FloatArray,
    channel_names: list[str],
    output_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    kappaが十分大きいときの近似
        z = sqrt(kappa) * residual
    をチャネルごとに描く。

    モデルが妥当なら概ね平均0、標準偏差1になることが期待される。
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)

    save_dir = None
    if output_dir is not None:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white")

    for j, name in enumerate(channel_names):
        z = np.sqrt(kappa[:, j]) * residuals[:, j]

        fig, ax = plt.subplots(figsize=(6.0, 4.2))

        sns.histplot(
            z,
            bins=40,
            stat="density",
            ax=ax,
        )

        x_grid = np.linspace(-5.0, 5.0, 400)
        normal_density = (
            np.exp(-0.5 * x_grid**2)
            / np.sqrt(2.0 * np.pi)
        )

        ax.plot(
            x_grid,
            normal_density,
            linestyle="--",
            linewidth=1.5,
            label="N(0, 1)",
        )

        ax.set_xlim(-5.0, 5.0)
        ax.set_xlabel(
            r"$\sqrt{\widehat{\kappa}_t}\,r_t$"
        )
        ax.set_ylabel("Density")
        ax.set_title(
            f"Standardized angular residual: {name}\n"
            f"mean={np.mean(z):.3f}, "
            f"std={np.std(z, ddof=1):.3f}"
        )
        ax.legend()

        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(
                save_dir
                / f"standardized_residual_ch{j + 1:02d}.png",
                dpi=180,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close(fig)


# ============================================================
# 実験全体
# ============================================================

def run_experiment(
    phases: FloatArray,
    n_lags: int = 5,
    initial_train_fraction: float = 0.4,
    n_crossfit_folds: int = 4,
    l2_penalty: float = 1e-3,
    max_iter: int = 5000,
    selected_pairs: list[tuple[int, int]] | None = None,
    channel_names: list[str] | None = None,
    n_kappa_quantiles: int = 3,
    kappa_score_method: KappaScoreMethod = "mean_log",
    max_pairs: int = 12,
    minimum_points: int = 50,
    output_dir: str | Path = "artg_crossfit_analysis",
    show_plots: bool = True,
) -> dict[str, object]:
    """
    実験全体:

    1. ラグ特徴量作成
    2. Expanding-window cross-fitting
    3. out-of-sample角度残差作成
    4. kappa診断
    5. 層別なしペアワイズjointplot
    6. 共同kappa分位層ごとのjointplot
    7. kappa層をhueで重ねたjointplot
    8. 標準化角度残差診断
    9. 数値結果保存
    """
    phases = np.asarray(phases, dtype=np.float64)

    if phases.ndim != 2:
        raise ValueError("phases must have shape (T, d).")

    if not np.all(np.isfinite(phases)):
        raise ValueError("phases contains NaN or infinite values.")

    phases = wrap_angle(phases)

    n_time, n_channels = phases.shape

    if channel_names is None:
        channel_names = [
            f"Ch {j + 1}" for j in range(n_channels)
        ]

    if len(channel_names) != n_channels:
        raise ValueError(
            "channel_names must match the number of channels."
        )

    if selected_pairs is None:
        selected_pairs = list(
            combinations(range(n_channels), 2)
        )[:max_pairs]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AR-TG cross-fitted residual experiment")
    print("=" * 70)
    print(f"Original observations: {n_time}")
    print(f"Channels: {n_channels}")
    print(f"Lags: {n_lags}")
    print(f"Initial training fraction: {initial_train_fraction}")
    print(f"Cross-fit folds: {n_crossfit_folds}")
    print(f"L2 penalty: {l2_penalty}")
    print(f"Selected pairs: {selected_pairs}")
    print("=" * 70)

    X, Y, time_indices = make_lagged_design(
        phases=phases,
        n_lags=n_lags,
        include_intercept=True,
    )

    print(f"Lagged observations: {X.shape[0]}")
    print(f"Number of predictors: {X.shape[1]}")

    cross_fit = expanding_window_cross_fit(
        X=X,
        Y=Y,
        time_indices=time_indices,
        n_lags=n_lags,
        initial_train_fraction=initial_train_fraction,
        n_folds=n_crossfit_folds,
        l2_penalty=l2_penalty,
        max_iter=max_iter,
    )

    print("\n" + "=" * 70)
    print(
        "Number of cross-fitted observations: "
        f"{cross_fit.residuals.shape[0]}"
    )
    print("=" * 70)

    diagnostics = compare_predicted_and_empirical_kappa(
        residuals=cross_fit.residuals,
        predicted_kappa=cross_fit.kappa,
        channel_names=channel_names,
    )

    # 数値保存
    np.save(
        output_path / "crossfit_residuals.npy",
        cross_fit.residuals,
    )
    np.save(
        output_path / "crossfit_mu.npy",
        cross_fit.mu,
    )
    np.save(
        output_path / "crossfit_kappa.npy",
        cross_fit.kappa,
    )
    np.save(
        output_path / "crossfit_observed.npy",
        cross_fit.observed,
    )
    np.save(
        output_path / "crossfit_time_indices.npy",
        cross_fit.time_indices,
    )
    np.save(
        output_path / "crossfit_fold_index.npy",
        cross_fit.fold_index,
    )

    diagnostics.to_csv(
        output_path / "kappa_diagnostics.csv",
        index=False,
    )

    # 層別なし
    plot_pairwise_residuals_unstratified(
        residuals=cross_fit.residuals,
        pairs=selected_pairs,
        channel_names=channel_names,
        point_size=18.0,
        alpha=0.50,
        output_dir=output_path / "unstratified_jointplots",
        show=show_plots,
    )

    # 共同kappa分位層ごとの別図
    plot_pairwise_residuals_by_joint_kappa(
        residuals=cross_fit.residuals,
        kappa=cross_fit.kappa,
        pairs=selected_pairs,
        channel_names=channel_names,
        n_quantiles=n_kappa_quantiles,
        kappa_score_method=kappa_score_method,
        minimum_points=minimum_points,
        point_size=18.0,
        alpha=0.50,
        output_dir=(
            output_path / "joint_kappa_stratified_jointplots"
        ),
        show=show_plots,
    )

    # kappa層を同一図へ重ねる
    plot_residual_pairs_with_kappa_hue(
        residuals=cross_fit.residuals,
        kappa=cross_fit.kappa,
        pairs=selected_pairs,
        channel_names=channel_names,
        n_quantiles=n_kappa_quantiles,
        kappa_score_method=kappa_score_method,
        point_size=18.0,
        alpha=0.50,
        output_dir=output_path / "joint_kappa_hue_jointplots",
        show=show_plots,
    )

    # 標準化残差
    plot_standardized_residual_histograms(
        residuals=cross_fit.residuals,
        kappa=cross_fit.kappa,
        channel_names=channel_names,
        output_dir=output_path / "standardized_residuals",
        show=show_plots,
    )

    return {
        "X": X,
        "Y": Y,
        "time_indices": time_indices,
        "cross_fit": cross_fit,
        "diagnostics": diagnostics,
        "channel_names": channel_names,
        "selected_pairs": selected_pairs,
    }


# ============================================================
# 実行例
# ============================================================

if __name__ == "__main__":
    from data_real import (
        load_marmoset_ecog,
        extract_feature_matrix,
        FeatureSpec,
    )

    def ecog_case3(StimIndex):
        from scipy.io import loadmat
        # DIR_PATH = "../data/riken-auditory-ECoG/Ji20181207S4c/"
        DIR_PATH = "../data/riken-auditory-ECoG/Rc20181219S8c/"
        EVENT_PATH = DIR_PATH + "Event.mat"
        mat = loadmat(EVENT_PATH)
        StimOn = mat["StimOn"].flatten()
        target_start = StimOn[StimIndex] + 850 #ITI
        target_end = StimOn[StimIndex+1]
        print("抽出された秒(ms): ", target_start, "~", target_end)
        print("提示されたtone: ", mat["allTrialIdx"][0, StimIndex], mat["allTrialIdx"][0,StimIndex+1])
        dataset = load_marmoset_ecog(animal="Rc2", session_index=8, window=slice(target_start, target_end))
        phase = extract_feature_matrix(
            dataset,
            FeatureSpec(name="phase", feature="phase", band=(12, 25)),
            trials=[0],
        )
        return phase

    selected_20_with_pfc = [7, 8, 9, 19, 20, 27, 28, 29, 35, 36, 37,
                        1, 2, 5, 6, 16, 21,
                        55, 61, 63] #JiとRcは共有でOK. 主にAuditory領域を抽出。
    raw = ecog_case3(StimIndex=100)[:, [x-1 for x in selected_20_with_pfc]] #electrodes selection.


    phases = raw

    if phases.ndim != 2:
        raise ValueError(
            "phases.npy must have shape (T, d)."
        )

    channel_names = [
        f"Ch {j + 1}"
        for j in range(phases.shape[1])
    ]

    # 0-indexed:
    # Ch1 vs Ch2 -> (0, 1)
    selected_pairs = [
        (0, 1),
        (0, 2),
        (1, 2),
    ]

    # 全ペアの先頭12組を使うなら:
    # selected_pairs = None

    results = run_experiment(
        phases=phases,
        n_lags=1,

        # n≈1800なら、最初の40%を学習専用とし、
        # 残り約60%からcross-fitted残差を作る。
        initial_train_fraction=0.4,
        n_crossfit_folds=4,

        # 0.0なら無正則化MLE。
        # 収束不安定・kappa過大なら1e-2や1e-1も比較する。
        l2_penalty=1e-3,
        max_iter=5000,

        selected_pairs=selected_pairs,
        channel_names=channel_names,

        # 共同kappaをlow/middle/highへ分割
        n_kappa_quantiles=3,

        # "mean_log", "minimum", "geometric", "mean"
        kappa_score_method="mean_log",

        max_pairs=12,
        minimum_points=50,

        output_dir="artg_crossfit_analysis",
        show_plots=True,
    )