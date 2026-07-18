from __future__ import annotations

"""
Group elastic-net FISTAの出力を可視化・評価するスクリプト。

想定する入力ディレクトリ:
    OUTPUT_DIR/
        gpu_fista_results.csv
        information_criterion_ranking.csv      # 存在する場合
        best_model_by_plic.json                # 存在する場合
        lambdas.npy                            # 存在する場合
        theta/
            lambda_000_value_..._gpu0.npy
            lambda_001_value_..._gpu1.npy
            ...

FISTA側の特徴量配置:
    [lag][directed edge][cc, cs, sc, ss]

thetaの長さ:
    order * dim * dim * 4

このスクリプトでは各 directed edge (i, j) について、

    sqrt(
        sum over lag and basis of theta[lag, edge, basis]^2
    )

をエッジ強度として可視化する。

生成物:
    plots/
        heatmaps/
            lambda_XXX_raw.png
            lambda_XXX_binary.png
        plic_vs_active_groups.png
        plic_vs_lambda.png
        metrics_vs_lambda.png              # GT指定時
        precision_recall_f1_vs_edges.png    # GT指定時
        best_plic_raw.png
        best_plic_binary.png

    theta_summary.csv
    metrics_all.csv                         # GT指定時
    metric_all.txt                          # GT指定時
    metric_best.txt                         # GT指定時
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read group elastic-net FISTA CSV/NPY outputs "
            "and create diagnostic figures."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "gpu_fista_results.csvとtheta/が保存されている"
            "FISTA出力ディレクトリ。"
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        required=True,
        help="ノード数。",
    )
    parser.add_argument(
        "--order",
        type=int,
        required=True,
        help="ラグ次数。",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help=(
            "結果CSVを明示指定する。省略時は"
            "OUTPUT_DIR/gpu_fista_results.csv。"
        ),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help=(
            "Ground truth edgeファイル。"
            "各行を 'src: dst1 dst2 ...' とする。"
            "指定しない場合、評価指標は計算しない。"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-8,
        help="エッジをactiveと判定するgroup normの閾値。",
    )
    parser.add_argument(
        "--include-self-loops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="評価時に自己ループを候補へ含める。",
    )
    parser.add_argument(
        "--max-heatmaps",
        type=int,
        default=0,
        help=(
            "作成するlambda別ヒートマップ数の上限。"
            "0なら全lambdaを描画する。"
        ),
    )
    parser.add_argument(
        "--heatmap-sort",
        choices=[
            "lambda_index",
            "lambda_asc",
            "lambda_desc",
            "plic",
            "active_groups",
        ],
        default="lambda_index",
        help="ヒートマップを生成する順序。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存に加えて画面表示する。",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.dim <= 0:
        raise ValueError(
            f"--dim must be positive, got {args.dim}"
        )

    if args.order <= 0:
        raise ValueError(
            f"--order must be positive, got {args.order}"
        )

    if args.threshold < 0:
        raise ValueError(
            "--threshold must be nonnegative."
        )

    if args.max_heatmaps < 0:
        raise ValueError(
            "--max-heatmaps must be zero or positive."
        )

    if args.dpi <= 0:
        raise ValueError(
            "--dpi must be positive."
        )


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    output_dir = args.output_dir.expanduser().resolve()

    results_csv = (
        args.results_csv.expanduser().resolve()
        if args.results_csv is not None
        else output_dir / "gpu_fista_results.csv"
    )

    plot_dir = output_dir / "plots"
    heatmap_dir = plot_dir / "heatmaps"

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_dir}"
        )

    if not results_csv.exists():
        raise FileNotFoundError(
            f"Results CSV does not exist: {results_csv}"
        )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    heatmap_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir, results_csv, plot_dir


def normalize_result_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "lambda_index",
        "lambda_value",
        "theta_file",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Results CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()

    numeric_columns = [
        "lambda_index",
        "lambda_value",
        "plic",
        "active_groups",
        "final_objective",
        "final_loss",
        "final_penalty_unscaled",
        "final_ridge_penalty_unscaled",
        "theta_norm",
        "theta_maxabs",
    ]

    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(
                out[column],
                errors="coerce",
            )

    out["lambda_index"] = (
        out["lambda_index"]
        .astype("Int64")
    )

    if "error" in out.columns:
        has_error = (
            out["error"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )
    else:
        has_error = pd.Series(
            False,
            index=out.index,
        )

    has_theta = (
        out["theta_file"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    out["usable_theta"] = (
        ~has_error
        & has_theta
        & out["lambda_index"].notna()
    )

    return out


def resolve_theta_path(
    theta_file: Any,
    output_dir: Path,
) -> Path:
    raw = str(theta_file).strip()

    if not raw:
        raise ValueError(
            "theta_file is empty."
        )

    path = Path(raw).expanduser()

    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                output_dir / path,
                output_dir / "theta" / path.name,
                path.resolve(),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Theta file not found. Tried: "
        + ", ".join(str(x) for x in candidates)
    )


def theta_to_edge_components(
    theta: np.ndarray,
    dim: int,
    order: int,
) -> np.ndarray:
    """
    thetaを

        (order, dim, dim, 4)

    に変換する。

    特徴量配置:
        [lag][directed edge][cc, cs, sc, ss]
    """
    theta = np.asarray(
        theta,
        dtype=np.float64,
    ).reshape(-1)

    expected_size = (
        order
        * dim
        * dim
        * 4
    )

    if theta.size != expected_size:
        raise ValueError(
            f"theta has {theta.size} elements, "
            f"expected {expected_size} for "
            f"dim={dim}, order={order}."
        )

    return theta.reshape(
        order,
        dim,
        dim,
        4,
    )


def theta_to_edge_norms(
    theta: np.ndarray,
    dim: int,
    order: int,
) -> np.ndarray:
    """
    各エッジについて全lag・4基底をまとめたgroup normを計算する。

        norm[i, j]
        = sqrt(sum_{lag,basis} theta[lag,i,j,basis]^2)
    """
    components = theta_to_edge_components(
        theta=theta,
        dim=dim,
        order=order,
    )

    return np.linalg.norm(
        components,
        axis=(0, 3),
    )


def theta_to_lag_edge_norms(
    theta: np.ndarray,
    dim: int,
    order: int,
) -> np.ndarray:
    """
    lagごとに4基底をまとめたノルムを返す。

    shape:
        (order, dim, dim)
    """
    components = theta_to_edge_components(
        theta=theta,
        dim=dim,
        order=order,
    )

    return np.linalg.norm(
        components,
        axis=3,
    )


def make_axis_labels(
    dim: int,
) -> list[str]:
    return [
        f"x{i + 1}"
        for i in range(dim)
    ]


def save_or_show(
    figure: plt.Figure,
    path: Path,
    dpi: int,
    show: bool,
) -> None:
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(figure)


def plot_edge_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
    dpi: int,
    show: bool,
) -> None:
    """
    group normは非負なので、seismic/TwoSlopeNormではなく
    0始点の連続カラーマップを使用する。
    """
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    dim = matrix.shape[0]

    vmax = float(
        np.nanmax(matrix)
    ) if matrix.size else 0.0

    if not math.isfinite(vmax) or vmax <= 0:
        vmax = 1e-12

    figure, axis = plt.subplots(
        figsize=(
            max(5.0, 0.46 * dim + 2.0),
            max(4.5, 0.46 * dim + 1.5),
        )
    )

    image = axis.imshow(
        matrix,
        cmap="viridis",
        norm=Normalize(
            vmin=0.0,
            vmax=vmax,
        ),
        interpolation="nearest",
        aspect="equal",
    )

    labels = make_axis_labels(dim)

    axis.set_xticks(
        np.arange(dim)
    )
    axis.set_xticklabels(
        labels,
        rotation=90,
    )

    axis.set_yticks(
        np.arange(dim)
    )
    axis.set_yticklabels(
        labels
    )

    axis.xaxis.tick_top()
    axis.xaxis.set_label_position(
        "top"
    )

    axis.set_xlabel(
        "Source at past lags"
    )
    axis.set_ylabel(
        "Target at current time"
    )
    axis.set_title(
        title,
        fontsize=9,
        pad=18,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Group coefficient norm"
    )

    figure.tight_layout()

    save_or_show(
        figure=figure,
        path=path,
        dpi=dpi,
        show=show,
    )


def plot_binary_heatmap(
    edge_norms: np.ndarray,
    threshold: float,
    path: Path,
    title: str,
    dpi: int,
    show: bool,
) -> None:
    """
    0 = inactive（白）
    1 = active（黒）

    元コードの
        (abs(M) < threshold).astype(int)
    は「ゼロを1」にしていたため、ここではactiveを1とする。
    """
    active = (
        np.asarray(edge_norms)
        > threshold
    ).astype(np.int8)

    dim = active.shape[0]

    figure, axis = plt.subplots(
        figsize=(
            max(5.0, 0.46 * dim + 2.0),
            max(4.5, 0.46 * dim + 1.5),
        )
    )

    image = axis.imshow(
        active,
        cmap="binary",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )

    labels = make_axis_labels(dim)

    axis.set_xticks(
        np.arange(dim)
    )
    axis.set_xticklabels(
        labels,
        rotation=90,
    )

    axis.set_yticks(
        np.arange(dim)
    )
    axis.set_yticklabels(
        labels
    )

    axis.xaxis.tick_top()
    axis.xaxis.set_label_position(
        "top"
    )

    axis.set_xlabel(
        "Source at past lags"
    )
    axis.set_ylabel(
        "Target at current time"
    )
    axis.set_title(
        title,
        fontsize=9,
        pad=18,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        ticks=[0, 1],
        fraction=0.046,
        pad=0.04,
    )

    colorbar.ax.set_yticklabels(
        [
            "Inactive",
            "Active",
        ]
    )

    figure.tight_layout()

    save_or_show(
        figure=figure,
        path=path,
        dpi=dpi,
        show=show,
    )


def plot_lag_heatmaps(
    lag_norms: np.ndarray,
    output_prefix: Path,
    title_prefix: str,
    dpi: int,
    show: bool,
) -> None:
    for lag_index, matrix in enumerate(
        lag_norms,
        start=1,
    ):
        path = output_prefix.with_name(
            output_prefix.name
            + f"_lag_{lag_index:02d}.png"
        )

        plot_edge_heatmap(
            matrix=matrix,
            path=path,
            title=(
                f"{title_prefix}\n"
                f"Lag {lag_index}"
            ),
            dpi=dpi,
            show=show,
        )


def make_theta_summary(
    df: pd.DataFrame,
    output_dir: Path,
    dim: int,
    order: int,
    threshold: float,
) -> pd.DataFrame:
    summary_rows: list[
        dict[str, Any]
    ] = []

    for _, row in df.iterrows():
        if not bool(row["usable_theta"]):
            continue

        lambda_index = int(
            row["lambda_index"]
        )

        try:
            theta_path = resolve_theta_path(
                theta_file=row["theta_file"],
                output_dir=output_dir,
            )

            theta = np.load(
                theta_path,
                allow_pickle=False,
            )

            edge_norms = theta_to_edge_norms(
                theta=theta,
                dim=dim,
                order=order,
            )

            active_flat = np.flatnonzero(
                edge_norms.reshape(-1)
                > threshold
            )

            support_string_recomputed = ",".join(
                str(int(index))
                for index in active_flat
            )

            summary_rows.append(
                {
                    "lambda_index": lambda_index,
                    "lambda_value": float(
                        row["lambda_value"]
                    ),
                    "plic": (
                        float(row["plic"])
                        if "plic" in row
                        and pd.notna(row["plic"])
                        else math.nan
                    ),
                    "theta_file": str(
                        theta_path
                    ),
                    "theta_size": int(
                        np.asarray(theta).size
                    ),
                    "theta_l2_norm": float(
                        np.linalg.norm(theta)
                    ),
                    "theta_max_abs": float(
                        np.max(
                            np.abs(theta),
                            initial=0.0,
                        )
                    ),
                    "max_edge_group_norm": float(
                        np.max(
                            edge_norms,
                            initial=0.0,
                        )
                    ),
                    "num_active_edges_recomputed": int(
                        active_flat.size
                    ),
                    "support_string_recomputed": (
                        support_string_recomputed
                    ),
                    "num_active_edges_csv": (
                        int(row["active_groups"])
                        if "active_groups" in row
                        and pd.notna(
                            row["active_groups"]
                        )
                        else math.nan
                    ),
                    "support_string_csv": (
                        str(row["support_string"])
                        if "support_string" in row
                        and pd.notna(
                            row["support_string"]
                        )
                        else ""
                    ),
                    "load_error": "",
                }
            )

        except Exception as exc:
            summary_rows.append(
                {
                    "lambda_index": lambda_index,
                    "lambda_value": float(
                        row["lambda_value"]
                    ),
                    "plic": (
                        float(row["plic"])
                        if "plic" in row
                        and pd.notna(row["plic"])
                        else math.nan
                    ),
                    "theta_file": str(
                        row["theta_file"]
                    ),
                    "theta_size": math.nan,
                    "theta_l2_norm": math.nan,
                    "theta_max_abs": math.nan,
                    "max_edge_group_norm": math.nan,
                    "num_active_edges_recomputed": math.nan,
                    "support_string_recomputed": "",
                    "num_active_edges_csv": (
                        row.get(
                            "active_groups",
                            math.nan,
                        )
                    ),
                    "support_string_csv": (
                        row.get(
                            "support_string",
                            "",
                        )
                    ),
                    "load_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

    return pd.DataFrame(
        summary_rows
    )


def sort_heatmap_rows(
    df: pd.DataFrame,
    sort_mode: str,
) -> pd.DataFrame:
    if sort_mode == "lambda_index":
        return df.sort_values(
            "lambda_index"
        )

    if sort_mode == "lambda_asc":
        return df.sort_values(
            "lambda_value",
            ascending=True,
        )

    if sort_mode == "lambda_desc":
        return df.sort_values(
            "lambda_value",
            ascending=False,
        )

    if sort_mode == "plic":
        if "plic" not in df.columns:
            raise ValueError(
                "PLIC sorting requested, but CSV has no plic column."
            )

        return df.sort_values(
            [
                "plic",
                "lambda_index",
            ],
            na_position="last",
        )

    if sort_mode == "active_groups":
        if "active_groups" not in df.columns:
            raise ValueError(
                "active_groups sorting requested, "
                "but CSV has no active_groups column."
            )

        return df.sort_values(
            [
                "active_groups",
                "lambda_index",
            ],
            na_position="last",
        )

    raise ValueError(
        f"Unknown sort mode: {sort_mode}"
    )


def make_all_heatmaps(
    df: pd.DataFrame,
    output_dir: Path,
    plot_dir: Path,
    dim: int,
    order: int,
    threshold: float,
    max_heatmaps: int,
    sort_mode: str,
    dpi: int,
    show: bool,
) -> None:
    heatmap_dir = (
        plot_dir / "heatmaps"
    )

    rows = df[
        df["usable_theta"]
    ].copy()

    rows = sort_heatmap_rows(
        df=rows,
        sort_mode=sort_mode,
    )

    if max_heatmaps > 0:
        rows = rows.head(
            max_heatmaps
        )

    for count, (_, row) in enumerate(
        rows.iterrows(),
        start=1,
    ):
        lambda_index = int(
            row["lambda_index"]
        )

        lambda_value = float(
            row["lambda_value"]
        )

        plic = (
            float(row["plic"])
            if "plic" in row
            and pd.notna(row["plic"])
            else math.nan
        )

        theta_path = resolve_theta_path(
            theta_file=row["theta_file"],
            output_dir=output_dir,
        )

        theta = np.load(
            theta_path,
            allow_pickle=False,
        )

        edge_norms = theta_to_edge_norms(
            theta=theta,
            dim=dim,
            order=order,
        )

        lag_norms = theta_to_lag_edge_norms(
            theta=theta,
            dim=dim,
            order=order,
        )

        active_count = int(
            np.count_nonzero(
                edge_norms > threshold
            )
        )

        plic_text = (
            f"{plic:.6g}"
            if math.isfinite(plic)
            else "NA"
        )

        title_prefix = (
            f"lambda index={lambda_index}, "
            f"lambda={lambda_value:.6g}, "
            f"active={active_count}, "
            f"PLIC={plic_text}"
        )

        base_name = (
            f"lambda_{lambda_index:03d}"
        )

        plot_edge_heatmap(
            matrix=edge_norms,
            path=(
                heatmap_dir
                / f"{base_name}_raw.png"
            ),
            title=(
                title_prefix
                + "\nAll-lag group norm"
            ),
            dpi=dpi,
            show=show,
        )

        plot_binary_heatmap(
            edge_norms=edge_norms,
            threshold=threshold,
            path=(
                heatmap_dir
                / f"{base_name}_binary.png"
            ),
            title=(
                title_prefix
                + f"\nActive if norm > {threshold:.3g}"
            ),
            dpi=dpi,
            show=show,
        )

        plot_lag_heatmaps(
            lag_norms=lag_norms,
            output_prefix=(
                heatmap_dir
                / f"{base_name}_raw"
            ),
            title_prefix=title_prefix,
            dpi=dpi,
            show=show,
        )

        print(
            f"[heatmap {count}/{len(rows)}] "
            f"lambda_index={lambda_index}, "
            f"theta={theta_path.name}"
        )


def plot_plic_vs_active_groups(
    df: pd.DataFrame,
    path: Path,
    dpi: int,
    show: bool,
) -> None:
    if (
        "plic" not in df.columns
        or "active_groups" not in df.columns
    ):
        print(
            "[skip] PLIC vs active groups: "
            "required columns are unavailable."
        )
        return

    plot_df = df[
        np.isfinite(
            pd.to_numeric(
                df["plic"],
                errors="coerce",
            )
        )
        & np.isfinite(
            pd.to_numeric(
                df["active_groups"],
                errors="coerce",
            )
        )
    ].copy()

    if plot_df.empty:
        print(
            "[skip] PLIC vs active groups: no finite rows."
        )
        return

    plot_df = plot_df.sort_values(
        [
            "active_groups",
            "lambda_index",
        ]
    )

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.plot(
        plot_df["active_groups"],
        plot_df["plic"],
        "o-",
        linewidth=1.5,
        markersize=5,
    )

    best_index = plot_df["plic"].idxmin()
    best = plot_df.loc[
        best_index
    ]

    axis.scatter(
        [best["active_groups"]],
        [best["plic"]],
        marker="*",
        s=150,
        label=(
            "Minimum PLIC "
            f"(lambda index={int(best['lambda_index'])})"
        ),
        zorder=4,
    )

    axis.set_xlabel(
        "Number of active groups"
    )
    axis.set_ylabel(
        "PLIC"
    )
    axis.set_title(
        "PLIC vs Number of Active Groups"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    save_or_show(
        figure=figure,
        path=path,
        dpi=dpi,
        show=show,
    )


def plot_plic_vs_lambda(
    df: pd.DataFrame,
    path: Path,
    dpi: int,
    show: bool,
) -> None:
    if "plic" not in df.columns:
        print(
            "[skip] PLIC vs lambda: plic column unavailable."
        )
        return

    plot_df = df[
        np.isfinite(
            pd.to_numeric(
                df["plic"],
                errors="coerce",
            )
        )
        & (
            pd.to_numeric(
                df["lambda_value"],
                errors="coerce",
            )
            > 0
        )
    ].copy()

    if plot_df.empty:
        print(
            "[skip] PLIC vs lambda: no finite rows."
        )
        return

    plot_df = plot_df.sort_values(
        "lambda_value"
    )

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.plot(
        plot_df["lambda_value"],
        plot_df["plic"],
        "o-",
        linewidth=1.5,
        markersize=5,
    )

    best_index = plot_df["plic"].idxmin()
    best = plot_df.loc[
        best_index
    ]

    axis.scatter(
        [best["lambda_value"]],
        [best["plic"]],
        marker="*",
        s=150,
        label=(
            "Minimum PLIC "
            f"(index={int(best['lambda_index'])})"
        ),
        zorder=4,
    )

    axis.set_xscale(
        "log"
    )
    axis.set_xlabel(
        "Group-lasso lambda"
    )
    axis.set_ylabel(
        "PLIC"
    )
    axis.set_title(
        "PLIC vs Regularization Parameter"
    )
    axis.grid(
        True,
        which="both",
    )
    axis.legend()
    figure.tight_layout()

    save_or_show(
        figure=figure,
        path=path,
        dpi=dpi,
        show=show,
    )


def plot_objective_components(
    df: pd.DataFrame,
    path: Path,
    dpi: int,
    show: bool,
) -> None:
    required = {
        "lambda_value",
        "final_loss",
        "final_objective",
    }

    if not required.issubset(
        df.columns
    ):
        print(
            "[skip] Objective components: "
            "required columns unavailable."
        )
        return

    plot_df = df.copy()

    numeric = [
        "lambda_value",
        "final_loss",
        "final_objective",
        "final_penalty_unscaled",
        "final_ridge_penalty_unscaled",
    ]

    for column in numeric:
        if column in plot_df.columns:
            plot_df[column] = pd.to_numeric(
                plot_df[column],
                errors="coerce",
            )

    plot_df = plot_df[
        np.isfinite(
            plot_df["lambda_value"]
        )
        & (
            plot_df["lambda_value"] > 0
        )
        & np.isfinite(
            plot_df["final_loss"]
        )
        & np.isfinite(
            plot_df["final_objective"]
        )
    ].sort_values(
        "lambda_value"
    )

    if plot_df.empty:
        return

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.plot(
        plot_df["lambda_value"],
        plot_df["final_loss"],
        "o-",
        label="Logistic loss",
    )

    axis.plot(
        plot_df["lambda_value"],
        plot_df["final_objective"],
        "o-",
        label="Full objective",
    )

    axis.set_xscale(
        "log"
    )
    axis.set_xlabel(
        "Group-lasso lambda"
    )
    axis.set_ylabel(
        "Objective value"
    )
    axis.set_title(
        "Objective Components Along Regularization Path"
    )
    axis.grid(
        True,
        which="both",
    )
    axis.legend()
    figure.tight_layout()

    save_or_show(
        figure=figure,
        path=path,
        dpi=dpi,
        show=show,
    )


def load_ground_truth(
    path: Path,
) -> set[tuple[int, int]]:
    """
    例:
        1: 2 3
        2: 4

    は
        (1,2), (1,3), (2,4)
    を表す。
    """
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth file does not exist: {path}"
        )

    edges: set[
        tuple[int, int]
    ] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file_obj:
        for line_number, raw_line in enumerate(
            file_obj,
            start=1,
        ):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                raise ValueError(
                    f"Invalid ground-truth line {line_number}: "
                    f"{raw_line.rstrip()}"
                )

            source_text, destinations_text = (
                line.split(
                    ":",
                    maxsplit=1,
                )
            )

            source = int(
                source_text.strip()
            )

            destinations = (
                destinations_text
                .strip()
                .split()
            )

            for destination_text in destinations:
                destination = int(
                    destination_text
                )

                edges.add(
                    (
                        source,
                        destination,
                    )
                )

    return edges


def flat_index_to_nodes(
    flat_index: int,
    dim: int,
) -> tuple[int, int]:
    source_zero, target_zero = np.unravel_index(
        int(flat_index),
        (
            dim,
            dim,
        ),
    )

    return (
        int(source_zero) + 1,
        int(target_zero) + 1,
    )


def support_string_to_edges(
    support_string: Any,
    dim: int,
) -> set[tuple[int, int]]:
    if pd.isna(support_string):
        return set()

    text = str(
        support_string
    ).strip()

    if not text:
        return set()

    edges: set[
        tuple[int, int]
    ] = set()

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        edges.add(
            flat_index_to_nodes(
                flat_index=int(item),
                dim=dim,
            )
        )

    return edges


def calculate_edge_metrics(
    predicted_edges: set[tuple[int, int]],
    ground_truth_edges: set[tuple[int, int]],
    dim: int,
    include_self_loops: bool,
) -> dict[str, Any]:
    if not include_self_loops:
        predicted_edges = {
            edge
            for edge in predicted_edges
            if edge[0] != edge[1]
        }

        ground_truth_edges = {
            edge
            for edge in ground_truth_edges
            if edge[0] != edge[1]
        }

    all_edges = {
        (
            source,
            target,
        )
        for source in range(
            1,
            dim + 1,
        )
        for target in range(
            1,
            dim + 1,
        )
        if (
            include_self_loops
            or source != target
        )
    }

    invalid_predictions = (
        predicted_edges
        - all_edges
    )

    invalid_ground_truth = (
        ground_truth_edges
        - all_edges
    )

    if invalid_predictions:
        raise ValueError(
            "Predicted edges outside valid range: "
            f"{sorted(invalid_predictions)}"
        )

    if invalid_ground_truth:
        raise ValueError(
            "Ground-truth edges outside valid range: "
            f"{sorted(invalid_ground_truth)}"
        )

    true_positive = (
        predicted_edges
        & ground_truth_edges
    )

    false_positive = (
        predicted_edges
        - ground_truth_edges
    )

    false_negative = (
        ground_truth_edges
        - predicted_edges
    )

    true_negative = (
        all_edges
        - (
            true_positive
            | false_positive
            | false_negative
        )
    )

    tp = len(
        true_positive
    )
    fp = len(
        false_positive
    )
    fn = len(
        false_negative
    )
    tn = len(
        true_negative
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if precision + recall > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / len(all_edges)
        if all_edges
        else 0.0
    )

    return {
        "num_pred_edges": len(
            predicted_edges
        ),
        "num_true_edges": len(
            ground_truth_edges
        ),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "true_positive_edges": (
            edge_set_to_string(
                true_positive
            )
        ),
        "false_positive_edges": (
            edge_set_to_string(
                false_positive
            )
        ),
        "false_negative_edges": (
            edge_set_to_string(
                false_negative
            )
        ),
    }


def edge_set_to_string(
    edges: set[tuple[int, int]],
) -> str:
    return ",".join(
        f"{source}->{target}"
        for source, target in sorted(
            edges
        )
    )


def calculate_metrics_table(
    df: pd.DataFrame,
    ground_truth_edges: set[tuple[int, int]],
    dim: int,
    include_self_loops: bool,
) -> pd.DataFrame:
    metric_rows: list[
        dict[str, Any]
    ] = []

    for dataframe_index, row in df.iterrows():
        support_string = row.get(
            "support_string",
            "",
        )

        predicted_edges = support_string_to_edges(
            support_string=support_string,
            dim=dim,
        )

        metrics = calculate_edge_metrics(
            predicted_edges=predicted_edges,
            ground_truth_edges=ground_truth_edges,
            dim=dim,
            include_self_loops=(
                include_self_loops
            ),
        )

        metric_rows.append(
            {
                "dataframe_index": dataframe_index,
                "lambda_index": (
                    int(row["lambda_index"])
                    if pd.notna(
                        row["lambda_index"]
                    )
                    else math.nan
                ),
                "lambda_value": row.get(
                    "lambda_value",
                    math.nan,
                ),
                "fista_ridge": row.get(
                    "fista_ridge",
                    math.nan,
                ),
                "plic": row.get(
                    "plic",
                    math.nan,
                ),
                "active_groups_csv": row.get(
                    "active_groups",
                    math.nan,
                ),
                "support_string": (
                    support_string
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        metric_rows
    )


def write_metric_block(
    file_obj: Any,
    title: str,
    row: pd.Series,
) -> None:
    lines = [
        title,
        f"lambda_index: {row.get('lambda_index', math.nan)}",
        f"lambda_value: {row.get('lambda_value', math.nan)}",
        f"PLIC: {row.get('plic', math.nan)}",
        f"support_string: {row.get('support_string', '')}",
        f"Number of predicted edges: {row.get('num_pred_edges', 0)}",
        f"Number of true edges: {row.get('num_true_edges', 0)}",
        (
            f"TP: {row.get('TP', 0)} "
            f"FP: {row.get('FP', 0)} "
            f"FN: {row.get('FN', 0)} "
            f"TN: {row.get('TN', 0)}"
        ),
        f"Precision: {row.get('precision', math.nan)}",
        f"Recall: {row.get('recall', math.nan)}",
        f"F1: {row.get('f1', math.nan)}",
        f"Specificity: {row.get('specificity', math.nan)}",
        f"Accuracy: {row.get('accuracy', math.nan)}",
        (
            "True-positive edges: "
            f"{row.get('true_positive_edges', '')}"
        ),
        (
            "False-positive edges: "
            f"{row.get('false_positive_edges', '')}"
        ),
        (
            "False-negative edges: "
            f"{row.get('false_negative_edges', '')}"
        ),
        "-" * 70,
    ]

    for line in lines:
        print(line)
        file_obj.write(
            line + "\n"
        )


def save_metric_reports(
    metric_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    metric_df.to_csv(
        output_dir
        / "metrics_all.csv",
        index=False,
    )

    with (
        output_dir
        / "metric_all.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file_obj:
        for _, row in metric_df.iterrows():
            write_metric_block(
                file_obj=file_obj,
                title=(
                    "Lambda index "
                    f"{row['lambda_index']}"
                ),
                row=row,
            )

    valid_plic = metric_df[
        np.isfinite(
            pd.to_numeric(
                metric_df["plic"],
                errors="coerce",
            )
        )
    ]

    best_rows: list[
        tuple[str, pd.Series]
    ] = [
        (
            "Best Precision",
            metric_df.loc[
                metric_df[
                    "precision"
                ].idxmax()
            ],
        ),
        (
            "Best Recall",
            metric_df.loc[
                metric_df[
                    "recall"
                ].idxmax()
            ],
        ),
        (
            "Best F1",
            metric_df.loc[
                metric_df[
                    "f1"
                ].idxmax()
            ],
        ),
        (
            "Best Accuracy",
            metric_df.loc[
                metric_df[
                    "accuracy"
                ].idxmax()
            ],
        ),
    ]

    if not valid_plic.empty:
        best_rows.append(
            (
                "Minimum PLIC",
                valid_plic.loc[
                    valid_plic[
                        "plic"
                    ].idxmin()
                ],
            )
        )

    with (
        output_dir
        / "metric_best.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file_obj:
        for title, row in best_rows:
            write_metric_block(
                file_obj=file_obj,
                title=title,
                row=row,
            )


def plot_metrics_vs_lambda(
    metric_df: pd.DataFrame,
    path: Path,
    dpi: int,
    show: bool,
) -> None:
    plot_df = metric_df[
        (
            pd.to_numeric(
                metric_df["lambda_value"],
                errors="coerce",
            )
            > 0
        )
    ].sort_values(
        "lambda_value"
    )

    if plot_df.empty:
        return

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    for column, label in [
        (
            "precision",
            "Precision",
        ),
        (
            "recall",
            "Recall",
        ),
        (
            "f1",
            "F1",
        ),
        (
            "accuracy",
            "Accuracy",
        ),
    ]:
        axis.plot(
            plot_df["lambda_value"],
            plot_df[column],
            "o-",
            label=label,
        )

    axis.set_xscale(
        "log"
    )
    axis.set_ylim(
        -0.02,
        1.02,
    )
    axis.set_xlabel(
        "Group-lasso lambda"
    )
    axis.set_ylabel(
        "Metric"
    )
    axis.set_title(
        "Edge Recovery Metrics vs Lambda"
    )
    axis.grid(
        True,
        which="both",
    )
    axis.legend()
    figure.tight_layout()

    save_or_show(
        figure=figure,
        path=path,
        dpi=dpi,
        show=show,
    )


def plot_metrics_vs_edge_count(
    metric_df: pd.DataFrame,
    path: Path,
    dpi: int,
    show: bool,
) -> None:
    plot_df = metric_df.sort_values(
        [
            "num_pred_edges",
            "lambda_index",
        ]
    )

    if plot_df.empty:
        return

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    for column, label in [
        (
            "precision",
            "Precision",
        ),
        (
            "recall",
            "Recall",
        ),
        (
            "f1",
            "F1",
        ),
    ]:
        axis.plot(
            plot_df["num_pred_edges"],
            plot_df[column],
            "o-",
            label=label,
        )

    axis.set_ylim(
        -0.02,
        1.02,
    )
    axis.set_xlabel(
        "Number of Predicted Edges"
    )
    axis.set_ylabel(
        "Metric"
    )
    axis.set_title(
        "Edge Recovery vs Number of Predicted Edges"
    )
    axis.grid(
        True
    )
    axis.legend()
    figure.tight_layout()

    save_or_show(
        figure=figure,
        path=path,
        dpi=dpi,
        show=show,
    )


def make_best_plic_figures(
    df: pd.DataFrame,
    output_dir: Path,
    plot_dir: Path,
    dim: int,
    order: int,
    threshold: float,
    dpi: int,
    show: bool,
) -> None:
    if "plic" not in df.columns:
        return

    valid = df[
        df["usable_theta"]
        & np.isfinite(
            pd.to_numeric(
                df["plic"],
                errors="coerce",
            )
        )
    ].copy()

    if valid.empty:
        return

    best = valid.loc[
        valid["plic"].idxmin()
    ]

    theta_path = resolve_theta_path(
        theta_file=best["theta_file"],
        output_dir=output_dir,
    )

    theta = np.load(
        theta_path,
        allow_pickle=False,
    )

    edge_norms = theta_to_edge_norms(
        theta=theta,
        dim=dim,
        order=order,
    )

    lambda_index = int(
        best["lambda_index"]
    )

    title = (
        "Minimum PLIC model\n"
        f"lambda index={lambda_index}, "
        f"lambda={float(best['lambda_value']):.6g}, "
        f"PLIC={float(best['plic']):.6g}"
    )

    plot_edge_heatmap(
        matrix=edge_norms,
        path=(
            plot_dir
            / "best_plic_raw.png"
        ),
        title=title,
        dpi=dpi,
        show=show,
    )

    plot_binary_heatmap(
        edge_norms=edge_norms,
        threshold=threshold,
        path=(
            plot_dir
            / "best_plic_binary.png"
        ),
        title=(
            title
            + f"\nActive if norm > {threshold:.3g}"
        ),
        dpi=dpi,
        show=show,
    )


def save_best_model_summary(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if "plic" not in df.columns:
        return

    valid = df[
        np.isfinite(
            pd.to_numeric(
                df["plic"],
                errors="coerce",
            )
        )
    ].copy()

    if valid.empty:
        return

    best = valid.loc[
        valid["plic"].idxmin()
    ]

    summary: dict[str, Any] = {}

    for key, value in best.to_dict().items():
        if isinstance(
            value,
            np.generic,
        ):
            value = value.item()

        if pd.isna(value):
            value = None

        summary[key] = value

    with (
        output_dir
        / "visualization_best_plic.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(
            summary,
            file_obj,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def main() -> None:
    args = parse_args()

    validate_args(
        args
    )

    (
        output_dir,
        results_csv,
        plot_dir,
    ) = resolve_paths(
        args
    )

    print(
        f"Reading results: {results_csv}"
    )

    results_df = pd.read_csv(
        results_csv
    )

    results_df = normalize_result_columns(
        results_df
    )

    print(
        f"Rows in results CSV: {len(results_df)}"
    )

    print(
        "Usable theta rows: "
        f"{int(results_df['usable_theta'].sum())}"
    )

    theta_summary = make_theta_summary(
        df=results_df,
        output_dir=output_dir,
        dim=args.dim,
        order=args.order,
        threshold=args.threshold,
    )

    theta_summary_path = (
        output_dir
        / "theta_summary.csv"
    )

    theta_summary.to_csv(
        theta_summary_path,
        index=False,
    )

    print(
        f"Saved theta summary: {theta_summary_path}"
    )

    make_all_heatmaps(
        df=results_df,
        output_dir=output_dir,
        plot_dir=plot_dir,
        dim=args.dim,
        order=args.order,
        threshold=args.threshold,
        max_heatmaps=args.max_heatmaps,
        sort_mode=args.heatmap_sort,
        dpi=args.dpi,
        show=args.show,
    )

    plot_plic_vs_active_groups(
        df=results_df,
        path=(
            plot_dir
            / "plic_vs_active_groups.png"
        ),
        dpi=args.dpi,
        show=args.show,
    )

    plot_plic_vs_lambda(
        df=results_df,
        path=(
            plot_dir
            / "plic_vs_lambda.png"
        ),
        dpi=args.dpi,
        show=args.show,
    )

    plot_objective_components(
        df=results_df,
        path=(
            plot_dir
            / "objective_vs_lambda.png"
        ),
        dpi=args.dpi,
        show=args.show,
    )

    make_best_plic_figures(
        df=results_df,
        output_dir=output_dir,
        plot_dir=plot_dir,
        dim=args.dim,
        order=args.order,
        threshold=args.threshold,
        dpi=args.dpi,
        show=args.show,
    )

    save_best_model_summary(
        df=results_df,
        output_dir=output_dir,
    )

    if args.ground_truth is not None:
        ground_truth_edges = load_ground_truth(
            args.ground_truth
        )

        print(
            "Ground-truth edges: "
            f"{len(ground_truth_edges)}"
        )

        metric_df = calculate_metrics_table(
            df=results_df,
            ground_truth_edges=(
                ground_truth_edges
            ),
            dim=args.dim,
            include_self_loops=(
                args.include_self_loops
            ),
        )

        save_metric_reports(
            metric_df=metric_df,
            output_dir=output_dir,
        )

        plot_metrics_vs_lambda(
            metric_df=metric_df,
            path=(
                plot_dir
                / "metrics_vs_lambda.png"
            ),
            dpi=args.dpi,
            show=args.show,
        )

        plot_metrics_vs_edge_count(
            metric_df=metric_df,
            path=(
                plot_dir
                / "precision_recall_f1_vs_edges.png"
            ),
            dpi=args.dpi,
            show=args.show,
        )

    print("\nFinished.")
    print(
        f"Figures: {plot_dir}"
    )


if __name__ == "__main__":
    main()