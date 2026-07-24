from __future__ import annotations

"""
Multi-GPU cold-start group elastic-net FISTA experiment.

実行の流れ:
1. Kuramoto_Model() でrawデータを生成する。
2. build_X_torus() でXを生成し、X.npyへ保存する。
3. X.npyを各GPUへ1回ずつ転送する。
4. 各lambdaをk-foldまたはLOOCVで評価する。
5. 各lambdaを全データで再推定し、推定結果と係数を保存する。
6. 平均CV損失が最小のlambdaを選択する。

Xのfeature layout:
    [lag][directed edge][cc, cs, sc, ss]

Xのshape:
    (n_pairs, order * 4 * dim * dim)

注意:
- rawは.npyへ保存しない。
- build_X_torus() はX全体をCPU RAM上に生成する。
- raw生成とX生成はmain()内でのみ実行する。
- multiprocessingのspawnによる子プロセスでは再生成しない。
- 各GPUにはX全体が1コピーずつ配置される。
"""

import argparse
import gc
import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from feature import build_X_torus
from data_sim import Kuramoto_Model, generate_5d_phase_timeseries_data
from data_real import (
    load_marmoset_ecog,
    load_marmoset_ecog_epoched,
    split_marmoset_pre_post_1500ms,
    extract_feature_matrix,
    FeatureSpec,
    NeuralDataset,
)

@dataclass
class FistaConfig:
    max_iter: int
    tol: float
    pg_tol: float
    initial_L: float
    line_search_factor: float
    max_line_search_steps: int
    support_abs_tol: float
    support_rel_tol: float
    objective_check_every: int
    dtype: str
    compute_ic: bool
    fista_ridge: float
    refit_ridge: float
    refit_max_iter: int
    refit_tolerance_grad: float
    refit_tolerance_change: float
    refit_history_size: int
    ic_ridge: float
    nw_bandwidth: int
    nw_center: bool
    ic_chunk_rows: int


@dataclass
class LambdaResult:
    lambda_index: int
    lambda_value: float
    gpu_id: int
    converged: bool
    iterations: int
    elapsed_sec: float
    final_objective: float
    final_loss: float
    final_penalty_unscaled: float
    final_ridge_penalty_unscaled: float
    final_L: float
    relative_step: float
    gradient_mapping_norm: float
    theta_norm: float
    theta_maxabs: float
    active_groups: int
    support_threshold: float
    support_string: str
    theta_file: str
    cv_method: str = ""
    cv_n_splits: int = 0
    cv_loss_mean: float = math.nan
    cv_loss_std: float = math.nan
    cv_loss_se: float = math.nan
    cv_total_validation_rows: int = 0
    cv_converged_folds: int = 0
    cv_fold_losses: str = ""
    refit_file: str = ""
    refit_converged: bool = False
    refit_iterations: int = 0
    refit_elapsed_sec: float = math.nan
    refit_loss_mean: float = math.nan
    refit_grad_norm: float = math.nan
    log_likelihood: float = math.nan
    minus2_log_likelihood: float = math.nan
    trace_jinv_i_iid: float = math.nan
    trace_jinv_i_nw: float = math.nan
    bias_iid: float = math.nan
    bias_nw: float = math.nan
    ic_iid: float = math.nan
    plic: float = math.nan
    nw_bandwidth_used: int = 0
    j_eig_min: float = math.nan
    j_eig_max: float = math.nan
    j_condition: float = math.nan
    ic_error: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Kuramoto raw data and X if requested, then run "
            "independent cold-start group elastic-net FISTA fits on multiple GPUs."
        )
    )

    parser.add_argument(
        "--x-npy",
        type=Path,
        required=True,
        help="生成または読み込むX.npyのパス。",
    )
    parser.add_argument(
        "--build-x",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Kuramoto_Modelからrawを生成し、"
            "build_X_torusでX.npyを作成してから実験を開始する。"
        ),
    )
    parser.add_argument(
        "--overwrite-x",
        action="store_true",
        help="既存のX.npyを上書きして再生成する。",
    )
    parser.add_argument(
        "--skip-x-finite-check",
        action="store_true",
        help=(
            "生成後のX全体に対するNaN/Inf検査を省略する。"
            "巨大Xで検査時間を短縮したい場合に使用する。"
        ),
    )

    parser.add_argument(
        "--dim",
        type=int,
        required=True,
        help="Kuramoto_ModelのNおよびXの次元。",
    )
    parser.add_argument(
        "--order",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated CUDA device IDs.",
    )

    parser.add_argument(
        "--lambda-log10-max",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--lambda-log10-min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--num-lambdas",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--lambda-scale-by-nrows",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lambda-npy",
        type=Path,
        default=None,
        help=(
            "Optional explicit 1-D lambda array. "
            "Overrides logspace settings."
        ),
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help=(
            "Relative iterate-change tolerance used to trigger "
            "an optimality check."
        ),
    )
    parser.add_argument(
        "--pg-tol",
        type=float,
        default=1e-6,
        help=(
            "Relative proximal-gradient mapping tolerance used "
            "for the final convergence decision."
        ),
    )
    parser.add_argument(
        "--initial-L",
        type=float,
        default=1.0,
        help="Initial Lipschitz estimate for backtracking.",
    )
    parser.add_argument(
        "--line-search-factor",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--max-line-search-steps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--objective-check-every",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--support-abs-tol",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--support-rel-tol",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
    )

    parser.add_argument(
        "--fista-ridge",
        type=float,
        default=1e-5,
        help=(
            "FISTA本体のmean-lossスケールridge係数。"
            "目的関数に0.5 * fista_ridge * ||theta||^2を加える。"
        ),
    )

    parser.add_argument(
        "--cv-method",
        choices=["kfold", "loocv"],
        default="kfold",
        help=(
            "正則化パラメタ選択法。kfoldは--cv-folds分割、"
            "loocvは1行ずつ検証に回す。"
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="--cv-method=kfoldで使う分割数。",
    )
    parser.add_argument(
        "--cv-shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="k-fold分割前に行インデックスをシャッフルする。",
    )
    parser.add_argument(
        "--cv-seed",
        type=int,
        default=12345,
        help="CV分割の乱数seed。",
    )

    parser.add_argument(
        "--compute-ic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "互換性のため残している旧オプション。CV選択では通常無効にする。"
        ),
    )
    parser.add_argument(
        "--refit-ridge",
        type=float,
        default=1e-5,
        help="選択後ロジスティック再推定のmean-lossスケールridge係数。",
    )
    parser.add_argument(
        "--refit-max-iter",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--refit-tolerance-grad",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--refit-tolerance-change",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--refit-history-size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--ic-ridge",
        type=float,
        default=1e-8,
        help="J_hatの線形方程式を安定化するridge。",
    )
    parser.add_argument(
        "--nw-bandwidth",
        type=int,
        default=-1,
        help="Newey-West帯域。-1ならデータ長から自動決定。",
    )
    parser.add_argument(
        "--nw-center",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ic-chunk-rows",
        type=int,
        default=50000,
        help="TIC/HAC集計時のGPUチャンク行数。",
    )

    parser.add_argument(
        "--x-transfer-chunk-rows",
        type=int,
        default=0,
        help=(
            "Rows per CPU-to-GPU transfer chunk. "
            "0 copies X in one operation. "
            "A positive value divides the transfer, but X is still fully "
            "resident on each GPU after transfer."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save aggregate CSV after every N completed lambda fits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./gpu_group_lasso_results"),
    )

    return parser.parse_args()


def numpy_dtype_from_name(dtype_name: str) -> np.dtype:
    if dtype_name == "float32":
        return np.dtype(np.float32)

    if dtype_name == "float64":
        return np.dtype(np.float64)

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def validate_basic_arguments(args: argparse.Namespace) -> None:
    if args.dim <= 0:
        raise ValueError(
            f"--dim must be positive, got {args.dim}"
        )

    if args.order <= 0:
        raise ValueError(
            f"--order must be positive, got {args.order}"
        )

    if args.num_lambdas <= 0 and args.lambda_npy is None:
        raise ValueError(
            "--num-lambdas must be positive when "
            "--lambda-npy is not used."
        )

    if args.cv_method == "kfold" and args.cv_folds < 2:
        raise ValueError(
            "--cv-folds must be at least 2 for kfold."
        )

    if args.max_iter <= 0:
        raise ValueError(
            f"--max-iter must be positive, got {args.max_iter}"
        )

    if args.tol <= 0:
        raise ValueError(
            f"--tol must be positive, got {args.tol}"
        )

    if args.pg_tol <= 0:
        raise ValueError(
            f"--pg-tol must be positive, got {args.pg_tol}"
        )

    if args.initial_L <= 0:
        raise ValueError(
            f"--initial-L must be positive, got {args.initial_L}"
        )

    if args.line_search_factor <= 1.0:
        raise ValueError(
            "--line-search-factor must be greater than 1."
        )

    if args.max_line_search_steps <= 0:
        raise ValueError(
            "--max-line-search-steps must be positive."
        )

    if args.objective_check_every <= 0:
        raise ValueError(
            "--objective-check-every must be positive."
        )

    if args.x_transfer_chunk_rows < 0:
        raise ValueError(
            "--x-transfer-chunk-rows must be zero or positive."
        )

    if args.save_every <= 0:
        raise ValueError(
            "--save-every must be positive."
        )

    if (
        args.fista_ridge < 0
        or args.refit_ridge < 0
        or args.ic_ridge < 0
    ):
        raise ValueError(
            "--fista-ridge, --refit-ridge, and --ic-ridge "
            "must be nonnegative."
        )

    if args.refit_max_iter <= 0 or args.refit_history_size <= 0:
        raise ValueError(
            "Refit iteration/history settings must be positive."
        )

    if args.refit_tolerance_grad <= 0 or args.refit_tolerance_change <= 0:
        raise ValueError(
            "Refit tolerances must be positive."
        )

    if args.nw_bandwidth < -1:
        raise ValueError(
            "--nw-bandwidth must be -1 or nonnegative."
        )

    if args.ic_chunk_rows <= 0:
        raise ValueError(
            "--ic-chunk-rows must be positive."
        )


def validate_raw_array(
    raw: np.ndarray,
    expected_dim: int,
) -> None:
    if raw.ndim != 2:
        raise ValueError(
            f"raw must be 2-D, got shape={raw.shape}"
        )

    if raw.shape[0] == 0:
        raise ValueError(
            "raw has zero rows."
        )

    if raw.shape[1] != expected_dim:
        raise ValueError(
            f"raw has {raw.shape[1]} columns, "
            f"but --dim={expected_dim}."
        )

    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(
            f"raw must have a numeric dtype, got dtype={raw.dtype}"
        )

    if not np.isfinite(raw).all():
        raise ValueError(
            "Generated raw data contains NaN or Inf."
        )


def validate_x_array(
    X: np.ndarray,
    dim: int,
    order: int,
    x_path: Path,
) -> None:
    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-D, got shape={X.shape}, path={x_path}"
        )

    expected_features = order * 4 * dim * dim

    if X.shape[1] != expected_features:
        raise ValueError(
            f"X has {X.shape[1]} columns, "
            f"expected {expected_features} "
            f"for dim={dim}, order={order}."
        )

    if X.shape[0] == 0:
        raise ValueError(
            f"X has zero rows: {x_path}"
        )

    if not np.issubdtype(X.dtype, np.floating):
        raise TypeError(
            f"X must have a floating-point dtype, "
            f"got dtype={X.dtype}"
        )


def atomic_save_npy(
    path: Path,
    array: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".partial"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with temporary_path.open("wb") as file_obj:
            np.save(
                file_obj,
                array,
                allow_pickle=False,
            )
            file_obj.flush()
            os.fsync(file_obj.fileno())

        os.replace(
            temporary_path,
            path,
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise



def prepare_x(
    args: argparse.Namespace,
) -> dict[str, Any]:
    x_path = args.x_npy.expanduser().resolve()

    metadata: dict[str, Any] = {
        "requested": bool(args.build_x),
        "created": False,
        "reused_existing": False,
        "x_path": str(x_path),
        "kuramoto_parameters": {
            "N": int(args.dim),
        },
    }

    if not args.build_x:
        if not x_path.exists():
            raise FileNotFoundError(
                f"X file does not exist: {x_path}\n"
                "Specify --build-x to generate raw and X."
            )

        metadata["reused_existing"] = True
        return metadata

    if x_path.exists() and not args.overwrite_x:
        print(
            f"[X build] Existing X will be reused: {x_path}"
        )

        metadata["reused_existing"] = True
        return metadata

    raw_start = perf_counter()

    ## When using Kuramoto Model simulation data, uncomment below.
    print(
        "[raw build] Starting Kuramoto_Model\n"
        f"  N          : {args.dim}"
    )
    raw, _ = Kuramoto_Model(
        N=args.dim,
        directed_K=False,
        base_k=0.4,
        T=5,
    )

    # from run_PPC2 import sample_plot; sample_plot(raw, str(args.output_dir))
    raw_elapsed = perf_counter() - raw_start
    raw = np.asarray(raw)

    validate_raw_array(
        raw=raw,
        expected_dim=args.dim,
    )

    print(
        "[raw build] Kuramoto_Model completed\n"
        f"  raw shape : {raw.shape}\n"
        f"  raw dtype : {raw.dtype}\n"
        f"  raw size  : {raw.nbytes / 1024**3:.4f} GiB\n"
        f"  elapsed   : {raw_elapsed:.2f} sec"
    )

    output_dtype = numpy_dtype_from_name(
        args.dtype
    )

    print(
        "[X build] Starting build_X_torus\n"
        f"  raw shape : {raw.shape}\n"
        f"  order     : {args.order}\n"
        f"  X dtype   : {output_dtype}"
    )

    build_start = perf_counter()

    X = build_X_torus(
        raw=raw,
        order=args.order,
        dtype=output_dtype.type,
        show_progress=True,
    )

    build_elapsed = perf_counter() - build_start

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    validate_x_array(
        X=X,
        dim=args.dim,
        order=args.order,
        x_path=x_path,
    )

    if X.dtype != output_dtype:
        print(
            f"[X build] Converting X dtype from "
            f"{X.dtype} to {output_dtype}."
        )

        X = X.astype(
            output_dtype,
            copy=False,
        )

    x_size_gib = X.nbytes / 1024**3

    print(
        "[X build] X construction completed\n"
        f"  X shape    : {X.shape}\n"
        f"  X dtype    : {X.dtype}\n"
        f"  X size     : {x_size_gib:.2f} GiB\n"
        f"  build time : {build_elapsed:.2f} sec"
    )

    finite_check_elapsed = 0.0

    if args.skip_x_finite_check:
        print(
            "[X build] Skipping the full X NaN/Inf check."
        )
    else:
        print(
            "[X build] Checking X for NaN and Inf..."
        )

        finite_check_start = perf_counter()

        if not np.isfinite(X).all():
            raise ValueError(
                "Generated X contains NaN or Inf."
            )

        finite_check_elapsed = (
            perf_counter() - finite_check_start
        )

        print(
            "[X build] Finite-value check completed in "
            f"{finite_check_elapsed:.2f} sec."
        )

    print(
        f"[X build] Saving X atomically: {x_path}"
    )

    save_start = perf_counter()

    atomic_save_npy(
        path=x_path,
        array=X,
    )

    save_elapsed = perf_counter() - save_start

    print(
        "[X build] X saved\n"
        f"  path      : {x_path}\n"
        f"  save time : {save_elapsed:.2f} sec"
    )

    metadata.update(
        {
            "created": True,
            "raw_shape": list(raw.shape),
            "raw_dtype": str(raw.dtype),
            "raw_size_gib": float(
                raw.nbytes / 1024**3
            ),
            "raw_elapsed_sec": float(
                raw_elapsed
            ),
            "X_shape": list(X.shape),
            "X_dtype": str(X.dtype),
            "X_size_gib": float(
                x_size_gib
            ),
            "X_build_elapsed_sec": float(
                build_elapsed
            ),
            "finite_check_elapsed_sec": float(
                finite_check_elapsed
            ),
            "save_elapsed_sec": float(
                save_elapsed
            ),
        }
    )

    del X
    del raw
    gc.collect()

    print(
        "[X build] Released parent-process X and raw references."
    )

    return metadata


def make_edge_lag_groups(
    dim: int,
    order: int,
) -> np.ndarray:
    groups: list[list[int]] = []
    per_lag = 4 * dim * dim

    for i in range(dim):
        for j in range(dim):
            edge_id = i * dim + j
            idx: list[int] = []

            for lag in range(order):
                start = (
                    lag * per_lag
                    + edge_id * 4
                )

                idx.extend(
                    range(start, start + 4)
                )

            groups.append(idx)

    return np.asarray(
        groups,
        dtype=np.int64,
    )


def split_indices_round_robin(
    n_items: int,
    gpu_ids: list[int],
) -> dict[int, list[int]]:
    assignments = {
        gpu_id: []
        for gpu_id in gpu_ids
    }

    for i in range(n_items):
        assignments[
            gpu_ids[i % len(gpu_ids)]
        ].append(i)

    return assignments


def torch_dtype_from_name(
    torch: Any,
    dtype_name: str,
):
    if dtype_name == "float32":
        return torch.float32

    if dtype_name == "float64":
        return torch.float64

    raise ValueError(
        f"Unsupported dtype: {dtype_name}"
    )


def copy_numpy_to_gpu(
    torch: Any,
    X_cpu: np.ndarray,
    device: Any,
    dtype: Any,
    chunk_rows: int,
):
    n_rows, n_cols = X_cpu.shape

    if chunk_rows <= 0:
        return torch.as_tensor(
            X_cpu,
            dtype=dtype,
            device=device,
        )

    X_gpu = torch.empty(
        (n_rows, n_cols),
        dtype=dtype,
        device=device,
    )

    for start in range(
        0,
        n_rows,
        chunk_rows,
    ):
        end = min(
            start + chunk_rows,
            n_rows,
        )

        chunk = torch.as_tensor(
            X_cpu[start:end],
            dtype=dtype,
            device=device,
        )

        X_gpu[start:end].copy_(chunk)
        del chunk

    return X_gpu


def smooth_loss(
    torch: Any,
    X: Any,
    theta: Any,
    ridge: float = 0.0,
) -> Any:
    eta = X @ theta

    loss = torch.nn.functional.softplus(
        -eta
    ).mean()

    if ridge:
        loss = (
            loss
            + 0.5
            * float(ridge)
            * torch.dot(theta, theta)
        )

    return loss


def smooth_loss_and_gradient(
    torch: Any,
    X: Any,
    theta: Any,
    ridge: float = 0.0,
) -> tuple[Any, Any]:
    eta = X @ theta

    loss = torch.nn.functional.softplus(
        -eta
    ).mean()

    residual = torch.sigmoid(
        -eta
    )

    grad = -(
        X.transpose(0, 1) @ residual
    ) / X.shape[0]

    if ridge:
        loss = (
            loss
            + 0.5
            * float(ridge)
            * torch.dot(theta, theta)
        )
        grad = grad + float(ridge) * theta

    return loss, grad


def group_penalty(
    torch: Any,
    theta: Any,
    groups: Any,
) -> Any:
    blocks = theta[groups]

    return torch.linalg.vector_norm(
        blocks,
        dim=1,
    ).sum()


def group_prox(
    torch: Any,
    z: Any,
    threshold: float,
    groups: Any,
) -> Any:
    blocks = z[groups]

    norms = torch.linalg.vector_norm(
        blocks,
        dim=1,
        keepdim=True,
    )

    eps = torch.finfo(
        z.dtype
    ).tiny

    scales = torch.clamp(
        1.0
        - threshold
        / torch.clamp(
            norms,
            min=eps,
        ),
        min=0.0,
    )

    prox_blocks = blocks * scales

    out = torch.empty_like(z)

    out[
        groups.reshape(-1)
    ] = prox_blocks.reshape(-1)

    return out


def objective(
    torch: Any,
    X: Any,
    theta: Any,
    lam: float,
    ridge: float,
    groups: Any,
) -> tuple[Any, Any, Any, Any]:
    logistic_loss = smooth_loss(
        torch=torch,
        X=X,
        theta=theta,
        ridge=0.0,
    )

    ridge_penalty = 0.5 * torch.dot(theta, theta)

    penalty = group_penalty(
        torch=torch,
        theta=theta,
        groups=groups,
    )

    return (
        logistic_loss
        + float(ridge) * ridge_penalty
        + lam * penalty,
        logistic_loss,
        penalty,
        ridge_penalty,
    )


def proximal_gradient_mapping_norm(
    torch: Any,
    X: Any,
    theta: Any,
    lam: float,
    ridge: float,
    groups: Any,
    L: float,
) -> float:
    _, grad = smooth_loss_and_gradient(
        torch=torch,
        X=X,
        theta=theta,
        ridge=ridge,
    )

    prox = group_prox(
        torch=torch,
        z=theta - grad / L,
        threshold=lam / L,
        groups=groups,
    )

    mapping = L * (
        theta - prox
    )

    return float(
        torch.linalg.vector_norm(
            mapping
        ).item()
    )


def backtracking_step(
    torch: Any,
    X: Any,
    y_point: Any,
    grad_y: Any,
    loss_y: Any,
    lam: float,
    ridge: float,
    groups: Any,
    L_start: float,
    factor: float,
    max_steps: int,
) -> tuple[Any, float]:
    L = float(L_start)

    for _ in range(max_steps):
        candidate = group_prox(
            torch=torch,
            z=y_point - grad_y / L,
            threshold=lam / L,
            groups=groups,
        )

        delta = candidate - y_point

        candidate_loss = smooth_loss(
            torch=torch,
            X=X,
            theta=candidate,
            ridge=ridge,
        )

        quadratic_bound = (
            loss_y
            + torch.dot(
                grad_y,
                delta,
            )
            + 0.5
            * L
            * torch.dot(
                delta,
                delta,
            )
        )

        if bool(
            (
                candidate_loss
                <= quadratic_bound + 1e-12
            ).item()
        ):
            return candidate, L

        L *= factor

    raise RuntimeError(
        f"Backtracking failed after {max_steps} steps; "
        f"final L={L:.6e}"
    )


def fit_group_lasso_fista(
    torch: Any,
    X: Any,
    groups: Any,
    lam: float,
    config: FistaConfig,
) -> tuple[Any, dict[str, Any]]:
    """
    Independent zero-initialized FISTA fit.

    Convergence is decided by the proximal-gradient mapping norm.
    Relative iterate change is used only to trigger an optimality check.
    """
    n_features = X.shape[1]

    theta = torch.zeros(
        n_features,
        dtype=X.dtype,
        device=X.device,
    )

    y_point = theta.clone()
    t_value = 1.0
    L = float(config.initial_L)
    accepted_L = float(config.initial_L)

    converged = False
    relative_step = math.inf
    pg_norm = math.inf
    last_objective = math.inf

    start = perf_counter()

    for iteration in range(
        1,
        config.max_iter + 1,
    ):
        loss_y, grad_y = smooth_loss_and_gradient(
            torch=torch,
            X=X,
            theta=y_point,
            ridge=config.fista_ridge,
        )

        theta_next, accepted_L = backtracking_step(
            torch=torch,
            X=X,
            y_point=y_point,
            grad_y=grad_y,
            loss_y=loss_y,
            lam=lam,
            ridge=config.fista_ridge,
            groups=groups,
            L_start=L,
            factor=config.line_search_factor,
            max_steps=config.max_line_search_steps,
        )

        L = max(
            accepted_L / config.line_search_factor,
            1e-12,
        )

        theta_diff = theta_next - theta

        diff_norm = torch.linalg.vector_norm(
            theta_diff
        )

        base_norm = torch.clamp(
            torch.linalg.vector_norm(theta),
            min=1.0,
        )

        relative_step = float(
            (diff_norm / base_norm).item()
        )

        t_next = 0.5 * (
            1.0
            + math.sqrt(
                1.0
                + 4.0
                * t_value
                * t_value
            )
        )

        momentum = (
            t_value - 1.0
        ) / t_next

        y_next = (
            theta_next
            + momentum * theta_diff
        )

        # 修正点3:
        # 旧判定は momentum * ||theta_next-theta||^2 となり、
        # ほぼ必ずrestartしていた。
        restart_inner = torch.dot(
            y_point - theta_next,
            theta_next - theta,
        )

        if bool(
            (restart_inner > 0).item()
        ):
            t_next = 1.0
            y_next = theta_next.clone()

        theta = theta_next
        y_point = y_next
        t_value = t_next

        check_optimality = (
            iteration == 1
            or iteration
            % config.objective_check_every
            == 0
            or relative_step <= config.tol
        )

        if check_optimality:
            obj, _, _, _ = objective(
                torch=torch,
                X=X,
                theta=theta,
                lam=lam,
                ridge=config.fista_ridge,
                groups=groups,
            )

            last_objective = float(
                obj.item()
            )

            pg_norm = proximal_gradient_mapping_norm(
                torch=torch,
                X=X,
                theta=theta,
                lam=lam,
                ridge=config.fista_ridge,
                groups=groups,
                L=max(accepted_L, 1e-12),
            )

            # 両方を満たした場合のみ収束
            if pg_norm <= config.pg_tol:
                converged = True
                break

    torch.cuda.synchronize(
        X.device
    )

    elapsed = perf_counter() - start

    (
        final_obj,
        final_loss,
        final_penalty,
        final_ridge_penalty,
    ) = objective(
        torch=torch,
        X=X,
        theta=theta,
        lam=lam,
        ridge=config.fista_ridge,
        groups=groups,
    )

    final_pg_norm = proximal_gradient_mapping_norm(
        torch=torch,
        X=X,
        theta=theta,
        lam=lam,
        ridge=config.fista_ridge,
        groups=groups,
        L=max(accepted_L, 1e-12),
    )

    final_pg_tolerance = config.pg_tol

    if (
        final_pg_norm <= final_pg_tolerance
        and relative_step <= config.tol
    ):
        converged = True

    return theta, {
        "converged": converged,
        "iterations": iteration,
        "elapsed_sec": elapsed,
        "final_objective": float(
            final_obj.item()
        ),
        "final_loss": float(
            final_loss.item()
        ),
        "final_penalty_unscaled": float(
            final_penalty.item()
        ),
        "final_ridge_penalty_unscaled": float(
            final_ridge_penalty.item()
        ),
        "final_L": float(accepted_L),
        "relative_step": float(
            relative_step
        ),
        "gradient_mapping_norm": float(
            final_pg_norm
        ),
        "gradient_mapping_tolerance": float(
            final_pg_tolerance
        ),
        "last_checked_objective": float(
            last_objective
        ),
    }


def active_support(
    torch: Any,
    theta: Any,
    groups: Any,
    abs_tol: float,
    rel_tol: float,
) -> tuple[np.ndarray, float]:
    norms = torch.linalg.vector_norm(
        theta[groups],
        dim=1,
    )

    threshold = float(abs_tol)

    support = torch.nonzero(
        norms > threshold,
        as_tuple=False,
    ).reshape(-1)

    return (
        support.detach()
        .cpu()
        .numpy()
        .astype(np.int64),
        threshold,
    )


def infer_endpoint_count_from_pair_rows(n_pairs: int) -> int:
    disc = 1 + 8 * int(n_pairs)
    root = math.isqrt(disc)

    if root * root != disc or (1 + root) % 2 != 0:
        raise ValueError(
            f"X row count {n_pairs} is not a triangular number; "
            "cannot reconstruct strict pair indexing."
        )

    n_endpoints = (1 + root) // 2

    if n_endpoints * (n_endpoints - 1) // 2 != n_pairs:
        raise ValueError(
            "Failed to infer endpoint count from X rows."
        )

    return int(n_endpoints)


def make_local_pair_indices(
    torch: Any,
    n_endpoints: int,
    device: Any,
) -> Any:
    return torch.triu_indices(
        n_endpoints,
        n_endpoints,
        offset=1,
        dtype=torch.long,
        device=device,
    ).transpose(0, 1).contiguous()


def refit_selected_support_lbfgs(
    torch: Any,
    X_sub: Any,
    ridge: float,
    max_iter: int,
    tolerance_grad: float,
    tolerance_change: float,
    history_size: int,
) -> tuple[Any, dict[str, Any]]:
    p_dim = int(X_sub.shape[1])

    if p_dim == 0:
        return torch.zeros(
            0,
            dtype=X_sub.dtype,
            device=X_sub.device,
        ), {
            "converged": True,
            "iterations": 0,
            "elapsed_sec": 0.0,
            "loss_mean": float(math.log(2.0)),
            "grad_norm": 0.0,
        }

    theta = torch.zeros(
        p_dim,
        dtype=X_sub.dtype,
        device=X_sub.device,
        requires_grad=True,
    )

    optimizer = torch.optim.LBFGS(
        [theta],
        lr=1.0,
        max_iter=int(max_iter),
        max_eval=int(max_iter * 5 // 4),
        tolerance_grad=float(tolerance_grad),
        tolerance_change=float(tolerance_change),
        history_size=int(history_size),
        line_search_fn="strong_wolfe",
    )

    state = {
        "calls": 0
    }

    def closure():
        optimizer.zero_grad(
            set_to_none=True
        )

        eta = X_sub @ theta

        loss = torch.nn.functional.softplus(
            -eta
        ).mean()

        if ridge:
            loss = (
                loss
                + 0.5
                * float(ridge)
                * torch.dot(theta, theta)
            )

        loss.backward()
        state["calls"] += 1

        return loss

    torch.cuda.synchronize(
        X_sub.device
    )

    start = perf_counter()
    optimizer.step(closure)

    torch.cuda.synchronize(
        X_sub.device
    )

    elapsed = perf_counter() - start

    with torch.no_grad():
        eta = X_sub @ theta

        loss_mean = torch.nn.functional.softplus(
            -eta
        ).mean()

        residual = torch.sigmoid(
            -eta
        )

        grad = -(
            X_sub.transpose(0, 1) @ residual
        ) / X_sub.shape[0]

        if ridge:
            grad = (
                grad
                + float(ridge) * theta
            )

        grad_norm = float(
            torch.linalg.vector_norm(
                grad
            ).item()
        )

    converged = bool(
        math.isfinite(grad_norm)
        and grad_norm
        <= max(
            10.0 * tolerance_grad,
            1e-7,
        )
    )

    return theta.detach(), {
        "converged": converged,
        "iterations": int(
            state["calls"]
        ),
        "elapsed_sec": float(
            elapsed
        ),
        "loss_mean": float(
            loss_mean.item()
        ),
        "grad_norm": grad_norm,
    }


def tic_hac_diagnostics_gpu(
    torch: Any,
    X_sub: Any,
    theta: Any,
    local_pairs: Any,
    ridge: float,
    nw_bandwidth: int,
    center_nw: bool,
    chunk_rows: int,
) -> dict[str, Any]:
    K_d = int(X_sub.shape[0])
    p_dim = int(X_sub.shape[1])

    N_d = (
        int(local_pairs.max().item()) + 1
        if K_d
        else 0
    )

    if p_dim == 0:
        log_likelihood = -K_d * math.log(2.0)
        minus2 = -2.0 * log_likelihood

        return {
            "log_likelihood": log_likelihood,
            "minus2_log_likelihood": minus2,
            "trace_jinv_i_iid": 0.0,
            "trace_jinv_i_nw": 0.0,
            "bias_iid": 0.0,
            "bias_nw": 0.0,
            "ic_iid": minus2,
            "plic": minus2,
            "nw_bandwidth_used": 0,
            "j_eig_min": math.nan,
            "j_eig_max": math.nan,
            "j_condition": math.nan,
        }

    acc_dtype = torch.float64
    theta_acc = theta.to(acc_dtype)

    J_sum = torch.zeros(
        (p_dim, p_dim),
        dtype=acc_dtype,
        device=X_sub.device,
    )

    endpoint_score_sum = torch.zeros(
        (N_d, p_dim),
        dtype=acc_dtype,
        device=X_sub.device,
    )

    log_likelihood = torch.zeros(
        (),
        dtype=acc_dtype,
        device=X_sub.device,
    )

    for start in range(
        0,
        K_d,
        int(chunk_rows),
    ):
        end = min(
            start + int(chunk_rows),
            K_d,
        )

        Xc = X_sub[start:end].to(
            acc_dtype
        )

        eta = Xc @ theta_acc
        pi = torch.sigmoid(eta)
        residual = 1.0 - pi
        weight = pi * residual

        log_likelihood -= (
            torch.nn.functional.softplus(
                -eta
            ).sum()
        )

        J_sum.add_(
            Xc.transpose(0, 1)
            @ (
                weight.unsqueeze(1)
                * Xc
            )
        )

        score_c = (
            residual.unsqueeze(1)
            * Xc
        )

        pair_c = local_pairs[
            start:end
        ]

        endpoint_score_sum.index_add_(
            0,
            pair_c[:, 0],
            score_c,
        )

        endpoint_score_sum.index_add_(
            0,
            pair_c[:, 1],
            score_c,
        )

    J_hat = J_sum / K_d

    phi = endpoint_score_sum / max(
        N_d - 1,
        1,
    )

    I_iid = (
        4.0
        * (
            phi.transpose(0, 1)
            @ phi
        )
        / N_d
    )

    if nw_bandwidth < 0:
        q_n = int(
            math.floor(
                4.0
                * (N_d / 100.0)
                ** (2.0 / 9.0)
            )
        )

        q_n = max(
            1,
            min(q_n, N_d - 1),
        )
    else:
        q_n = max(
            0,
            min(
                int(nw_bandwidth),
                N_d - 1,
            ),
        )

    phi_nw = (
        phi
        - phi.mean(
            dim=0,
            keepdim=True,
        )
        if center_nw
        else phi
    )

    I_nw = (
        phi_nw.transpose(0, 1)
        @ phi_nw
    ) / N_d

    for h in range(
        1,
        q_n + 1,
    ):
        gamma_h = (
            phi_nw[h:].transpose(0, 1)
            @ phi_nw[:-h]
        ) / N_d

        bartlett = (
            1.0
            - h / (q_n + 1.0)
        )

        I_nw.add_(
            bartlett
            * (
                gamma_h
                + gamma_h.transpose(0, 1)
            )
        )

    I_nw.mul_(4.0)

    eye = torch.eye(
        p_dim,
        dtype=acc_dtype,
        device=X_sub.device,
    )

    J_reg = (
        J_hat
        + float(ridge) * eye
    )

    solution_iid = torch.linalg.solve(
        J_reg,
        I_iid,
    )

    solution_nw = torch.linalg.solve(
        J_reg,
        I_nw,
    )

    trace_iid = float(
        torch.trace(
            solution_iid
        ).item()
    )

    trace_nw = float(
        torch.trace(
            solution_nw
        ).item()
    )

    bias_iid = float(
        (K_d / N_d)
        * trace_iid
    )

    bias_nw = float(
        (K_d / N_d)
        * trace_nw
    )

    ll = float(
        log_likelihood.item()
    )

    minus2 = -2.0 * ll

    eigvals = torch.linalg.eigvalsh(
        J_hat
    )

    eig_min = float(
        eigvals.min().item()
    )

    eig_max = float(
        eigvals.max().item()
    )

    cond = (
        eig_max / eig_min
        if eig_min > 0
        else math.inf
    )

    return {
        "log_likelihood": ll,
        "minus2_log_likelihood": minus2,
        "trace_jinv_i_iid": trace_iid,
        "trace_jinv_i_nw": trace_nw,
        "bias_iid": bias_iid,
        "bias_nw": bias_nw,
        "ic_iid": minus2 + 2.0 * bias_iid,
        "plic": minus2 + 2.0 * bias_nw,
        "nw_bandwidth_used": int(q_n),
        "j_eig_min": eig_min,
        "j_eig_max": eig_max,
        "j_condition": cond,
    }



def make_cv_validation_folds(
    n_rows: int,
    method: str,
    n_splits: int,
    shuffle: bool,
    seed: int,
) -> list[np.ndarray]:
    """
    Return validation-row indices for each fold.

    kfold:
        Each row appears in exactly one validation fold.
    loocv:
        One validation row per fold.
    """
    if n_rows < 2:
        raise ValueError(
            f"Cross-validation requires at least 2 rows, got {n_rows}."
        )

    indices = np.arange(
        n_rows,
        dtype=np.int64,
    )

    if method == "loocv":
        return [
            indices[i:i + 1]
            for i in range(n_rows)
        ]

    if method != "kfold":
        raise ValueError(
            f"Unsupported CV method: {method}"
        )

    if n_splits < 2 or n_splits > n_rows:
        raise ValueError(
            f"k-fold requires 2 <= cv_folds <= n_rows; "
            f"got cv_folds={n_splits}, n_rows={n_rows}."
        )

    if shuffle:
        rng = np.random.default_rng(
            int(seed)
        )
        rng.shuffle(indices)

    return [
        fold.astype(
            np.int64,
            copy=False,
        )
        for fold in np.array_split(
            indices,
            n_splits,
        )
    ]


def complement_indices(
    n_rows: int,
    validation_indices: np.ndarray,
) -> np.ndarray:
    mask = np.ones(
        n_rows,
        dtype=bool,
    )
    mask[validation_indices] = False
    return np.flatnonzero(
        mask
    ).astype(
        np.int64,
        copy=False,
    )


def evaluate_cv_for_lambda(
    torch: Any,
    X: Any,
    groups: Any,
    lam: float,
    config: FistaConfig,
    validation_folds: list[np.ndarray],
    gpu_id: int | None = None,
    lambda_index: int | None = None,
) -> dict[str, Any]:
    """
    Fit one cold-start model per fold and evaluate the unpenalized
    logistic loss on held-out rows.
    """
    n_rows = int(X.shape[0])
    n_folds = len(validation_folds)

    fold_losses: list[float] = []
    fold_sizes: list[int] = []
    fold_elapsed_sec: list[float] = []
    fold_fit_sec: list[float] = []
    fold_index_select_sec: list[float] = []

    converged_folds = 0
    total_weighted_loss = 0.0

    cv_start = perf_counter()

    prefix = (
        f"[GPU {gpu_id}] "
        if gpu_id is not None
        else ""
    )

    for fold_index, validation_np in enumerate(
        validation_folds,
        start=1,
    ):
        fold_start = perf_counter()

        print(
            f"{prefix}"
            f"lambda[{lambda_index}] "
            f"fold {fold_index}/{n_folds} started",
            flush=True,
        )

        # --------------------------------------------------
        # CPU側インデックス作成
        # --------------------------------------------------
        index_start = perf_counter()

        train_np = complement_indices(
            n_rows=n_rows,
            validation_indices=validation_np,
        )

        train_indices = torch.as_tensor(
            train_np,
            dtype=torch.long,
            device=X.device,
        )

        validation_indices = torch.as_tensor(
            validation_np,
            dtype=torch.long,
            device=X.device,
        )

        torch.cuda.synchronize(X.device)

        index_elapsed = perf_counter() - index_start

        # --------------------------------------------------
        # GPU上で行抽出
        # index_selectは新しいテンソルを作る
        # --------------------------------------------------
        select_start = perf_counter()

        X_train = X.index_select(
            0,
            train_indices,
        )

        X_validation = X.index_select(
            0,
            validation_indices,
        )

        torch.cuda.synchronize(X.device)

        select_elapsed = perf_counter() - select_start

        # --------------------------------------------------
        # FISTA
        # --------------------------------------------------
        fit_start = perf_counter()

        theta_fold, fit_info = fit_group_lasso_fista(
            torch=torch,
            X=X_train,
            groups=groups,
            lam=lam,
            config=config,
        )

        torch.cuda.synchronize(X.device)

        fit_elapsed = perf_counter() - fit_start

        # --------------------------------------------------
        # 検証損失
        # --------------------------------------------------
        validation_start = perf_counter()

        with torch.no_grad():
            validation_loss = smooth_loss(
                torch=torch,
                X=X_validation,
                theta=theta_fold,
                ridge=0.0,
            )

        loss_value = float(
            validation_loss.item()
        )

        torch.cuda.synchronize(X.device)

        validation_elapsed = (
            perf_counter() - validation_start
        )

        validation_size = int(
            validation_np.size
        )

        if not math.isfinite(loss_value):
            raise FloatingPointError(
                "Non-finite validation loss at "
                f"fold {fold_index}."
            )

        fold_losses.append(loss_value)
        fold_sizes.append(validation_size)
        fold_fit_sec.append(fit_elapsed)
        fold_index_select_sec.append(select_elapsed)

        total_weighted_loss += (
            loss_value * validation_size
        )

        if bool(fit_info["converged"]):
            converged_folds += 1

        fold_elapsed = (
            perf_counter() - fold_start
        )

        fold_elapsed_sec.append(
            fold_elapsed
        )

        elapsed_total = (
            perf_counter() - cv_start
        )

        average_fold_sec = (
            elapsed_total / fold_index
        )

        remaining_fold_sec = (
            average_fold_sec
            * (n_folds - fold_index)
        )

        print(
            f"{prefix}"
            f"lambda[{lambda_index}] "
            f"fold {fold_index}/{n_folds} completed | "
            f"loss={loss_value:.8g} | "
            f"iter={fit_info['iterations']} | "
            f"converged={fit_info['converged']} | "
            f"indices={index_elapsed:.2f}s | "
            f"select={select_elapsed:.2f}s | "
            f"fit={fit_elapsed:.2f}s | "
            f"validation={validation_elapsed:.2f}s | "
            f"fold_total={fold_elapsed:.2f}s | "
            f"ETA={remaining_fold_sec / 60.0:.1f}min",
            flush=True,
        )

        del theta_fold
        del X_train
        del X_validation
        del train_indices
        del validation_indices

    total_validation_rows = int(
        sum(fold_sizes)
    )

    cv_loss_mean = (
        total_weighted_loss
        / total_validation_rows
    )

    if len(fold_losses) > 1:
        cv_loss_std = float(
            np.std(
                np.asarray(
                    fold_losses,
                    dtype=np.float64,
                ),
                ddof=1,
            )
        )

        cv_loss_se = (
            cv_loss_std
            / math.sqrt(len(fold_losses))
        )
    else:
        cv_loss_std = 0.0
        cv_loss_se = 0.0

    cv_elapsed_sec = (
        perf_counter() - cv_start
    )

    print(
        f"{prefix}"
        f"lambda[{lambda_index}] CV completed | "
        f"mean_loss={cv_loss_mean:.8g} | "
        f"total={cv_elapsed_sec / 60.0:.2f}min | "
        f"fit_total={sum(fold_fit_sec) / 60.0:.2f}min | "
        f"select_total="
        f"{sum(fold_index_select_sec):.2f}s",
        flush=True,
    )

    return {
        "cv_loss_mean": float(cv_loss_mean),
        "cv_loss_std": float(cv_loss_std),
        "cv_loss_se": float(cv_loss_se),
        "cv_total_validation_rows": total_validation_rows,
        "cv_converged_folds": converged_folds,
        "cv_fold_losses": fold_losses,
        "cv_fold_elapsed_sec": fold_elapsed_sec,
        "cv_fold_fit_sec": fold_fit_sec,
        "cv_fold_index_select_sec": fold_index_select_sec,
        "cv_elapsed_sec": float(cv_elapsed_sec),
    }


def gpu_worker(
    gpu_id: int,
    lambda_indices: list[int],
    lambdas: np.ndarray,
    x_npy: str,
    dim: int,
    order: int,
    output_dir: str,
    config_dict: dict[str, Any],
    transfer_chunk_rows: int,
    cv_method: str,
    cv_folds: int,
    cv_shuffle: bool,
    cv_seed: int,
) -> list[dict[str, Any]]:
    os.environ.setdefault(
        "OMP_NUM_THREADS",
        "1",
    )
    os.environ.setdefault(
        "MKL_NUM_THREADS",
        "1",
    )
    os.environ.setdefault(
        "OPENBLAS_NUM_THREADS",
        "1",
    )
    os.environ.setdefault(
        "NUMEXPR_NUM_THREADS",
        "1",
    )

    import torch

    config = FistaConfig(
        **config_dict
    )

    device = torch.device(
        f"cuda:{gpu_id}"
    )

    torch.cuda.set_device(
        device
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    np_dtype = (
        np.float32
        if config.dtype == "float32"
        else np.float64
    )

    torch_dtype = torch_dtype_from_name(
        torch=torch,
        dtype_name=config.dtype,
    )

    X_cpu = np.load(
        x_npy,
        mmap_mode="r",
        allow_pickle=False,
    )

    if X_cpu.ndim != 2:
        raise ValueError(
            f"X must be 2-D, got {X_cpu.shape}"
        )

    expected_features = (
        order
        * 4
        * dim
        * dim
    )

    if X_cpu.shape[1] != expected_features:
        raise ValueError(
            f"X has {X_cpu.shape[1]} features, "
            f"expected {expected_features} "
            f"for dim={dim}, order={order}."
        )

    if X_cpu.dtype != np_dtype:
        print(
            f"[GPU {gpu_id}] "
            f"X dtype {X_cpu.dtype} differs "
            f"from requested {np.dtype(np_dtype)}; "
            "conversion will occur during GPU transfer."
        )

    groups_np = make_edge_lag_groups(
        dim=dim,
        order=order,
    )

    groups = torch.as_tensor(
        groups_np,
        dtype=torch.long,
        device=device,
    )

    n_endpoints = infer_endpoint_count_from_pair_rows(
        int(X_cpu.shape[0])
    )

    local_pairs = make_local_pair_indices(
        torch=torch,
        n_endpoints=n_endpoints,
        device=device,
    )

    if int(local_pairs.shape[0]) != int(X_cpu.shape[0]):
        raise ValueError(
            "Pair-index row count does not match X."
        )

    (
        free_before,
        total_memory,
    ) = torch.cuda.mem_get_info(
        device
    )

    transfer_start = perf_counter()

    X = copy_numpy_to_gpu(
        torch=torch,
        X_cpu=X_cpu,
        device=device,
        dtype=torch_dtype,
        chunk_rows=transfer_chunk_rows,
    )

    torch.cuda.synchronize(
        device
    )

    transfer_sec = (
        perf_counter()
        - transfer_start
    )

    free_after, _ = torch.cuda.mem_get_info(
        device
    )

    print(
        f"[GPU {gpu_id}] "
        f"X={tuple(X.shape)}, "
        f"dtype={X.dtype}, "
        f"transfer={transfer_sec:.2f}s, "
        f"VRAM used="
        f"{(free_before - free_after) / 1024**3:.2f} GiB, "
        f"free="
        f"{free_after / 1024**3:.2f}/"
        f"{total_memory / 1024**3:.2f} GiB"
    )

    validation_folds = make_cv_validation_folds(
        n_rows=int(X.shape[0]),
        method=cv_method,
        n_splits=cv_folds,
        shuffle=cv_shuffle,
        seed=cv_seed,
    )

    print(
        f"[GPU {gpu_id}] CV method={cv_method}, "
        f"folds={len(validation_folds)}, "
        f"shuffle={cv_shuffle}, seed={cv_seed}"
    )

    theta_dir = (
        Path(output_dir)
        / "theta"
    )

    theta_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[
        dict[str, Any]
    ] = []

    for lambda_index in lambda_indices:
        lam = float(
            lambdas[lambda_index]
        )

        print(
            f"[GPU {gpu_id}] "
            f"lambda[{lambda_index}]="
            f"{lam:.12g}"
        )

        try:
            cv_info = evaluate_cv_for_lambda(
                torch=torch,
                X=X,
                groups=groups,
                lam=lam,
                config=config,
                validation_folds=validation_folds,
                gpu_id=gpu_id,
                lambda_index=lambda_index,
            )

            print(
                f"[GPU {gpu_id}] "
                f"lambda[{lambda_index}] CV loss="
                f"{cv_info['cv_loss_mean']:.8g}"
            )

            # CV評価後、保存用係数を全データでcold-start再推定する。
            theta, info = fit_group_lasso_fista(
                torch=torch,
                X=X,
                groups=groups,
                lam=lam,
                config=config,
            )

            (
                support,
                threshold,
            ) = active_support(
                torch=torch,
                theta=theta,
                groups=groups,
                abs_tol=config.support_abs_tol,
                rel_tol=config.support_rel_tol,
            )

            theta_cpu = (
                theta.detach()
                .cpu()
                .numpy()
            )

            theta_filename = (
                f"lambda_"
                f"{lambda_index:03d}_"
                f"value_{lam:.12g}_"
                f"gpu{gpu_id}_lasso.npy"
            )

            theta_path = (
                theta_dir
                / theta_filename
            )

            # np.save(
            #     theta_path,
            #     theta_cpu,
            #     allow_pickle=False,
            # )

            ic_values: dict[str, Any] = {}
            refit_filename = ""
            ic_error = ""

            if config.compute_ic:
                try:
                    feature_indices_np = (
                        groups_np[support]
                        .reshape(-1)
                    )

                    feature_indices = torch.as_tensor(
                        feature_indices_np,
                        dtype=torch.long,
                        device=device,
                    )

                    X_sub = X.index_select(
                        1,
                        feature_indices,
                    )

                    (
                        theta_refit,
                        refit_info,
                    ) = refit_selected_support_lbfgs(
                        torch=torch,
                        X_sub=X_sub,
                        ridge=config.refit_ridge,
                        max_iter=config.refit_max_iter,
                        tolerance_grad=(
                            config.refit_tolerance_grad
                        ),
                        tolerance_change=(
                            config.refit_tolerance_change
                        ),
                        history_size=(
                            config.refit_history_size
                        ),
                    )

                    ic_values = tic_hac_diagnostics_gpu(
                        torch=torch,
                        X_sub=X_sub,
                        theta=theta_refit,
                        local_pairs=local_pairs,
                        ridge=config.ic_ridge,
                        nw_bandwidth=config.nw_bandwidth,
                        center_nw=config.nw_center,
                        chunk_rows=config.ic_chunk_rows,
                    )

                    refit_filename = (
                        f"lambda_{lambda_index:03d}_"
                        f"value_{lam:.12g}_"
                        f"gpu{gpu_id}_refit.npy"
                    )

                    refit_path = (
                        theta_dir
                        / refit_filename
                    )

                    # np.save(
                    #     refit_path,
                    #     theta_refit.detach()
                    #     .cpu()
                    #     .numpy(),
                    #     allow_pickle=False,
                    # )

                    refit_filename = str(
                        refit_path
                    )

                    ic_values.update(
                        {
                            "refit_converged": bool(
                                refit_info["converged"]
                            ),
                            "refit_iterations": int(
                                refit_info["iterations"]
                            ),
                            "refit_elapsed_sec": float(
                                refit_info["elapsed_sec"]
                            ),
                            "refit_loss_mean": float(
                                refit_info["loss_mean"]
                            ),
                            "refit_grad_norm": float(
                                refit_info["grad_norm"]
                            ),
                        }
                    )

                    del theta_refit
                    del X_sub
                    del feature_indices

                except Exception:
                    ic_error = traceback.format_exc()

            result = LambdaResult(
                lambda_index=lambda_index,
                lambda_value=lam,
                gpu_id=gpu_id,
                converged=bool(
                    info["converged"]
                ),
                iterations=int(
                    info["iterations"]
                ),
                elapsed_sec=float(
                    info["elapsed_sec"]
                ),
                final_objective=float(
                    info["final_objective"]
                ),
                final_loss=float(
                    info["final_loss"]
                ),
                final_penalty_unscaled=float(
                    info["final_penalty_unscaled"]
                ),
                final_ridge_penalty_unscaled=float(
                    info["final_ridge_penalty_unscaled"]
                ),
                final_L=float(
                    info["final_L"]
                ),
                relative_step=float(
                    info["relative_step"]
                ),
                gradient_mapping_norm=float(
                    info["gradient_mapping_norm"]
                ),
                theta_norm=float(
                    np.linalg.norm(
                        theta_cpu
                    )
                ),
                theta_maxabs=float(
                    np.max(
                        np.abs(theta_cpu),
                        initial=0.0,
                    )
                ),
                active_groups=int(
                    support.size
                ),
                support_threshold=float(
                    threshold
                ),
                support_string=",".join(
                    map(
                        str,
                        support.tolist(),
                    )
                ),
                theta_file=str(
                    theta_path
                ),
                cv_method=str(
                    cv_method
                ),
                cv_n_splits=int(
                    len(validation_folds)
                ),
                cv_loss_mean=float(
                    cv_info["cv_loss_mean"]
                ),
                cv_loss_std=float(
                    cv_info["cv_loss_std"]
                ),
                cv_loss_se=float(
                    cv_info["cv_loss_se"]
                ),
                cv_total_validation_rows=int(
                    cv_info[
                        "cv_total_validation_rows"
                    ]
                ),
                cv_converged_folds=int(
                    cv_info[
                        "cv_converged_folds"
                    ]
                ),
                cv_fold_losses=",".join(
                    f"{value:.17g}"
                    for value in cv_info[
                        "cv_fold_losses"
                    ]
                ),
                refit_file=refit_filename,
                refit_converged=bool(
                    ic_values.get(
                        "refit_converged",
                        False,
                    )
                ),
                refit_iterations=int(
                    ic_values.get(
                        "refit_iterations",
                        0,
                    )
                ),
                refit_elapsed_sec=float(
                    ic_values.get(
                        "refit_elapsed_sec",
                        math.nan,
                    )
                ),
                refit_loss_mean=float(
                    ic_values.get(
                        "refit_loss_mean",
                        math.nan,
                    )
                ),
                refit_grad_norm=float(
                    ic_values.get(
                        "refit_grad_norm",
                        math.nan,
                    )
                ),
                log_likelihood=float(
                    ic_values.get(
                        "log_likelihood",
                        math.nan,
                    )
                ),
                minus2_log_likelihood=float(
                    ic_values.get(
                        "minus2_log_likelihood",
                        math.nan,
                    )
                ),
                trace_jinv_i_iid=float(
                    ic_values.get(
                        "trace_jinv_i_iid",
                        math.nan,
                    )
                ),
                trace_jinv_i_nw=float(
                    ic_values.get(
                        "trace_jinv_i_nw",
                        math.nan,
                    )
                ),
                bias_iid=float(
                    ic_values.get(
                        "bias_iid",
                        math.nan,
                    )
                ),
                bias_nw=float(
                    ic_values.get(
                        "bias_nw",
                        math.nan,
                    )
                ),
                ic_iid=float(
                    ic_values.get(
                        "ic_iid",
                        math.nan,
                    )
                ),
                plic=float(
                    ic_values.get(
                        "plic",
                        math.nan,
                    )
                ),
                nw_bandwidth_used=int(
                    ic_values.get(
                        "nw_bandwidth_used",
                        0,
                    )
                ),
                j_eig_min=float(
                    ic_values.get(
                        "j_eig_min",
                        math.nan,
                    )
                ),
                j_eig_max=float(
                    ic_values.get(
                        "j_eig_max",
                        math.nan,
                    )
                ),
                j_condition=float(
                    ic_values.get(
                        "j_condition",
                        math.nan,
                    )
                ),
                ic_error=ic_error,
            )

            del theta
            del theta_cpu

        except Exception:
            result = LambdaResult(
                lambda_index=lambda_index,
                lambda_value=lam,
                gpu_id=gpu_id,
                converged=False,
                iterations=0,
                elapsed_sec=0.0,
                final_objective=math.nan,
                final_loss=math.nan,
                final_penalty_unscaled=math.nan,
                final_ridge_penalty_unscaled=math.nan,
                final_L=math.nan,
                relative_step=math.nan,
                gradient_mapping_norm=math.nan,
                theta_norm=math.nan,
                theta_maxabs=math.nan,
                active_groups=0,
                support_threshold=math.nan,
                support_string="",
                theta_file="",
                cv_method=str(cv_method),
                cv_n_splits=int(
                    len(validation_folds)
                ),
                error=traceback.format_exc(),
            )

        result_dict = asdict(
            result
        )

        result_dict[
            "x_transfer_sec_for_gpu"
        ] = transfer_sec

        results.append(
            result_dict
        )

        # pd.DataFrame(
        #     results
        # ).sort_values(
        #     "lambda_index"
        # ).to_csv(
        #     Path(output_dir)
        #     / f"gpu_{gpu_id}_partial.csv",
        #     index=False,
        # )

    del X
    del groups
    del local_pairs
    del X_cpu

    torch.cuda.empty_cache()

    return results


def load_lambda_grid(
    args: argparse.Namespace,
    n_rows: int,
) -> np.ndarray:
    if args.lambda_npy is not None:
        lambda_path = (
            args.lambda_npy
            .expanduser()
            .resolve()
        )

        if not lambda_path.exists():
            raise FileNotFoundError(
                f"Lambda file does not exist: "
                f"{lambda_path}"
            )

        lambdas = np.asarray(
            np.load(
                lambda_path,
                allow_pickle=False,
            ),
            dtype=np.float64,
        ).reshape(-1)

    else:
        lambdas = np.logspace(
            args.lambda_log10_max,
            args.lambda_log10_min,
            args.num_lambdas,
        )

        if args.lambda_scale_by_nrows:
            lambdas = (
                lambdas / n_rows
            )

    if lambdas.size == 0:
        raise ValueError(
            "Lambda grid is empty."
        )

    if not np.isfinite(
        lambdas
    ).all():
        raise ValueError(
            "Lambda grid contains NaN or Inf."
        )

    if np.any(
        lambdas < 0
    ):
        raise ValueError(
            "All lambda values must be nonnegative."
        )

    return lambdas


def main() -> None:
    args = parse_args()

    validate_basic_arguments(
        args
    )

    args.x_npy = (
        args.x_npy
        .expanduser()
        .resolve()
    )

    args.output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if args.lambda_npy is not None:
        args.lambda_npy = (
            args.lambda_npy
            .expanduser()
            .resolve()
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_build_metadata = prepare_x(
        args
    )

    X_meta = np.load(
        args.x_npy,
        mmap_mode="r",
        allow_pickle=False,
    )

    validate_x_array(
        X=X_meta,
        dim=args.dim,
        order=args.order,
        x_path=args.x_npy,
    )

    if (
        args.cv_method == "kfold"
        and args.cv_folds > X_meta.shape[0]
    ):
        raise ValueError(
            f"--cv-folds={args.cv_folds} exceeds "
            f"the number of rows {X_meta.shape[0]}."
        )

    gpu_ids = [
        int(x.strip())
        for x in args.gpus.split(",")
        if x.strip()
    ]

    if not gpu_ids:
        raise ValueError(
            "At least one GPU ID is required."
        )

    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(
            "Duplicate GPU IDs were specified: "
            f"{gpu_ids}"
        )

    lambdas = load_lambda_grid(
        args=args,
        n_rows=X_meta.shape[0],
    )

    # np.save(
    #     args.output_dir
    #     / "lambdas.npy",
    #     lambdas,
    #     allow_pickle=False,
    # )

    config = FistaConfig(
        max_iter=args.max_iter,
        tol=args.tol,
        pg_tol=args.pg_tol,
        initial_L=args.initial_L,
        line_search_factor=(
            args.line_search_factor
        ),
        max_line_search_steps=(
            args.max_line_search_steps
        ),
        support_abs_tol=(
            args.support_abs_tol
        ),
        support_rel_tol=(
            args.support_rel_tol
        ),
        objective_check_every=(
            args.objective_check_every
        ),
        dtype=args.dtype,
        compute_ic=False,
        fista_ridge=args.fista_ridge,
        refit_ridge=args.refit_ridge,
        refit_max_iter=args.refit_max_iter,
        refit_tolerance_grad=(
            args.refit_tolerance_grad
        ),
        refit_tolerance_change=(
            args.refit_tolerance_change
        ),
        refit_history_size=(
            args.refit_history_size
        ),
        ic_ridge=args.ic_ridge,
        nw_bandwidth=args.nw_bandwidth,
        nw_center=args.nw_center,
        ic_chunk_rows=args.ic_chunk_rows,
    )

    assignments = split_indices_round_robin(
        n_items=len(lambdas),
        gpu_ids=gpu_ids,
    )

    metadata = {
        "args": {
            key: (
                str(value)
                if isinstance(
                    value,
                    Path,
                )
                else value
            )
            for key, value
            in vars(args).items()
        },
        "x_build": x_build_metadata,
        "fista_config": asdict(config),
        "X_shape": list(
            X_meta.shape
        ),
        "X_dtype": str(
            X_meta.dtype
        ),
        "X_size_gib": float(
            X_meta.nbytes
            / 1024**3
        ),
        "lambdas": lambdas.tolist(),
        "assignments": assignments,
        "environment": {
            "OMP_NUM_THREADS": os.environ.get(
                "OMP_NUM_THREADS"
            ),
            "MKL_NUM_THREADS": os.environ.get(
                "MKL_NUM_THREADS"
            ),
            "OPENBLAS_NUM_THREADS": os.environ.get(
                "OPENBLAS_NUM_THREADS"
            ),
            "NUMEXPR_NUM_THREADS": os.environ.get(
                "NUMEXPR_NUM_THREADS"
            ),
            "CUDA_VISIBLE_DEVICES": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
        },
    }

    config_path = (
        args.output_dir
        / "config.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(
            metadata,
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"X path: {args.x_npy}"
    )
    print(
        f"X shape: {X_meta.shape}"
    )
    print(
        f"X dtype: {X_meta.dtype}"
    )
    print(
        f"X size: "
        f"{X_meta.nbytes / 1024**3:.2f} GiB"
    )
    print(
        f"Lambda count: {len(lambdas)}"
    )
    print(
        f"GPU assignments: {assignments}"
    )

    del X_meta
    gc.collect()

    all_results: list[
        dict[str, Any]
    ] = []

    start_all = perf_counter()

    context = get_context(
        "spawn"
    )

    with ProcessPoolExecutor(
        max_workers=len(gpu_ids),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                gpu_worker,
                gpu_id,
                assignments[gpu_id],
                lambdas,
                str(args.x_npy),
                args.dim,
                args.order,
                str(args.output_dir),
                asdict(config),
                args.x_transfer_chunk_rows,
                args.cv_method,
                args.cv_folds,
                args.cv_shuffle,
                args.cv_seed,
            ): gpu_id
            for gpu_id in gpu_ids
        }

        completed = 0

        for future in as_completed(
            futures
        ):
            gpu_id = futures[
                future
            ]

            try:
                worker_results = (
                    future.result()
                )

            except Exception:
                error_path = (
                    args.output_dir
                    / (
                        f"gpu_{gpu_id}_"
                        "fatal_error.txt"
                    )
                )

                error_path.write_text(
                    traceback.format_exc(),
                    encoding="utf-8",
                )

                print(
                    f"GPU {gpu_id} worker failed. "
                    f"See {error_path}"
                )

                continue

            all_results.extend(
                worker_results
            )

            completed += len(
                worker_results
            )

            # if (
            #     completed
            #     % args.save_every
            #     == 0
            #     or completed
            #     == len(lambdas)
            # ):
            #     pd.DataFrame(
            #         all_results
            #     ).sort_values(
            #         "lambda_index"
            #     ).to_csv(
            #         args.output_dir
            #         / "gpu_fista_results.csv",
            #         index=False,
            #     )

    elapsed_all = (
        perf_counter()
        - start_all
    )

    if all_results:
        result_df = pd.DataFrame(
            all_results
        ).sort_values(
            "lambda_index"
        )
    else:
        result_df = pd.DataFrame(
            columns=[
                field.name
                for field
                in LambdaResult
                .__dataclass_fields__
                .values()
            ]
        )

    result_path = (
        args.output_dir
        / "gpu_fista_results.csv"
    )

    # result_df.to_csv(
    #     result_path,
    #     index=False,
    # )

    if (
        len(result_df)
        and "cv_loss_mean" in result_df.columns
    ):
        cv_numeric = pd.to_numeric(
            result_df["cv_loss_mean"],
            errors="coerce",
        )

        valid_cv = result_df[
            np.isfinite(
                cv_numeric
            )
        ].copy()

        if len(valid_cv):
            valid_cv["cv_loss_mean"] = pd.to_numeric(
                valid_cv["cv_loss_mean"],
                errors="coerce",
            )
            valid_cv = valid_cv.sort_values(
                [
                    "cv_loss_mean",
                    "lambda_index",
                ]
            )

            # valid_cv.to_csv(
            #     args.output_dir
            #     / "cross_validation_ranking.csv",
            #     index=False,
            # )

            best_row = (
                valid_cv.iloc[0]
                .to_dict()
            )

            with (
                args.output_dir
                / "best_model_by_cv.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as file_obj:
                json.dump(
                    {
                        key: (
                            value.item()
                            if isinstance(
                                value,
                                np.generic,
                            )
                            else value
                        )
                        for key, value
                        in best_row.items()
                    },
                    file_obj,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            print(
                "Best CV model: "
                f"lambda_index="
                f"{int(best_row['lambda_index'])}, "
                f"lambda="
                f"{float(best_row['lambda_value']):.12g}, "
                f"active_groups="
                f"{int(best_row['active_groups'])}, "
                f"CV loss="
                f"{float(best_row['cv_loss_mean']):.8g}"
            )

    if len(result_df):
        num_converged = int(
            result_df[
                "converged"
            ].fillna(
                False
            ).sum()
        )

        num_errors = int(
            (
                result_df[
                    "error"
                ]
                .fillna("")
                .astype(str)
                != ""
            ).sum()
        )

        mean_fit_sec = float(
            result_df[
                "elapsed_sec"
            ].mean()
        )

        max_fit_sec = float(
            result_df[
                "elapsed_sec"
            ].max()
        )
    else:
        num_converged = 0
        num_errors = 0
        mean_fit_sec = math.nan
        max_fit_sec = math.nan

    summary = {
        "wall_time_sec": float(
            elapsed_all
        ),
        "num_requested_lambdas": int(
            len(lambdas)
        ),
        "num_completed_results": int(
            len(result_df)
        ),
        "num_converged": num_converged,
        "num_errors": num_errors,
        "mean_fit_sec": mean_fit_sec,
        "max_fit_sec": max_fit_sec,
    }

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(
            summary,
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    print("\nFinished.")

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"Results: {result_path}"
    )


if __name__ == "__main__":
    main()