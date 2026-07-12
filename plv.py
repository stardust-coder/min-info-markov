import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_sim import Kuramoto_Model

def compute_plv_matrix(theta):
    """
    theta : np.ndarray, shape (T, D)
        Kuramotoモデルから得られた位相データ。
        T: データ長、時点数
        D: 次元数、チャンネル数、ノード数

    Returns
    -------
    plv : np.ndarray, shape (D, D)
        ペアワイズPLV行列
    """
    theta = np.asarray(theta)

    if theta.ndim != 2:
        raise ValueError("theta must have shape (T, D).")

    T, D = theta.shape

    # theta[:, i] と theta[:, j] の差を全ペアで計算
    # shape: (T, D, D)
    phase_diff = theta[:, :, None] - theta[:, None, :]

    # 時間方向 T で平均
    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))

    np.fill_diagonal(plv, 0.0)

    return plv


def make_true_adjacency(K_true, k_threshold=0.0):
    """
    K_trueから真の無向グラフを作る。

    Parameters
    ----------
    K_true : np.ndarray, shape (D, D)
        真のKuramoto結合係数行列。

    k_threshold : float
        |K_ij| > k_threshold を真のエッジとみなす。
    """
    K_true = np.asarray(K_true)

    if K_true.ndim != 2 or K_true.shape[0] != K_true.shape[1]:
        raise ValueError("K_true must be a square matrix.")

    A_true = np.abs(K_true) > k_threshold
    np.fill_diagonal(A_true, False)

    A_true = np.logical_or(A_true, A_true.T)

    return A_true


def evaluate_binary_graph(A_pred, A_true):
    """
    無向グラフなので、上三角成分 i < j のみで評価する。
    """
    if A_pred.shape != A_true.shape:
        raise ValueError("A_pred and A_true must have the same shape.")

    D = A_true.shape[0]
    idx = np.triu_indices(D, k=1)

    y_pred = A_pred[idx].astype(bool)
    y_true = A_true[idx].astype(bool)

    TP = np.sum(y_pred & y_true)
    FP = np.sum(y_pred & ~y_true)
    TN = np.sum(~y_pred & ~y_true)
    FN = np.sum(~y_pred & y_true)

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    accuracy = (TP + TN) / (TP + FP + TN + FN)

    return {
        "TP": int(TP),
        "FP": int(FP),
        "TN": int(TN),
        "FN": int(FN),
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "accuracy": accuracy,
    }


def sweep_plv_thresholds(
    theta,
    K_true,
    plv_thresholds=None,
    k_threshold=0.0,
):
    """
    複数のPLV閾値でグラフを構成し、真のK行列と比較する。

    Parameters
    ----------
    theta : np.ndarray, shape (T, D)
        Kuramoto位相データ。
        T: データ長
        D: 次元数、チャンネル数

    K_true : np.ndarray, shape (D, D)
        真の結合係数行列。対称行列を想定。

    plv_thresholds : array-like or None
        PLV閾値の候補。
        Noneなら 0.00, 0.01, ..., 1.00 を試す。

    k_threshold : float
        真のエッジ判定に使うKの閾値。
        デフォルトでは K_ij != 0 をエッジとみなす。

    Returns
    -------
    result_df : pd.DataFrame
        各PLV閾値での評価結果。

    plv : np.ndarray, shape (D, D)
        PLV行列。

    A_true : np.ndarray, shape (D, D)
        真の隣接行列。
    """
    theta = np.asarray(theta)
    K_true = np.asarray(K_true)

    T, D = theta.shape

    if K_true.shape != (D, D):
        raise ValueError(
            f"K_true must have shape ({D}, {D}), but got {K_true.shape}."
        )

    if plv_thresholds is None:
        plv_thresholds = np.linspace(0.0, 1.0, 101)

    plv = compute_plv_matrix(theta)
    A_true = make_true_adjacency(K_true, k_threshold=k_threshold)

    rows = []

    for thr in plv_thresholds:
        A_pred = plv >= thr
        np.fill_diagonal(A_pred, False)

        A_pred = np.logical_or(A_pred, A_pred.T)

        metrics = evaluate_binary_graph(A_pred, A_true)
        metrics["plv_threshold"] = float(thr)
        metrics["num_pred_edges"] = int(np.sum(np.triu(A_pred, k=1)))
        metrics["num_true_edges"] = int(np.sum(np.triu(A_true, k=1)))

        rows.append(metrics)

    result_df = pd.DataFrame(rows)

    cols = [
        "plv_threshold",
        "TP",
        "FP",
        "TN",
        "FN",
        "precision",
        "recall",
        "F1",
        "accuracy",
        "num_pred_edges",
        "num_true_edges",
    ]

    return result_df[cols], plv, A_true


def run_plv_experiment_5_trials(
    n_trials=5,
    model_N=25,
    thresholds=None,
    k_threshold=0.0,
    output_csv="plv_mean_se.csv",
    output_fig="plv_mean_se.png",
):
    """
    Kuramoto_Modelを複数試行し、PLV閾値ごとに
    precision, recall, F1 の Mean ± SE を計算・保存・プロットする。

    Parameters
    ----------
    n_trials : int
        試行回数。

    model_N : int
        Kuramoto_Modelに渡すN。
        ここでは元コードに合わせて N=25。

    thresholds : array-like or None
        PLV閾値。Noneなら 0.0, 0.1, ..., 1.0。

    k_threshold : float
        真のK行列から正解エッジを作るときの閾値。
        |K_ij| > k_threshold を真のエッジとみなす。

    output_csv : str
        Mean ± SE の集計結果を保存するCSV名。

    output_fig : str
        グラフ画像の保存名。

    Returns
    -------
    summary_df : pd.DataFrame
        閾値ごとの Mean, SE を含むDataFrame。

    all_results_df : pd.DataFrame
        全試行・全閾値の生結果。
    """

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 11)

    thresholds = np.asarray(thresholds)

    all_results = []

    for trial in range(n_trials):
        raw, K_true = Kuramoto_Model(N=model_N)

        result_df, plv_matrix, A_true = sweep_plv_thresholds(
            theta=raw,
            K_true=K_true,
            plv_thresholds=thresholds,
            k_threshold=k_threshold,
        )

        result_df = result_df.copy()
        result_df["trial"] = trial + 1

        all_results.append(result_df)

    all_results_df = pd.concat(all_results, ignore_index=True)

    metrics = ["precision", "recall", "F1"]

    summary_rows = []

    for thr, group in all_results_df.groupby("plv_threshold"):
        row = {"plv_threshold": thr}

        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)

            mean = np.mean(values)
            sd = np.std(values, ddof=1)
            se = sd / np.sqrt(len(values))

            row[f"{metric}_mean"] = mean
            row[f"{metric}_se"] = se
            row[f"{metric}_mean±se"] = f"{mean:.4f} ± {se:.4f}"

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("plv_threshold").reset_index(drop=True)

    summary_df.to_csv(output_csv, index=False)
    all_results_df.to_csv("plv_all_trials.csv", index=False)

    print(summary_df)

    # -------------------------
    # Plot: Mean ± SE
    # -------------------------
    plt.figure(figsize=(8, 5))

    for metric in metrics:
        x = summary_df["plv_threshold"]
        y = summary_df[f"{metric}_mean"]
        yerr = summary_df[f"{metric}_se"]

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=metric,
        )

    plt.xlabel("PLV threshold")
    plt.ylabel("Score")
    plt.title(f"PLV graph estimation performance, {n_trials} trials")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_fig, dpi=300)
    plt.show()

    return summary_df, all_results_df

if __name__ == "__main__":
    run_plv_experiment_5_trials(n_trials=30)

    ###単発実行

    # raw, K_true = Kuramoto_Model(N=25)

    # thresholds = np.linspace(0.0, 1.0, 11)

    # result_df, plv_matrix, A_true = sweep_plv_thresholds(
    #     theta=raw,
    #     K_true=K_true,
    #     plv_thresholds=thresholds,
    #     k_threshold=0.0,
    # )

    # print(result_df)
    # result_df.to_csv("plv.csv", index=False)