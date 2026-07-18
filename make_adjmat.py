import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import glob
import re
import pandas as pd

DIR_NAME = "logs/tmp1/"
npy_names = glob.glob(DIR_NAME+"*_lasso*.npy")
npy_names.sort()
d = 5

for i, npy_name in enumerate(npy_names[::]):
    M1 = np.load(npy_name)
    M2 = M1.reshape(d**2, 4)
    M3 = np.linalg.norm(M2, axis=1)
    M4 = M3.reshape(d, d)
    M = M4

    # ===== 元のヒートマップ =====    
    rows, cols = np.nonzero(M)
    max_abs = np.max(np.abs(M))
    norm = TwoSlopeNorm(vmin=min(-max_abs,-1e-9), vcenter=0, vmax=max(max_abs,1e-9))

    plt.imshow(M, cmap='seismic', norm=norm)
    plt.xticks(ticks=[j for j in range(d)],labels=[f"x{j+1}" for j in range(d)], rotation=90)
    plt.yticks(ticks=[j for j in range(d)],labels=[f"x{j+1}" for j in range(d)])
    plt.gca().xaxis.tick_top() 
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel("t-1")
    plt.ylabel("t")
    plt.colorbar()
    plt.title(npy_name + " (raw)", fontsize=5)
    plt.tight_layout()
    plt.savefig(DIR_NAME+str(i))

    # ===== ここから追加：{0,1} 可視化 =====
    threshold = 1e-8  # 好きに調整OK
    M_bin = (np.abs(M) < threshold).astype(int)

    plt.imshow(M_bin, cmap='gray')  # 0=黒, 1=白
    plt.xticks(ticks=[j for j in range(d)],labels=[f"x{j+1}" for j in range(d)], rotation=90)
    plt.yticks(ticks=[j for j in range(d)],labels=[f"x{j+1}" for j in range(d)])
    plt.gca().xaxis.tick_top() 
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel("t-1")
    plt.ylabel("t")
    plt.title(npy_name + " (binary)", fontsize=5)
    plt.tight_layout()
    plt.savefig(DIR_NAME+str(i)+"_bin")
    print(f"Processing {i} ... ")
    plt.clf()



edges = []
plics = []

pattern = r"edge=(\d+)_PLIC=([0-9.]+)"

for name in npy_names:
    match = re.search(pattern, name)
    if match:
        num_edge = int(match.group(1))
        plic = float(match.group(2))

        edges.append(num_edge)
        plics.append(plic)

# edge数で昇順ソート
sorted_data = sorted(zip(edges, plics))
edges_sorted, plics_sorted = zip(*sorted_data)

plt.figure(figsize=(7, 5))
plt.plot(edges_sorted, plics_sorted, "o-", linewidth=2, markersize=6)

plt.xlabel("Number of edges")
plt.ylabel("PLIC")
plt.title("PLIC vs Number of edges")
plt.grid(True)
plt.tight_layout()
plt.savefig(DIR_NAME+"graph.png")



GT_edges = []

file_path = "25dim"  # 読み込むファイル名に変更してください

with open(DIR_NAME+file_path, "r") as f:
    for line in f:
        line = line.strip()

        # 空行はスキップ
        if not line:
            continue

        src_str, dsts_str = line.split(":")
        src = int(src_str)

        dsts = dsts_str.strip().split()

        for dst_str in dsts:
            dst = int(dst_str)
            GT_edges.append((src, dst))

df = pd.read_csv(DIR_NAME + "regularization_path_ic_debug.csv")

def idx_to_nodes(flat_idx):
    i, j = np.unravel_index(flat_idx, (d, d))
    return (i + 1).item(), (j + 1).item()

def str_to_nodes(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [idx_to_nodes(int(item)) for item in str(x).split(",")]

def calc_metrics(support_str):
    # 推定された directed edge
    pred_edges = set(str_to_nodes(support_str))

    # 自己ループを除く
    pred_edges = {(i, j) for i, j in pred_edges if i != j}
    GT_directed = set(GT_edges)

    # 全候補 edge（向きあり、自己ループなし）
    all_edges = {
        (i, j)
        for i in range(1, d + 1)
        for j in range(1, d + 1)
        if i != j
    }

    TP = pred_edges & GT_directed
    FP = pred_edges - GT_directed
    FN = GT_directed - pred_edges
    TN = all_edges - (TP | FP | FN)

    precision = len(TP) / (len(TP) + len(FP)) if len(TP) + len(FP) > 0 else 0
    recall = len(TP) / (len(TP) + len(FN)) if len(TP) + len(FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "num_pred_edges": len(pred_edges),
        "TP": len(TP),
        "FP": len(FP),
        "FN": len(FN),
        "TN": len(TN),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# 全行について metric を計算
metric_rows = []

for idx, row in df.iterrows():
    metrics = calc_metrics(row["support_string"])

    metric_rows.append({
        "row_index": idx,
        "plic": row["plic"],
        "support_string": row["support_string"],
        **metrics,
    })

metric_df = pd.DataFrame(metric_rows)

# 全結果をCSVとして保存
metric_df.to_csv(DIR_NAME + "metrics_all.csv", index=False)

def write_metric_block(f, title, row):
    lines = [
        title,
        f"row_index: {row['row_index']}",
        f"plic: {row['plic']}",
        f"support_string: {row['support_string']}",
        f"#Num of pred edges: {row['num_pred_edges']}",
        f"TP: {row['TP']} FP: {row['FP']} FN: {row['FN']} TN: {row['TN']}",
        f"Precision: {row['precision']}",
        f"Recall: {row['recall']}",
        f"F1: {row['f1']}",
        "-" * 50,
    ]

    for line in lines:
        print(line)
        f.write(line + "\n")

# metric_all に全候補の結果を保存
with open(DIR_NAME + "metric_all", "w", encoding="utf-8") as f:
    for _, row in metric_df.iterrows():
        write_metric_block(f, f"Row {row['row_index']}", row)

# precision 最大・recall 最大・PLIC 最小の行を取得
best_precision_row = metric_df.loc[metric_df["precision"].idxmax()]
best_recall_row = metric_df.loc[metric_df["recall"].idxmax()]
best_plic_row = metric_df.loc[metric_df["plic"].idxmin()]
best_f1_row = metric_df.loc[metric_df["f1"].idxmax()]

# metric に要約結果を保存
with open(DIR_NAME + "metric", "w", encoding="utf-8") as f:
    write_metric_block(f, "Best Precision", best_precision_row)
    write_metric_block(f, "Best Recall", best_recall_row)
    write_metric_block(f, "Best PLIC", best_plic_row)
    write_metric_block(f, "Best F1", best_f1_row)