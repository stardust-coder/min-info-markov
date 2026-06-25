import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import glob
import re
import pandas as pd

DIR_NAME = "logs"
npy_names = glob.glob(DIR_NAME+"*_lasso_*.npy")
npy_names.sort()
d = 25

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
best_str = df.loc[df["plic"].idxmin(), "support_string"]

def idx_to_nodes(flat_idx):
    i,j = np.unravel_index(flat_idx, (d,d))
    return (i+1).item(), (j+1).item()

def str_to_nodes(x):
    return [idx_to_nodes(int(item)) for item in x.split(",")]

# 推定された directed edge
pred_edges = set(str_to_nodes(best_str))
print("#Num of pred edges: ", len(pred_edges))

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


with open(DIR_NAME+"metric", "w", encoding="utf-8") as f:
    line1 = f"TP: {len(TP)} FP: {len(FP)} FN: {len(FN)} TN: {len(TN)}"
    line2 = f"Precision: {precision}"
    line3 = f"Recall: {recall}"
    line4 = f"F1: {f1}"

    print(line1)
    print(line2)
    print(line3)
    print(line4)

    f.write(line1 + "\n")
    f.write(line2 + "\n")
    f.write(line3 + "\n")
    f.write(line4 + "\n")

# print("="*5)
# print("TP:", len(TP))
# print(sorted(TP))

# print("FP:", len(FP))
# print(sorted(FP))

# print()
# print(sorted(FN))

# print("TN:", len(TN))
# print(sorted(TN))