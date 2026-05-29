import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv("bias_check.csv")

# 成功したrepだけ使う
df = df[df["ok"] == True].copy()
df = df.sort_values("rep_id").reset_index(drop=True)
df["B_hat"] = df["B_hat_plus"]

# 累積平均
m = np.arange(1, len(df) + 1)

df["cummean_B_true"] = df["B_true"].expanding().mean()
df["cummean_B_hat"] = df["B_hat"].expanding().mean()

# 累積標準誤差 → 標準偏差に修正
df["cumse_B_true"] = df["B_true"].expanding().std(ddof=1) # / np.sqrt(m)
df["cumse_B_hat"] = df["B_hat"].expanding().std(ddof=1) # / np.sqrt(m)

df[["cumse_B_true", "cumse_B_hat"]] = df[["cumse_B_true", "cumse_B_hat"]].fillna(0)

# plot
df = df.tail(100)
plt.figure(figsize=(9, 5))

plt.plot(m, df["cummean_B_true"], label="B_true")
# plt.fill_between(
#     m,
#     df["cummean_B_true"] - 1.96 * df["cumse_B_true"],
#     df["cummean_B_true"] + 1.96 * df["cumse_B_true"],
#     alpha=0.2,
# )

plt.plot(m, df["cummean_B_hat"], label="E[B_hat] estimate")
plt.fill_between(
    m,
    df["cummean_B_hat"] - 1.96 * df["cumse_B_hat"],
    df["cummean_B_hat"] + 1.96 * df["cumse_B_hat"],
    alpha=0.2,
)

plt.xlabel("number of Monte Carlo replications")
plt.ylabel("B")
plt.title("Convergence check: B_true vs E[B_hat]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bias.png")

# 数値確認
print("mean B_true =", df["B_true"].mean())
print("SE B_true   =", df["B_true"].std(ddof=1) / np.sqrt(len(df)))
print("SD B_true   =", df["B_true"].std(ddof=1))

print("mean B_hat  =", df["B_hat"].mean())
print("SE B_hat    =", df["B_hat"].std(ddof=1) / np.sqrt(len(df)))
print("SD B_hat    =", df["B_hat"].std(ddof=1))

print("difference =", df["B_hat"].mean() - df["B_true"].mean())
