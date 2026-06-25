from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DIR_PATH = "/home/sukeda/min-info-markov/logs/bias_check/smallridge/"
CSV_PATH = DIR_PATH + "tic_bple_bias_check-inner5000-vM.csv"
OUT_PATH = "B_true_vs_B_hat.png"


def main():
    df = pd.read_csv(CSV_PATH)

    required_cols = [
        "rep_id",
        "n",
        "B_true",
        "B_hat_theory",  # Newey-West なし
        "B_hat_nw",      # Newey-West あり
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("rep_id").reset_index(drop=True)

    n_values = sorted(df["n"].dropna().unique())
    n_label = int(n_values[0]) if len(n_values) == 1 else "mixed"

    df["mc_rep"] = range(1, len(df) + 1)

    # 累積平均
    df["cum_B_true"] = df["B_true"].expanding().mean()
    df["cum_B_hat_no_NW"] = df["B_hat_theory"].expanding().mean()
    df["cum_B_hat_NW"] = df["B_hat_nw"].expanding().mean()

    # 累積標準偏差
    df["sd_B_true"] = df["B_true"].expanding().std(ddof=1)
    df["sd_B_hat_no_NW"] = df["B_hat_theory"].expanding().std(ddof=1)
    df["sd_B_hat_NW"] = df["B_hat_nw"].expanding().std(ddof=1)

    # r=1 では標準偏差が NaN になるので 0 にする
    sd_cols = ["sd_B_true", "sd_B_hat_no_NW", "sd_B_hat_NW"]
    df[sd_cols] = df[sd_cols].fillna(0.0)

    # df = df.tail(100)
    x = df["mc_rep"]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    series = [
        (
            "cum_B_true",
            "sd_B_true",
            r"cumulative mean of $B_{\mathrm{true}}$",
        ),
        (
            "cum_B_hat_no_NW",
            "sd_B_hat_no_NW",
            r"cumulative mean of $\widehat B$ without Newey-West",
        ),
        (
            "cum_B_hat_NW",
            "sd_B_hat_NW",
            r"cumulative mean of $\widehat B$ with Newey-West",
        ),
    ]

    for y_col, sd_col, label in series:
        y = df[y_col]
        sd = df[sd_col]

        ax.plot(x, y, linewidth=2, label=label)

        if "cum_B_hat" in y_col:
            ax.fill_between(x, y - sd, y + sd, alpha=0.15)

    ax.set_xlabel("Number of outer Monte Carlo replications")
    ax.set_ylabel("Cumulative mean")
    ax.set_title(
        rf"Sequential convergence of $\widehat B$ to $B_{{\mathrm{{true}}}}$ "
        rf"at $n={n_label}$ with ±SD bands"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    plt.show()

    print("Final cumulative means and SDs:")
    print(
        df[
            [
                "mc_rep",
                "cum_B_true",
                "cum_B_hat_no_NW",
                "cum_B_hat_NW",
                "sd_B_true",
                "sd_B_hat_no_NW",
                "sd_B_hat_NW",
            ]
        ].tail(1)
    )
    print(f"Saved plot to: {OUT_PATH}")


if __name__ == "__main__":
    main()