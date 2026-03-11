from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BETS = SCRIPT_DIR / "output" / "positive_epl_draw_bets.csv"
DEFAULT_DOCS = SCRIPT_DIR.parent / "docs"


def load_bets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def plot_cumulative_profit(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.copy()
    plot_df["bet_number"] = range(1, len(plot_df) + 1)
    plot_df["cumulative_profit"] = plot_df["profit"].cumsum()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        plot_df["bet_number"],
        plot_df["cumulative_profit"],
        color="#0b6e4f",
        linewidth=2.5,
    )
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.fill_between(
        plot_df["bet_number"],
        plot_df["cumulative_profit"],
        0.0,
        where=plot_df["cumulative_profit"] >= 0.0,
        color="#b7e4c7",
        alpha=0.45,
    )
    ax.fill_between(
        plot_df["bet_number"],
        plot_df["cumulative_profit"],
        0.0,
        where=plot_df["cumulative_profit"] < 0.0,
        color="#f4a261",
        alpha=0.25,
    )

    final_profit = plot_df["cumulative_profit"].iloc[-1]
    final_roi = df["profit"].mean() * 100.0
    ax.set_title(
        f"EPL draw strategy on 2025/26 holdout\n46 bets, cumulative profit {final_profit:.2f}u, ROI {final_roi:.2f}%",
        fontsize=13,
    )
    ax.set_xlabel("Bet number")
    ax.set_ylabel("Cumulative profit (units)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_roi(df: pd.DataFrame, output_path: Path) -> None:
    monthly = (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(bets=("profit", "size"), roi=("profit", "mean"), profit=("profit", "sum"))
    )

    colors = ["#0b6e4f" if roi >= 0.0 else "#bc4749" for roi in monthly["roi"]]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(monthly["month"], monthly["roi"] * 100.0, color=colors)
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_title("Monthly ROI on selected EPL draw bets (2025/26 holdout)", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("ROI (%)")
    ax.grid(axis="y", alpha=0.25)

    for bar, bets, profit in zip(bars, monthly["bets"], monthly["profit"]):
        height = bar.get_height()
        y = height + 2.0 if height >= 0.0 else height - 5.0
        va = "bottom" if height >= 0.0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"n={bets}\n{profit:.2f}u",
            ha="center",
            va=va,
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_probability_gap(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    won_mask = df["won_bet"].astype(bool)

    ax.scatter(
        df.loc[~won_mask, "market_probability"],
        df.loc[~won_mask, "predicted_probability"],
        s=df.loc[~won_mask, "selected_odds"] * 14.0,
        color="#bc4749",
        alpha=0.68,
        label="Lost bet",
    )
    ax.scatter(
        df.loc[won_mask, "market_probability"],
        df.loc[won_mask, "predicted_probability"],
        s=df.loc[won_mask, "selected_odds"] * 14.0,
        color="#0b6e4f",
        alpha=0.78,
        label="Won bet",
    )

    min_prob = min(df["market_probability"].min(), df["predicted_probability"].min()) - 0.02
    max_prob = max(df["market_probability"].max(), df["predicted_probability"].max()) + 0.02
    ax.plot([min_prob, max_prob], [min_prob, max_prob], linestyle="--", color="#444444")
    ax.set_xlim(min_prob, max_prob)
    ax.set_ylim(min_prob, max_prob)
    ax.set_title("Model probability vs market probability on retained bets", fontsize=13)
    ax.set_xlabel("Market implied probability (opening odds)")
    ax.set_ylabel("Model predicted probability")
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DEFAULT_DOCS.mkdir(parents=True, exist_ok=True)
    df = load_bets(DEFAULT_BETS)

    plot_cumulative_profit(df, DEFAULT_DOCS / "positive_epl_draw_cumulative_profit.png")
    plot_monthly_roi(df, DEFAULT_DOCS / "positive_epl_draw_monthly_roi.png")
    plot_probability_gap(df, DEFAULT_DOCS / "positive_epl_draw_probability_gap.png")

    print("Generated README figures in", DEFAULT_DOCS)


if __name__ == "__main__":
    main()
