from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PORTFOLIO_BETS = SCRIPT_DIR / "output" / "positive_strategy_portfolio_bets.csv"
DEFAULT_DOCS = SCRIPT_DIR.parent / "docs"
LEAGUE_COLORS = {
    "EPL": "#0b6e4f",
    "Bundesliga": "#1d4ed8",
    "La_liga": "#8d0801",
    "Ligue_1": "#bc4749",
    "Serie_A": "#f4a261",
}
STRATEGY_COLORS = {
    "train=ALL|bet=Bundesliga|outcome=draw|odds=[4.00,10.00)|fav=nonfavorite": "#1d4ed8",
    "train=ALL|bet=EPL|outcome=draw|odds=[2.00,10.00)|fav=nonfavorite": "#0b6e4f",
    "train=ALL|bet=Serie_A|outcome=draw|odds=[2.00,10.00)|fav=nonfavorite": "#f4a261",
    "train=ALL|bet=La_liga|outcome=draw|odds=[2.20,4.00)|fav=nonfavorite": "#8d0801",
}


def load_bets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def plot_portfolio_cumulative_profit(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values("date").reset_index(drop=True).copy()
    plot_df["bet_number"] = range(1, len(plot_df) + 1)
    plot_df["cumulative_profit"] = plot_df["profit"].cumsum()

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(plot_df["bet_number"], plot_df["cumulative_profit"], color="#102a43", linewidth=2.6)
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.fill_between(
        plot_df["bet_number"],
        plot_df["cumulative_profit"],
        0.0,
        where=plot_df["cumulative_profit"] >= 0.0,
        color="#d9f0ff",
        alpha=0.8,
    )
    ax.fill_between(
        plot_df["bet_number"],
        plot_df["cumulative_profit"],
        0.0,
        where=plot_df["cumulative_profit"] < 0.0,
        color="#ffd6a5",
        alpha=0.4,
    )

    final_profit = plot_df["cumulative_profit"].iloc[-1]
    final_roi = df["profit"].mean() * 100.0
    ax.set_title(
        (
            "Validation-selected multi-strategy portfolio on 2025/26\n"
            f"{len(df)} bets, cumulative profit {final_profit:.2f}u, ROI {final_roi:.2f}%"
        ),
        fontsize=13,
    )
    ax.set_xlabel("Bet number")
    ax.set_ylabel("Cumulative profit (units)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_cumulative_by_strategy(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values("date").reset_index(drop=True).copy()
    plot_df["bet_number"] = range(1, len(plot_df) + 1)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for strategy_name, group in plot_df.groupby("strategy_name", sort=False):
        strategy_group = group.copy()
        strategy_group["strategy_cumulative_profit"] = strategy_group["profit"].cumsum()
        ax.plot(
            strategy_group["bet_number"],
            strategy_group["strategy_cumulative_profit"],
            linewidth=2.2,
            label=strategy_name.replace("train=", "").replace("|bet=", " -> "),
            color=STRATEGY_COLORS.get(strategy_name),
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_title("Cumulative profit by strategy inside the validation-selected portfolio", fontsize=13)
    ax.set_xlabel("Global bet number")
    ax.set_ylabel("Strategy cumulative profit (units)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_monthly_roi_by_league(df: pd.DataFrame, output_path: Path) -> None:
    monthly = (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
        .groupby(["month", "league"], as_index=False)
        .agg(bets=("profit", "size"), roi=("profit", "mean"), profit=("profit", "sum"))
    )
    months = list(dict.fromkeys(monthly["month"].tolist()))
    leagues = [league for league in ["Bundesliga", "EPL", "La_liga", "Ligue_1", "Serie_A"] if league in set(monthly["league"])]
    x = np.arange(len(months))
    width = 0.18 if leagues else 0.6

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    for idx, league in enumerate(leagues):
        league_monthly = monthly[monthly["league"] == league].set_index("month").reindex(months).reset_index()
        offsets = x + (idx - (len(leagues) - 1) / 2.0) * width
        bars = ax.bar(
            offsets,
            league_monthly["roi"].fillna(0.0) * 100.0,
            width=width,
            color=LEAGUE_COLORS.get(league, "#6c757d"),
            label=league,
            alpha=0.9,
        )
        for bar, bets in zip(bars, league_monthly["bets"].fillna(0).astype(int)):
            if bets == 0:
                continue
            height = bar.get_height()
            y = height + 1.5 if height >= 0.0 else height - 2.5
            va = "bottom" if height >= 0.0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                f"n={bets}",
                ha="center",
                va=va,
                fontsize=8,
            )

    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_title("Monthly ROI by league inside the validation-selected portfolio", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("ROI (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_strategy_contribution(df: pd.DataFrame, output_path: Path) -> None:
    summary = (
        df.groupby("strategy_name", as_index=False)
        .agg(
            bets=("profit", "size"),
            roi=("profit", "mean"),
            profit=("profit", "sum"),
            avg_odds=("selected_odds", "mean"),
        )
        .sort_values("profit", ascending=True)
    )
    labels = [name.replace("train=", "").replace("|bet=", " -> ") for name in summary["strategy_name"]]
    colors = [STRATEGY_COLORS.get(name, "#6c757d") for name in summary["strategy_name"]]

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    bars = ax.barh(labels, summary["profit"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_title("Strategy contribution inside the validation-selected portfolio", fontsize=13)
    ax.set_xlabel("Profit (units)")
    ax.set_ylabel("Strategy")
    ax.grid(axis="x", alpha=0.25)

    for bar, bets, roi, odds in zip(bars, summary["bets"], summary["roi"], summary["avg_odds"]):
        width = bar.get_width()
        x = width + 0.35 if width >= 0.0 else width - 0.35
        ha = "left" if width >= 0.0 else "right"
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2.0,
            f"n={bets} | ROI {roi*100:.1f}% | avg odds {odds:.2f}",
            va="center",
            ha=ha,
            fontsize=8.5,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DEFAULT_DOCS.mkdir(parents=True, exist_ok=True)
    portfolio_df = load_bets(DEFAULT_PORTFOLIO_BETS)

    plot_portfolio_cumulative_profit(portfolio_df, DEFAULT_DOCS / "portfolio_cumulative_profit.png")
    plot_portfolio_cumulative_by_strategy(portfolio_df, DEFAULT_DOCS / "portfolio_cumulative_by_strategy.png")
    plot_portfolio_monthly_roi_by_league(portfolio_df, DEFAULT_DOCS / "portfolio_monthly_roi_by_league.png")
    plot_portfolio_strategy_contribution(portfolio_df, DEFAULT_DOCS / "portfolio_strategy_contribution.png")

    print("Generated README figures in", DEFAULT_DOCS)


if __name__ == "__main__":
    main()
