from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def bootstrap_mean_distribution(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, values.size, size=(iterations, values.size))
    return values[sample_idx].mean(axis=1)


def bootstrap_ci(
    values: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float | None, float | None, np.ndarray]:
    distribution = bootstrap_mean_distribution(values, iterations=iterations, seed=seed)
    if distribution.size == 0:
        return None, None, distribution
    alpha = (1.0 - confidence_level) / 2.0
    low = float(np.quantile(distribution, alpha))
    high = float(np.quantile(distribution, 1.0 - alpha))
    return low, high, distribution


def compute_max_drawdown(profits: np.ndarray) -> float:
    if profits.size == 0:
        return 0.0
    equity_curve = profits.cumsum()
    peaks = np.maximum.accumulate(np.concatenate(([0.0], equity_curve)))[:-1]
    drawdowns = equity_curve - peaks
    return float(drawdowns.min(initial=0.0))


def compute_longest_losing_streak(won_bets: np.ndarray) -> int:
    longest = 0
    current = 0
    for won in won_bets:
        if won:
            current = 0
            continue
        current += 1
        longest = max(longest, current)
    return longest


def summarize_bets(
    bets_df: pd.DataFrame,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if bets_df.empty:
        return {
            "bet_count": 0,
            "total_profit": 0.0,
            "roi": None,
            "roi_ci_low": None,
            "roi_ci_high": None,
            "bootstrap_prob_roi_positive": None,
            "hit_rate": None,
            "hit_rate_ci_low": None,
            "hit_rate_ci_high": None,
            "avg_odds": None,
            "avg_edge": None,
            "avg_expected_value": None,
            "max_drawdown": 0.0,
            "longest_losing_streak": 0,
            "start_date": None,
            "end_date": None,
        }

    profits = bets_df["profit"].to_numpy(dtype=float)
    won_bets = bets_df["won_bet"].astype(bool).to_numpy()
    roi_ci_low, roi_ci_high, roi_distribution = bootstrap_ci(
        profits,
        iterations=iterations,
        confidence_level=confidence_level,
        seed=seed,
    )
    hit_values = won_bets.astype(float)
    hit_ci_low, hit_ci_high, hit_distribution = bootstrap_ci(
        hit_values,
        iterations=iterations,
        confidence_level=confidence_level,
        seed=seed + 1,
    )

    return {
        "bet_count": int(len(bets_df)),
        "total_profit": float(profits.sum()),
        "roi": float(profits.mean()),
        "roi_ci_low": roi_ci_low,
        "roi_ci_high": roi_ci_high,
        "bootstrap_prob_roi_positive": (
            float((roi_distribution > 0.0).mean()) if roi_distribution.size else None
        ),
        "hit_rate": float(hit_values.mean()),
        "hit_rate_ci_low": hit_ci_low,
        "hit_rate_ci_high": hit_ci_high,
        "avg_odds": float(bets_df["selected_odds"].mean()),
        "avg_edge": float(bets_df["edge"].mean()),
        "avg_expected_value": float(bets_df["expected_value"].mean()),
        "max_drawdown": compute_max_drawdown(profits),
        "longest_losing_streak": compute_longest_losing_streak(won_bets),
        "start_date": bets_df["date"].min().strftime("%Y-%m-%d") if "date" in bets_df.columns else None,
        "end_date": bets_df["date"].max().strftime("%Y-%m-%d") if "date" in bets_df.columns else None,
    }


def group_roi_table(bets_df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    if bets_df.empty or group_col not in bets_df.columns:
        return []

    grouped = (
        bets_df.groupby(group_col, as_index=False)
        .agg(
            bets=("profit", "size"),
            roi=("profit", "mean"),
            profit=("profit", "sum"),
            hit_rate=("won_bet", "mean"),
            avg_odds=("selected_odds", "mean"),
        )
        .sort_values(["roi", "profit", "bets"], ascending=[False, False, False])
    )
    return grouped.to_dict(orient="records")


def monthly_roi_table(bets_df: pd.DataFrame) -> list[dict[str, Any]]:
    if bets_df.empty or "date" not in bets_df.columns:
        return []

    grouped = (
        bets_df.assign(month=bets_df["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(
            bets=("profit", "size"),
            roi=("profit", "mean"),
            profit=("profit", "sum"),
            hit_rate=("won_bet", "mean"),
        )
        .sort_values("month")
    )
    return grouped.to_dict(orient="records")
