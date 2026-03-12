from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


OUTCOME_TO_ODDS_COL = {
    "home_win": "home_win_odds_close",
    "draw": "draw_odds_close",
    "away_win": "away_win_odds_close",
}

OUTCOME_TO_PROB_COL = {
    "home_win": "closing_home_market_probability",
    "draw": "closing_draw_market_probability",
    "away_win": "closing_away_market_probability",
}


def _safe_inverse(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.where(values > 0.0, 1.0 / values, np.nan)


def add_clv_columns(bets_df: pd.DataFrame) -> pd.DataFrame:
    if bets_df.empty:
        return bets_df.copy()

    enriched = bets_df.copy()
    home_raw = pd.Series(_safe_inverse(enriched["home_win_odds_close"]), index=enriched.index, dtype="float64")
    draw_raw = pd.Series(_safe_inverse(enriched["draw_odds_close"]), index=enriched.index, dtype="float64")
    away_raw = pd.Series(_safe_inverse(enriched["away_win_odds_close"]), index=enriched.index, dtype="float64")
    overround = home_raw + draw_raw + away_raw

    enriched["closing_overround"] = overround
    enriched["closing_home_market_probability"] = home_raw / overround
    enriched["closing_draw_market_probability"] = draw_raw / overround
    enriched["closing_away_market_probability"] = away_raw / overround

    enriched["closing_selected_odds"] = np.nan
    enriched["closing_market_probability"] = np.nan
    for outcome, odds_col in OUTCOME_TO_ODDS_COL.items():
        mask = enriched["selected_outcome"] == outcome
        enriched.loc[mask, "closing_selected_odds"] = pd.to_numeric(
            enriched.loc[mask, odds_col],
            errors="coerce",
        )
        enriched.loc[mask, "closing_market_probability"] = pd.to_numeric(
            enriched.loc[mask, OUTCOME_TO_PROB_COL[outcome]],
            errors="coerce",
        )

    opening_selected_odds = pd.to_numeric(enriched["selected_odds"], errors="coerce")
    enriched["clv_odds_diff"] = opening_selected_odds - enriched["closing_selected_odds"]
    enriched["clv_odds_ratio"] = (opening_selected_odds / enriched["closing_selected_odds"]) - 1.0
    enriched["clv_probability_diff"] = enriched["closing_market_probability"] - pd.to_numeric(
        enriched["market_probability"],
        errors="coerce",
    )
    enriched["positive_clv"] = enriched["clv_odds_diff"] > 0.0
    return enriched


def summarize_clv(bets_df: pd.DataFrame) -> dict[str, Any]:
    total_bets = int(len(bets_df))
    if total_bets == 0:
        return {
            "matched_bet_count": 0,
            "matched_coverage": None,
            "avg_closing_odds": None,
            "avg_clv_odds_diff": None,
            "median_clv_odds_diff": None,
            "avg_clv_odds_ratio": None,
            "positive_clv_rate": None,
            "avg_clv_probability_diff": None,
            "median_clv_probability_diff": None,
        }

    matched = bets_df.dropna(subset=["closing_selected_odds"]).copy()
    matched_count = int(len(matched))
    if matched_count == 0:
        return {
            "matched_bet_count": 0,
            "matched_coverage": 0.0,
            "avg_closing_odds": None,
            "avg_clv_odds_diff": None,
            "median_clv_odds_diff": None,
            "avg_clv_odds_ratio": None,
            "positive_clv_rate": None,
            "avg_clv_probability_diff": None,
            "median_clv_probability_diff": None,
        }

    return {
        "matched_bet_count": matched_count,
        "matched_coverage": matched_count / float(total_bets),
        "avg_closing_odds": float(matched["closing_selected_odds"].mean()),
        "avg_clv_odds_diff": float(matched["clv_odds_diff"].mean()),
        "median_clv_odds_diff": float(matched["clv_odds_diff"].median()),
        "avg_clv_odds_ratio": float(matched["clv_odds_ratio"].mean()),
        "positive_clv_rate": float(matched["positive_clv"].mean()),
        "avg_clv_probability_diff": float(matched["clv_probability_diff"].mean()),
        "median_clv_probability_diff": float(matched["clv_probability_diff"].median()),
    }
