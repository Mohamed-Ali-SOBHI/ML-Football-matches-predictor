from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


OUTCOME_INDEX = {"away_win": 0, "draw": 1, "home_win": 2}
OUTCOME_LABELS = np.array(["away_win", "draw", "home_win"], dtype=object)


@dataclass(frozen=True)
class StrategyFamily:
    train_league: str
    bet_league: str
    outcome: str
    odds_min: float
    odds_max: float
    market_favorite_mode: str

    @property
    def name(self) -> str:
        train_label = self.train_league or "ALL"
        bet_label = self.bet_league or "ALL"
        return (
            f"train={train_label}|bet={bet_label}|outcome={self.outcome}|"
            f"odds=[{self.odds_min:.2f},{self.odds_max:.2f})|fav={self.market_favorite_mode}"
        )


def sample_params(rng: np.random.Generator) -> dict:
    return {
        "max_depth": int(rng.integers(3, 9)),
        "min_child_weight": float(rng.uniform(1.0, 12.0)),
        "subsample": float(rng.uniform(0.55, 1.0)),
        "colsample_bytree": float(rng.uniform(0.55, 1.0)),
        "gamma": float(rng.uniform(0.0, 4.0)),
        "reg_lambda": float(rng.uniform(0.5, 8.0)),
        "learning_rate": float(rng.uniform(0.015, 0.08)),
    }


def build_base_bets(eval_df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    odds = np.column_stack(
        [
            eval_df["market_away_win_odds_open"].to_numpy(),
            eval_df["market_draw_odds_open"].to_numpy(),
            eval_df["market_home_win_odds_open"].to_numpy(),
        ]
    )
    fair_probs = 1.0 / odds
    fair_probs = fair_probs / fair_probs.sum(axis=1, keepdims=True)
    expected_value = proba * odds - 1.0
    chosen = expected_value.argmax(axis=1)
    chosen_ev = expected_value[np.arange(len(expected_value)), chosen]

    bet_df = eval_df.copy()
    bet_df["selected_outcome"] = OUTCOME_LABELS[chosen]
    bet_df["selected_odds"] = odds[np.arange(len(odds)), chosen]
    bet_df["predicted_probability"] = proba[np.arange(len(proba)), chosen]
    bet_df["market_probability"] = fair_probs[np.arange(len(fair_probs)), chosen]
    bet_df["edge"] = bet_df["predicted_probability"] - bet_df["market_probability"]
    bet_df["expected_value"] = chosen_ev
    bet_df["won_bet"] = chosen == bet_df["target"].astype(int).to_numpy()
    bet_df["profit"] = np.where(bet_df["won_bet"], bet_df["selected_odds"] - 1.0, -1.0)

    valid_mask = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    bet_df = bet_df[valid_mask].copy()

    market_probs = bet_df[
        [
            "market_home_prob_open",
            "market_draw_prob_open",
            "market_away_prob_open",
        ]
    ].to_numpy()
    market_fav_idx = market_probs.argmax(axis=1)
    selected_idx = pd.Series(bet_df["selected_outcome"]).map(OUTCOME_INDEX).to_numpy()
    bet_df["bet_is_market_favorite"] = selected_idx == market_fav_idx
    bet_df["bet_key"] = bet_df["match_id"].astype(str) + "|" + bet_df["selected_outcome"].astype(str)
    return bet_df


def apply_strategy(
    base_bets: pd.DataFrame,
    *,
    threshold: float,
    edge_min: float,
    bet_league: str,
    outcome: str,
    odds_min: float,
    odds_max: float,
    market_favorite_mode: str,
) -> pd.DataFrame:
    bets = base_bets[base_bets["expected_value"] > threshold].copy()
    if bet_league:
        bets = bets[bets["league"] == bet_league]
    bets = bets[bets["selected_outcome"] == outcome]
    bets = bets[(bets["selected_odds"] >= odds_min) & (bets["selected_odds"] < odds_max)]
    bets = bets[bets["edge"] >= edge_min]
    if market_favorite_mode == "favorite":
        bets = bets[bets["bet_is_market_favorite"]]
    elif market_favorite_mode == "nonfavorite":
        bets = bets[~bets["bet_is_market_favorite"]]
    return bets


def threshold_values(start: float, stop: float, step: float) -> list[float]:
    values = np.arange(start, stop + step / 2.0, step)
    return [round(float(value), 10) for value in values]


def parse_list_argument(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def parse_odds_ranges(raw: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for token in parse_list_argument(raw):
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid odds range {token!r}; expected MIN:MAX")
        low = float(parts[0])
        high = float(parts[1])
        if high <= low:
            raise ValueError(f"Invalid odds range {token!r}; MAX must be > MIN")
        ranges.append((low, high))
    return ranges


def summarize_bets(bets: pd.DataFrame, prefix: str) -> dict[str, float | int | None]:
    if bets.empty:
        return {
            f"{prefix}_bets": 0,
            f"{prefix}_roi": None,
            f"{prefix}_profit": 0.0,
            f"{prefix}_hit_rate": None,
            f"{prefix}_avg_odds": None,
            f"{prefix}_avg_edge": None,
            f"{prefix}_avg_ev": None,
        }

    return {
        f"{prefix}_bets": int(len(bets)),
        f"{prefix}_roi": float(bets["profit"].mean()),
        f"{prefix}_profit": float(bets["profit"].sum()),
        f"{prefix}_hit_rate": float(bets["won_bet"].mean()),
        f"{prefix}_avg_odds": float(bets["selected_odds"].mean()),
        f"{prefix}_avg_edge": float(bets["edge"].mean()),
        f"{prefix}_avg_ev": float(bets["expected_value"].mean()),
    }

