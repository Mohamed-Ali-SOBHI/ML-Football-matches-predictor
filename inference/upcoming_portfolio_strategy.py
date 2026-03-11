from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_pipeline.market_data import normalize_team_name
from inference.portfolio_presets import FrozenStrategy
from train.make_dataset import (
    EloConfig,
    WINDOWS_DEFAULT,
    add_team_prematch_features,
    build_home_perspective_dataset,
    compute_match_elo,
    load_team_match_rows,
)
from train.ml_common import build_xgb_model, get_feature_cols, make_sample_weight


TRAIN_MAX_SEASON = 2024
OUTCOME_TO_INDEX = {"away_win": 0, "draw": 1, "home_win": 2}
OUTCOME_TO_PROBA_COL = {
    "away_win": "pred_away_win",
    "draw": "pred_draw",
    "home_win": "pred_home_win",
}
OUTCOME_TO_MARKET_COL = {
    "away_win": "market_away_prob_open",
    "draw": "market_draw_prob_open",
    "home_win": "market_home_prob_open",
}
OUTCOME_TO_ODDS_COL = {
    "away_win": "market_away_win_odds_open",
    "draw": "market_draw_odds_open",
    "home_win": "market_home_win_odds_open",
}


@dataclass(frozen=True)
class ModelBundle:
    train_league: str
    feature_cols: list[str]
    model: object


def infer_season_from_date(date_value: pd.Timestamp) -> int:
    return date_value.year if date_value.month >= 7 else date_value.year - 1


def load_historical_team_rows(data_dir: str) -> pd.DataFrame:
    return load_team_match_rows(data_dir)


def prepare_fixture_frame(fixtures: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "league",
        "home_team",
        "away_team",
        "home_win_odds_open",
        "draw_odds_open",
        "away_win_odds_open",
    }
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f"fixtures dataframe is missing columns: {sorted(missing)}")

    result = fixtures.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["season"] = result["date"].map(infer_season_from_date).astype(int)
    result["home_team_norm"] = result["home_team"].map(normalize_team_name)
    result["away_team_norm"] = result["away_team"].map(normalize_team_name)
    return result.sort_values(["date", "league", "home_team_norm", "away_team_norm"]).reset_index(drop=True)


def build_team_lookup(team_rows: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    rows = team_rows.copy()
    rows["team_name_norm"] = rows["team_name"].map(normalize_team_name)
    latest = (
        rows.sort_values(["season", "date"])
        .groupby(["league", "team_name_norm"], as_index=False)
        .tail(1)
        .copy()
    )

    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in latest.itertuples(index=False):
        lookup[(row.league, row.team_name_norm)] = {
            "team_id": str(row.team_id),
            "team_name": row.team_name,
        }
    return lookup


def append_future_fixtures(team_rows: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if fixtures.empty:
        return team_rows.copy(), []

    lookup = build_team_lookup(team_rows)
    latest_seen_date_by_league = (
        team_rows.groupby("league", sort=False)["date"].max().to_dict()
    )

    synthetic_rows = []
    future_match_ids: list[str] = []

    for idx, fixture in enumerate(fixtures.itertuples(index=False), start=1):
        home_info = lookup.get((fixture.league, fixture.home_team_norm))
        away_info = lookup.get((fixture.league, fixture.away_team_norm))
        if home_info is None or away_info is None:
            missing = fixture.home_team_norm if home_info is None else fixture.away_team_norm
            raise ValueError(
                f"Unable to map future fixture team to Understat team_id for league {fixture.league}: {missing!r}"
            )

        match_date = pd.Timestamp(fixture.date)
        latest_seen = latest_seen_date_by_league.get(fixture.league)
        if pd.notna(latest_seen) and match_date <= latest_seen:
            match_date = latest_seen + pd.Timedelta(minutes=idx)

        home_id = home_info["team_id"]
        away_id = away_info["team_id"]
        sorted_ids = sorted([home_id, away_id])
        match_id = f"{fixture.league} {fixture.season}_{sorted_ids[0]}_{sorted_ids[1]}_{match_date.isoformat()}"
        future_match_ids.append(match_id)

        synthetic_rows.extend(
            [
                {
                    "match_id": match_id,
                    "date": match_date,
                    "is_home": True,
                    "team_id": home_id,
                    "team_name": home_info["team_name"],
                    "result": np.nan,
                    "opponent_id": away_id,
                    "opponent_name": away_info["team_name"],
                    "team_xG": np.nan,
                    "opponent_xG": np.nan,
                    "team_deep": np.nan,
                    "opponent_deep": np.nan,
                    "team_ppda_att": np.nan,
                    "team_ppda_def": np.nan,
                    "team_win_odds_open": fixture.home_win_odds_open,
                    "draw_odds_open": fixture.draw_odds_open,
                    "opponent_win_odds_open": fixture.away_win_odds_open,
                    "season_key": f"{fixture.league} {fixture.season}",
                    "league": fixture.league,
                    "season": fixture.season,
                },
                {
                    "match_id": match_id,
                    "date": match_date,
                    "is_home": False,
                    "team_id": away_id,
                    "team_name": away_info["team_name"],
                    "result": np.nan,
                    "opponent_id": home_id,
                    "opponent_name": home_info["team_name"],
                    "team_xG": np.nan,
                    "opponent_xG": np.nan,
                    "team_deep": np.nan,
                    "opponent_deep": np.nan,
                    "team_ppda_att": np.nan,
                    "team_ppda_def": np.nan,
                    "team_win_odds_open": fixture.away_win_odds_open,
                    "draw_odds_open": fixture.draw_odds_open,
                    "opponent_win_odds_open": fixture.home_win_odds_open,
                    "season_key": f"{fixture.league} {fixture.season}",
                    "league": fixture.league,
                    "season": fixture.season,
                },
            ]
        )

    future_rows = pd.DataFrame(synthetic_rows)
    combined = pd.concat([team_rows, future_rows], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined, future_match_ids


def add_elo_features(matches: pd.DataFrame) -> pd.DataFrame:
    elo_input = matches[
        ["match_id", "league", "season", "date", "team_id", "opponent_id", "result"]
    ].rename(columns={"team_id": "home_team_id", "opponent_id": "away_team_id"})
    elo = compute_match_elo(
        elo_input,
        EloConfig(k=20.0, home_adv=60.0, season_carry=0.75),
    )
    return matches.merge(elo, on="match_id", how="left", validate="one_to_one")


def build_dataset_with_fixtures(team_rows: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    combined_rows, future_match_ids = append_future_fixtures(team_rows, fixtures)
    combined_rows = add_team_prematch_features(combined_rows, windows=WINDOWS_DEFAULT)
    dataset = build_home_perspective_dataset(combined_rows, windows=WINDOWS_DEFAULT)
    dataset = add_elo_features(dataset)
    return dataset, future_match_ids


def train_frozen_models(dataset: pd.DataFrame, strategies: list[FrozenStrategy]) -> dict[str, ModelBundle]:
    bundles: dict[str, ModelBundle] = {}
    for strategy in strategies:
        train_league = strategy.train_league
        train_df = dataset[(dataset["season"] <= TRAIN_MAX_SEASON) & dataset["target"].notna()].copy()
        if train_league:
            train_df = train_df[train_df["league"] == train_league].copy()
        if train_df.empty:
            raise ValueError(f"No training rows available for train_league={train_league or 'ALL'}")

        feature_cols = get_feature_cols(train_df)
        model = build_xgb_model(seed=42, n_estimators=500, **strategy.params)
        y_train = train_df["target"].astype(int)
        model.fit(
            train_df[feature_cols],
            y_train,
            sample_weight=make_sample_weight(y_train),
        )
        bundles[strategy.name] = ModelBundle(
            train_league=train_league,
            feature_cols=feature_cols,
            model=model,
        )
    return bundles


def add_probability_columns(scored: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    result = scored.copy()
    result["pred_home_win"] = proba[:, 2]
    result["pred_draw"] = proba[:, 1]
    result["pred_away_win"] = proba[:, 0]
    return result


def score_strategy_rows(
    future_df: pd.DataFrame,
    bundles: dict[str, ModelBundle],
    strategies: list[FrozenStrategy],
) -> pd.DataFrame:
    scored_frames = []
    for strategy in strategies:
        bundle = bundles[strategy.name]
        league_df = future_df[future_df["league"] == strategy.bet_league].copy()
        if league_df.empty:
            continue

        proba = bundle.model.predict_proba(league_df[bundle.feature_cols])
        league_df = add_probability_columns(league_df, proba)

        outcome = strategy.outcome
        market_col = OUTCOME_TO_MARKET_COL[outcome]
        odds_col = OUTCOME_TO_ODDS_COL[outcome]
        proba_col = OUTCOME_TO_PROBA_COL[outcome]

        league_df["strategy_name"] = strategy.name
        league_df["train_league"] = strategy.train_league or "ALL"
        league_df["bet_league"] = strategy.bet_league
        league_df["selected_outcome"] = outcome
        league_df["selected_odds"] = league_df[odds_col]
        league_df["predicted_probability"] = league_df[proba_col]
        league_df["market_probability"] = league_df[market_col]
        league_df["edge"] = league_df["predicted_probability"] - league_df["market_probability"]
        league_df["expected_value"] = league_df["predicted_probability"] * league_df["selected_odds"] - 1.0

        market_probs = league_df[
            ["market_home_prob_open", "market_draw_prob_open", "market_away_prob_open"]
        ].to_numpy()
        league_df["bet_is_market_favorite"] = market_probs.argmax(axis=1) == OUTCOME_TO_INDEX[outcome]

        selected_mask = (
            (league_df["expected_value"] > strategy.threshold)
            & (league_df["edge"] >= strategy.edge_min)
            & (league_df["selected_odds"] >= strategy.odds_min)
            & (league_df["selected_odds"] < strategy.odds_max)
        )
        if strategy.market_favorite_mode == "favorite":
            selected_mask &= league_df["bet_is_market_favorite"]
        elif strategy.market_favorite_mode == "nonfavorite":
            selected_mask &= ~league_df["bet_is_market_favorite"]

        league_df["recommended_bet"] = np.where(selected_mask, outcome, "")
        league_df["bet_key"] = league_df["match_id"].astype(str) + "|" + league_df["selected_outcome"].astype(str)
        scored_frames.append(league_df)

    if not scored_frames:
        return pd.DataFrame()
    return pd.concat(scored_frames, ignore_index=True).sort_values(["date", "league", "team_name", "strategy_name"])


def dedupe_recommended_bets(strategy_rows: pd.DataFrame) -> pd.DataFrame:
    recommendations = strategy_rows[strategy_rows["recommended_bet"] != ""].copy()
    if recommendations.empty:
        return recommendations

    strategy_names = (
        recommendations.groupby("bet_key", sort=False)["strategy_name"]
        .agg(lambda values: "|".join(dict.fromkeys(values)))
        .rename("strategy_names")
    )
    deduped = (
        recommendations.sort_values(["bet_key", "expected_value", "edge"], ascending=[True, False, False])
        .drop_duplicates(subset=["bet_key"], keep="first")
        .copy()
    )
    deduped = deduped.merge(strategy_names, on="bet_key", how="left", validate="one_to_one")
    return deduped


def assign_flat_stakes(
    recommendations: pd.DataFrame,
    *,
    bankroll_eur: float,
    stake_fraction: float,
    max_total_exposure_fraction: float,
) -> pd.DataFrame:
    bets = recommendations.copy()
    if bets.empty:
        bets["stake_eur"] = pd.Series(dtype="float64")
        bets["max_total_exposure_eur"] = pd.Series(dtype="float64")
        bets["potential_profit_eur_if_win"] = pd.Series(dtype="float64")
        return bets

    flat_stake = bankroll_eur * stake_fraction
    max_total = bankroll_eur * max_total_exposure_fraction
    stake = flat_stake if len(bets) * flat_stake <= max_total else max_total / len(bets)

    bets["stake_eur"] = round(stake, 2)
    bets["max_total_exposure_eur"] = round(min(len(bets) * stake, max_total), 2)
    bets["potential_profit_eur_if_win"] = ((bets["selected_odds"] - 1.0) * bets["stake_eur"]).round(2)
    return bets
