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
from train.make_dataset import (
    EloConfig,
    WINDOWS_DEFAULT,
    add_team_prematch_features,
    build_home_perspective_dataset,
    compute_match_elo,
    load_team_match_rows,
)
from train.ml_common import build_xgb_model, get_feature_cols, make_sample_weight


LEAGUE = "EPL"
TRAIN_MAX_SEASON = 2024


BEST_PARAMS = {
    "max_depth": 5,
    "min_child_weight": 5.0750567663835575,
    "subsample": 0.7613001150741135,
    "colsample_bytree": 0.6352621115879286,
    "gamma": 0.5196860213418866,
    "reg_lambda": 4.067786946694503,
    "learning_rate": 0.029749107688307467,
}


@dataclass(frozen=True)
class DrawStrategy:
    expected_value_threshold: float = 0.55
    edge_min: float = 0.08
    odds_min: float = 2.0
    odds_max: float = 10.0


def infer_season_from_date(date_value: pd.Timestamp) -> int:
    return date_value.year if date_value.month >= 7 else date_value.year - 1


def load_historical_team_rows(data_dir: str) -> pd.DataFrame:
    return load_team_match_rows(data_dir)


def prepare_fixture_frame(fixtures: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
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
    result["league"] = LEAGUE
    result["season"] = result["date"].map(infer_season_from_date).astype(int)
    result["home_team_norm"] = result["home_team"].map(normalize_team_name)
    result["away_team_norm"] = result["away_team"].map(normalize_team_name)
    return result.sort_values(["date", "home_team_norm", "away_team_norm"]).reset_index(drop=True)


def build_team_lookup(team_rows: pd.DataFrame) -> dict[str, dict[str, str]]:
    epl_rows = team_rows[team_rows["league"] == LEAGUE].copy()
    epl_rows["team_name_norm"] = epl_rows["team_name"].map(normalize_team_name)
    latest = (
        epl_rows.sort_values(["season", "date"])
        .groupby("team_name_norm", as_index=False)
        .tail(1)
        .copy()
    )

    lookup: dict[str, dict[str, str]] = {}
    for row in latest.itertuples(index=False):
        lookup[row.team_name_norm] = {
            "team_id": str(row.team_id),
            "team_name": row.team_name,
        }
    return lookup


def append_future_fixtures(team_rows: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if fixtures.empty:
        return team_rows.copy(), []

    lookup = build_team_lookup(team_rows)
    latest_seen_date = team_rows[team_rows["league"] == LEAGUE]["date"].max()

    synthetic_rows = []
    future_match_ids: list[str] = []

    for idx, fixture in enumerate(fixtures.itertuples(index=False), start=1):
        home_info = lookup.get(fixture.home_team_norm)
        away_info = lookup.get(fixture.away_team_norm)
        if home_info is None or away_info is None:
            missing = fixture.home_team_norm if home_info is None else fixture.away_team_norm
            raise ValueError(f"Unable to map future fixture team to Understat team_id: {missing!r}")

        match_date = pd.Timestamp(fixture.date)
        if pd.notna(latest_seen_date) and match_date <= latest_seen_date:
            match_date = latest_seen_date + pd.Timedelta(minutes=idx)

        home_id = home_info["team_id"]
        away_id = away_info["team_id"]
        sorted_ids = sorted([home_id, away_id])
        match_id = f"{LEAGUE} {fixture.season}_{sorted_ids[0]}_{sorted_ids[1]}_{match_date.isoformat()}"
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
                    "season_key": f"{LEAGUE} {fixture.season}",
                    "league": LEAGUE,
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
                    "season_key": f"{LEAGUE} {fixture.season}",
                    "league": LEAGUE,
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


def train_frozen_model(dataset: pd.DataFrame):
    train_df = dataset[
        (dataset["league"] == LEAGUE)
        & (dataset["season"] <= TRAIN_MAX_SEASON)
        & dataset["target"].notna()
    ].copy()
    feature_cols = get_feature_cols(train_df)

    model = build_xgb_model(seed=42, n_estimators=500, **BEST_PARAMS)
    y_train = train_df["target"].astype(int)
    model.fit(
        train_df[feature_cols],
        y_train,
        sample_weight=make_sample_weight(y_train),
    )
    return model, feature_cols


def score_upcoming_matches(
    dataset: pd.DataFrame,
    future_match_ids: list[str],
    feature_cols: list[str],
    model,
    strategy: DrawStrategy,
) -> pd.DataFrame:
    future_df = dataset[dataset["match_id"].isin(future_match_ids)].copy()
    future_df = future_df.sort_values(["date", "team_name"]).reset_index(drop=True)
    future_df["target"] = pd.Series([pd.NA] * len(future_df), dtype="Int64")

    proba = model.predict_proba(future_df[feature_cols])
    future_df["pred_home_win"] = proba[:, 2]
    future_df["pred_draw"] = proba[:, 1]
    future_df["pred_away_win"] = proba[:, 0]

    future_df["draw_market_probability"] = future_df["market_draw_prob_open"]
    future_df["draw_edge"] = future_df["pred_draw"] - future_df["draw_market_probability"]
    future_df["draw_expected_value"] = future_df["pred_draw"] * future_df["market_draw_odds_open"] - 1.0

    market_probs = future_df[
        ["market_home_prob_open", "market_draw_prob_open", "market_away_prob_open"]
    ].to_numpy()
    future_df["draw_is_market_favorite"] = market_probs.argmax(axis=1) == 1

    selected_mask = (
        (future_df["draw_expected_value"] > strategy.expected_value_threshold)
        & (future_df["draw_edge"] >= strategy.edge_min)
        & (future_df["market_draw_odds_open"] >= strategy.odds_min)
        & (future_df["market_draw_odds_open"] < strategy.odds_max)
        & (~future_df["draw_is_market_favorite"])
    )
    future_df["recommended_bet"] = np.where(selected_mask, "draw", "")
    return future_df


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
    bets["potential_profit_eur_if_win"] = ((bets["market_draw_odds_open"] - 1.0) * bets["stake_eur"]).round(2)
    return bets
