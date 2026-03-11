import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TEAM_MATCH_COLS = [
    "match_id",
    "date",
    "is_home",
    "team_id",
    "team_name",
    "result",
    "opponent_id",
    "opponent_name",
    # post-match stats (allowed ONLY via lag/rolling)
    "team_xG",
    "opponent_xG",
    "team_deep",
    "opponent_deep",
    "team_ppda_att",
    "team_ppda_def",
    "team_win_odds_open",
    "draw_odds_open",
    "opponent_win_odds_open",
]

WINDOWS_DEFAULT = (1, 3, 5)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "Data"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "dataset_home.csv"


def load_team_match_rows(data_dir: str) -> pd.DataFrame:
    paths = glob.glob(f"{data_dir}/**/*.csv", recursive=True)
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {data_dir!r}")

    dfs = []
    for path in paths:
        dfs.append(pd.read_csv(path, usecols=TEAM_MATCH_COLS))

    raw = pd.concat(dfs, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])

    for c in [
        "team_xG",
        "opponent_xG",
        "team_deep",
        "opponent_deep",
        "team_ppda_att",
        "team_ppda_def",
        "team_win_odds_open",
        "draw_odds_open",
        "opponent_win_odds_open",
    ]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw["season_key"] = raw["match_id"].str.rsplit("_", n=3).str[0]
    raw["league"] = raw["season_key"].str.rsplit(" ", n=1).str[0]
    raw["season"] = raw["season_key"].str.rsplit(" ", n=1).str[1].astype(int)

    raw["team_id"] = raw["team_id"].astype(str)
    raw["opponent_id"] = raw["opponent_id"].astype(str)

    return raw


def _rolling_mean_prematch(series: pd.Series, window: int) -> pd.Series:
    if window == 1:
        return series.shift(1)
    return series.shift(1).rolling(window, min_periods=1).mean()


def _blend_mean_with_prior(
    current_mean: pd.Series,
    prior_mean: pd.Series,
    matches_before: pd.Series,
    window: int,
) -> pd.Series:
    current_count = np.minimum(matches_before.fillna(0.0).astype(float), float(window))
    prior_count = float(window) - current_count

    blended = current_mean.copy()
    has_prior = prior_mean.notna()
    blended.loc[has_prior] = (
        current_mean.loc[has_prior].fillna(0.0) * current_count.loc[has_prior]
        + prior_mean.loc[has_prior] * prior_count.loc[has_prior]
    ) / float(window)
    return blended


def _blend_sum_with_prior(
    current_sum: pd.Series,
    prior_points_per_game: pd.Series,
    matches_before: pd.Series,
    window: int,
) -> pd.Series:
    current_count = np.minimum(matches_before.fillna(0.0).astype(float), float(window))
    prior_count = float(window) - current_count

    blended = current_sum.copy()
    has_prior = prior_points_per_game.notna()
    blended.loc[has_prior] = (
        current_sum.loc[has_prior].fillna(0.0)
        + prior_points_per_game.loc[has_prior] * prior_count.loc[has_prior]
    )
    return blended


def add_team_prematch_features(team_rows: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """Adds prematch features with previous-season carry-over."""

    team_rows = team_rows.sort_values(["league", "season", "team_id", "date"]).copy()
    team_rows["_points"] = team_rows["result"].map({"w": 3, "d": 1, "l": 0}).astype(float)

    def _per_team_season(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        matches_before = pd.Series(np.arange(len(group)), index=group.index).astype(float)
        group["matches_played_in_season_before"] = matches_before

        for w in windows:
            group[f"team_xG_last_{w}"] = _rolling_mean_prematch(group["team_xG"], w)
            group[f"team_deep_last_{w}"] = _rolling_mean_prematch(group["team_deep"], w)
            group[f"team_xG_against_last_{w}"] = _rolling_mean_prematch(group["opponent_xG"], w)
            group[f"team_deep_against_last_{w}"] = _rolling_mean_prematch(group["opponent_deep"], w)
            group[f"team_ppda_att_last_{w}"] = _rolling_mean_prematch(group["team_ppda_att"], w)
            group[f"team_ppda_def_last_{w}"] = _rolling_mean_prematch(group["team_ppda_def"], w)

        group["form_score_5"] = group["_points"].shift(1).rolling(5, min_periods=1).sum()
        group["form_score_10"] = group["_points"].shift(1).rolling(10, min_periods=1).sum()
        group["form_momentum"] = (group["form_score_5"] / 5.0) - (group["form_score_10"] / 10.0)

        group["xG_efficiency_5"] = (
            group["_points"].shift(1).rolling(5, min_periods=1).sum()
            / (group["team_xG"].shift(1).rolling(5, min_periods=1).sum() + 1e-9)
        )
        group["recent_xG_trend"] = (
            group["team_xG"].shift(1).rolling(3, min_periods=1).mean()
            - group["team_xG"].shift(1).rolling(8, min_periods=1).mean()
        )
        group["defensive_trend"] = (
            group["opponent_xG"].shift(1).rolling(3, min_periods=1).mean()
            - group["opponent_xG"].shift(1).rolling(8, min_periods=1).mean()
        )

        group["team_rest_days"] = (group["date"] - group["date"].shift(1)).dt.total_seconds() / 86400.0

        denom = matches_before.replace(0.0, np.nan)
        group["team_season_points_per_game"] = group["_points"].shift(1).cumsum() / denom

        group["team_season_avg_xG"] = group["team_xG"].shift(1).expanding(min_periods=1).mean()
        group["team_season_avg_xG_against"] = group["opponent_xG"].shift(1).expanding(min_periods=1).mean()
        group["team_season_avg_deep"] = group["team_deep"].shift(1).expanding(min_periods=1).mean()
        group["team_season_avg_deep_against"] = group["opponent_deep"].shift(1).expanding(min_periods=1).mean()
        group["team_season_avg_ppda_att"] = group["team_ppda_att"].shift(1).expanding(min_periods=1).mean()
        group["team_season_avg_ppda_def"] = group["team_ppda_def"].shift(1).expanding(min_periods=1).mean()

        current_streak = []
        unbeaten_streak = []
        winless_streak = []

        win_streak = 0
        loss_streak = 0
        unbeaten = 0
        winless = 0

        for res in group["result"].tolist():
            if win_streak > 0:
                current_streak.append(win_streak)
            elif loss_streak > 0:
                current_streak.append(-loss_streak)
            else:
                current_streak.append(0)

            unbeaten_streak.append(unbeaten)
            winless_streak.append(winless)

            if res == "w":
                win_streak += 1
                loss_streak = 0
            elif res == "l":
                loss_streak += 1
                win_streak = 0
            else:
                win_streak = 0
                loss_streak = 0

            if res in ("w", "d"):
                unbeaten += 1
            else:
                unbeaten = 0

            if res in ("d", "l"):
                winless += 1
            else:
                winless = 0

        group["current_streak"] = current_streak
        group["unbeaten_streak"] = unbeaten_streak
        group["winless_streak"] = winless_streak

        return group

    team_rows = team_rows.groupby(["league", "season", "team_id"], group_keys=False).apply(_per_team_season)

    season_summary = (
        team_rows.groupby(["league", "team_id", "season"], as_index=False)
        .agg(
            prev_season_points_per_game=("_points", "mean"),
            prev_season_avg_xG=("team_xG", "mean"),
            prev_season_avg_xG_against=("opponent_xG", "mean"),
            prev_season_avg_deep=("team_deep", "mean"),
            prev_season_avg_deep_against=("opponent_deep", "mean"),
            prev_season_avg_ppda_att=("team_ppda_att", "mean"),
            prev_season_avg_ppda_def=("team_ppda_def", "mean"),
            prev_season_match_count=("match_id", "size"),
        )
        .copy()
    )
    season_summary["season"] = season_summary["season"] + 1

    team_rows = team_rows.merge(
        season_summary,
        on=["league", "team_id", "season"],
        how="left",
        validate="many_to_one",
    )

    matches_before = team_rows["matches_played_in_season_before"]
    for w in windows:
        team_rows[f"team_xG_last_{w}_carry"] = _blend_mean_with_prior(
            team_rows[f"team_xG_last_{w}"],
            team_rows["prev_season_avg_xG"],
            matches_before,
            w,
        )
        team_rows[f"team_deep_last_{w}_carry"] = _blend_mean_with_prior(
            team_rows[f"team_deep_last_{w}"],
            team_rows["prev_season_avg_deep"],
            matches_before,
            w,
        )
        team_rows[f"team_xG_against_last_{w}_carry"] = _blend_mean_with_prior(
            team_rows[f"team_xG_against_last_{w}"],
            team_rows["prev_season_avg_xG_against"],
            matches_before,
            w,
        )
        team_rows[f"team_deep_against_last_{w}_carry"] = _blend_mean_with_prior(
            team_rows[f"team_deep_against_last_{w}"],
            team_rows["prev_season_avg_deep_against"],
            matches_before,
            w,
        )
        team_rows[f"team_ppda_att_last_{w}_carry"] = _blend_mean_with_prior(
            team_rows[f"team_ppda_att_last_{w}"],
            team_rows["prev_season_avg_ppda_att"],
            matches_before,
            w,
        )
        team_rows[f"team_ppda_def_last_{w}_carry"] = _blend_mean_with_prior(
            team_rows[f"team_ppda_def_last_{w}"],
            team_rows["prev_season_avg_ppda_def"],
            matches_before,
            w,
        )

    team_rows["form_score_5_carry"] = _blend_sum_with_prior(
        team_rows["form_score_5"],
        team_rows["prev_season_points_per_game"],
        matches_before,
        5,
    )
    team_rows["form_score_10_carry"] = _blend_sum_with_prior(
        team_rows["form_score_10"],
        team_rows["prev_season_points_per_game"],
        matches_before,
        10,
    )

    team_rows["team_season_points_per_game_carry"] = team_rows["team_season_points_per_game"].fillna(
        team_rows["prev_season_points_per_game"]
    )
    team_rows["team_season_avg_xG_carry"] = team_rows["team_season_avg_xG"].fillna(team_rows["prev_season_avg_xG"])
    team_rows["team_season_avg_xG_against_carry"] = team_rows["team_season_avg_xG_against"].fillna(
        team_rows["prev_season_avg_xG_against"]
    )
    team_rows["team_season_avg_deep_carry"] = team_rows["team_season_avg_deep"].fillna(
        team_rows["prev_season_avg_deep"]
    )
    team_rows["team_season_avg_deep_against_carry"] = team_rows["team_season_avg_deep_against"].fillna(
        team_rows["prev_season_avg_deep_against"]
    )
    team_rows["team_season_avg_ppda_att_carry"] = team_rows["team_season_avg_ppda_att"].fillna(
        team_rows["prev_season_avg_ppda_att"]
    )
    team_rows["team_season_avg_ppda_def_carry"] = team_rows["team_season_avg_ppda_def"].fillna(
        team_rows["prev_season_avg_ppda_def"]
    )

    return team_rows


@dataclass
class EloConfig:
    initial: float = 1500.0
    k: float = 20.0
    home_adv: float = 60.0
    season_carry: float = 0.75


def compute_match_elo(matches: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    """Computes prematch Elo per league with cross-season carry-over."""

    matches = matches.sort_values(["league", "date", "match_id"]).copy()

    team_elo_home = []
    team_elo_away = []
    p_home = []

    for _, group in matches.groupby(["league"], sort=False):
        ratings: dict[str, float] = {}
        current_season: int | None = None

        for row in group.itertuples(index=False):
            if current_season is None:
                current_season = int(row.season)
            elif int(row.season) != current_season:
                ratings = {
                    team_id: config.initial + (rating - config.initial) * config.season_carry
                    for team_id, rating in ratings.items()
                }
                current_season = int(row.season)

            h = str(row.home_team_id)
            a = str(row.away_team_id)

            h_elo = ratings.get(h, config.initial)
            a_elo = ratings.get(a, config.initial)

            team_elo_home.append(h_elo)
            team_elo_away.append(a_elo)

            exp_home = 1.0 / (1.0 + 10 ** ((a_elo - (h_elo + config.home_adv)) / 400.0))
            p_home.append(exp_home)

            if row.result == "w":
                score_home = 1.0
            elif row.result == "d":
                score_home = 0.5
            else:
                score_home = 0.0

            ratings[h] = h_elo + config.k * (score_home - exp_home)
            ratings[a] = a_elo + config.k * ((1.0 - score_home) - (1.0 - exp_home))

    matches["team_elo_rating"] = team_elo_home
    matches["opponent_elo_rating"] = team_elo_away
    matches["elo_win_probability"] = p_home
    matches["elo_rating_gap"] = matches["team_elo_rating"] - matches["opponent_elo_rating"]

    return matches[
        [
            "match_id",
            "team_elo_rating",
            "opponent_elo_rating",
            "elo_rating_gap",
            "elo_win_probability",
        ]
    ]


def add_market_implied_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()

    inv_home = 1.0 / matches["market_home_win_odds_open"]
    inv_draw = 1.0 / matches["market_draw_odds_open"]
    inv_away = 1.0 / matches["market_away_win_odds_open"]

    overround = inv_home + inv_draw + inv_away
    matches["market_overround_open"] = overround
    matches["market_home_prob_open"] = inv_home / overround
    matches["market_draw_prob_open"] = inv_draw / overround
    matches["market_away_prob_open"] = inv_away / overround
    matches["market_home_minus_away_prob_open"] = (
        matches["market_home_prob_open"] - matches["market_away_prob_open"]
    )
    matches["market_non_draw_prob_open"] = 1.0 - matches["market_draw_prob_open"]

    probs = matches[
        [
            "market_home_prob_open",
            "market_draw_prob_open",
            "market_away_prob_open",
        ]
    ]
    missing_market = probs.isna().any(axis=1)
    matches["market_favorite_prob_open"] = probs.max(axis=1)

    probs_array = np.nan_to_num(probs.to_numpy(), nan=-1.0)
    second_highest = np.sort(probs_array, axis=1)[:, -2]
    second_highest[missing_market.to_numpy()] = np.nan
    matches["market_favorite_gap_open"] = matches["market_favorite_prob_open"] - second_highest

    entropy_input = probs.clip(lower=1e-12)
    matches["market_entropy_open"] = -(entropy_input * np.log(entropy_input)).sum(axis=1)
    matches.loc[missing_market, "market_entropy_open"] = np.nan
    return matches


def build_home_perspective_dataset(team_rows: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    home = team_rows[team_rows["is_home"] == True].copy()
    away = team_rows[team_rows["is_home"] == False].copy()

    team_feature_cols = []
    for w in windows:
        team_feature_cols += [
            f"team_xG_last_{w}",
            f"team_deep_last_{w}",
            f"team_xG_against_last_{w}",
            f"team_deep_against_last_{w}",
            f"team_ppda_att_last_{w}",
            f"team_ppda_def_last_{w}",
            f"team_xG_last_{w}_carry",
            f"team_deep_last_{w}_carry",
            f"team_xG_against_last_{w}_carry",
            f"team_deep_against_last_{w}_carry",
            f"team_ppda_att_last_{w}_carry",
            f"team_ppda_def_last_{w}_carry",
        ]

    team_feature_cols += [
        "matches_played_in_season_before",
        "form_score_5",
        "form_score_10",
        "form_score_5_carry",
        "form_score_10_carry",
        "form_momentum",
        "current_streak",
        "unbeaten_streak",
        "winless_streak",
        "prev_season_points_per_game",
        "prev_season_avg_xG",
        "prev_season_avg_xG_against",
        "prev_season_avg_deep",
        "prev_season_avg_deep_against",
        "prev_season_avg_ppda_att",
        "prev_season_avg_ppda_def",
        "prev_season_match_count",
        "team_season_points_per_game",
        "team_season_points_per_game_carry",
        "team_season_avg_xG",
        "team_season_avg_xG_against",
        "team_season_avg_deep",
        "team_season_avg_deep_against",
        "team_season_avg_ppda_att",
        "team_season_avg_ppda_def",
        "team_season_avg_xG_carry",
        "team_season_avg_xG_against_carry",
        "team_season_avg_deep_carry",
        "team_season_avg_deep_against_carry",
        "team_season_avg_ppda_att_carry",
        "team_season_avg_ppda_def_carry",
        "team_rest_days",
        "xG_efficiency_5",
        "recent_xG_trend",
        "defensive_trend",
    ]

    base_cols = [
        "match_id",
        "date",
        "league",
        "season",
        "team_id",
        "team_name",
        "opponent_id",
        "opponent_name",
        "result",
    ]
    market_cols = [
        "team_win_odds_open",
        "draw_odds_open",
        "opponent_win_odds_open",
    ]

    home_df = home[base_cols + team_feature_cols + market_cols].copy()
    home_df = home_df.rename(
        columns={
            "team_win_odds_open": "market_home_win_odds_open",
            "draw_odds_open": "market_draw_odds_open",
            "opponent_win_odds_open": "market_away_win_odds_open",
        }
    )

    away_df = away[["match_id", "team_id", "team_name", "opponent_id"] + team_feature_cols].copy()

    rename_map = {
        "team_id": "away_team_id",
        "team_name": "away_team_name",
        "opponent_id": "away_opponent_id",
    }
    for c in team_feature_cols:
        if c.startswith("team_"):
            rename_map[c] = "opponent_" + c[len("team_") :]
        else:
            rename_map[c] = "opponent_" + c

    away_df = away_df.rename(columns=rename_map)

    matches = home_df.merge(away_df, on="match_id", how="inner", validate="one_to_one")

    bad = matches[
        (matches["opponent_id"] != matches["away_team_id"])
        | (matches["away_opponent_id"] != matches["team_id"])
    ]
    if len(bad) > 0:
        raise ValueError(f"Join mismatch: {len(bad)} rows have inconsistent opponents")

    matches["rest_days_diff"] = matches["team_rest_days"] - matches["opponent_rest_days"]
    matches["rest_days_ratio"] = matches["team_rest_days"] / matches["opponent_rest_days"]

    matches["relative_form_5"] = matches["form_score_5"] - matches["opponent_form_score_5"]
    matches["relative_form_10"] = matches["form_score_10"] - matches["opponent_form_score_10"]
    matches["relative_form_5_carry"] = matches["form_score_5_carry"] - matches["opponent_form_score_5_carry"]
    matches["relative_form_10_carry"] = matches["form_score_10_carry"] - matches["opponent_form_score_10_carry"]

    matches["xG_efficiency_gap_5"] = matches["xG_efficiency_5"] - matches["opponent_xG_efficiency_5"]
    matches["xG_trend_gap"] = matches["recent_xG_trend"] - matches["opponent_recent_xG_trend"]
    matches["defensive_trend_gap"] = matches["defensive_trend"] - matches["opponent_defensive_trend"]

    matches["prev_season_points_per_game_gap"] = (
        matches["prev_season_points_per_game"] - matches["opponent_prev_season_points_per_game"]
    )
    matches["prev_season_xG_gap"] = matches["prev_season_avg_xG"] - matches["opponent_prev_season_avg_xG"]
    matches["prev_season_defensive_gap"] = (
        matches["opponent_prev_season_avg_xG_against"] - matches["prev_season_avg_xG_against"]
    )
    matches["season_points_per_game_gap"] = (
        matches["team_season_points_per_game_carry"] - matches["opponent_season_points_per_game_carry"]
    )

    for w in windows:
        matches[f"xG_advantage_{w}"] = matches[f"team_xG_last_{w}"] - matches[f"opponent_xG_last_{w}"]
        matches[f"defensive_advantage_{w}"] = (
            matches[f"opponent_xG_against_last_{w}"] - matches[f"team_xG_against_last_{w}"]
        )
        matches[f"deep_advantage_{w}"] = matches[f"team_deep_last_{w}"] - matches[f"opponent_deep_last_{w}"]
        matches[f"ppda_advantage_{w}"] = (
            matches[f"opponent_ppda_att_last_{w}"] - matches[f"team_ppda_att_last_{w}"]
        )

        matches[f"xG_advantage_{w}_carry"] = (
            matches[f"team_xG_last_{w}_carry"] - matches[f"opponent_xG_last_{w}_carry"]
        )
        matches[f"defensive_advantage_{w}_carry"] = (
            matches[f"opponent_xG_against_last_{w}_carry"] - matches[f"team_xG_against_last_{w}_carry"]
        )
        matches[f"deep_advantage_{w}_carry"] = (
            matches[f"team_deep_last_{w}_carry"] - matches[f"opponent_deep_last_{w}_carry"]
        )
        matches[f"ppda_advantage_{w}_carry"] = (
            matches[f"opponent_ppda_att_last_{w}_carry"] - matches[f"team_ppda_att_last_{w}_carry"]
        )

    matches = add_market_implied_features(matches)
    matches["target"] = matches["result"].map({"l": 0, "d": 1, "w": 2}).astype(int)
    matches = matches.drop(columns=["away_opponent_id"])

    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--elo-k", type=float, default=20.0)
    parser.add_argument("--elo-home-adv", type=float, default=60.0)
    parser.add_argument("--elo-season-carry", type=float, default=0.75)
    parser.add_argument(
        "--windows",
        default=",".join(str(w) for w in WINDOWS_DEFAULT),
        help="Comma-separated rolling windows (e.g. 1,3,5)",
    )
    args = parser.parse_args()

    windows = tuple(int(x) for x in args.windows.split(",") if x.strip())
    if not windows:
        raise ValueError("--windows must contain at least one integer")

    team_rows = load_team_match_rows(args.data_dir)
    team_rows = add_team_prematch_features(team_rows, windows=windows)

    dataset = build_home_perspective_dataset(team_rows, windows=windows)

    elo_input = dataset[["match_id", "league", "season", "date", "team_id", "opponent_id", "result"]].rename(
        columns={"team_id": "home_team_id", "opponent_id": "away_team_id"}
    )
    elo = compute_match_elo(
        elo_input,
        EloConfig(
            k=args.elo_k,
            home_adv=args.elo_home_adv,
            season_carry=args.elo_season_carry,
        ),
    )
    dataset = dataset.merge(elo, on="match_id", how="left", validate="one_to_one")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Wrote {len(dataset)} rows to {output_path}")


if __name__ == "__main__":
    main()
