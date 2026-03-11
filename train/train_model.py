import argparse
import os
from pathlib import Path

import numpy as np

from ml_common import (
    evaluate_value_bets,
    fit_and_predict,
    get_feature_cols,
    load_dataset,
    print_full_report,
    time_split_grouped,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "dataset_home.csv"


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path.cwd() / candidate).resolve())


def build_model_df(data_path: str):
    df = load_dataset(data_path)
    feature_cols = get_feature_cols(df)
    model_df = df.dropna(subset=["target"]).copy()
    return model_df, feature_cols


def split_for_holdout_season(model_df, holdout_season: int):
    train_df = model_df[model_df["season"] < holdout_season].copy()
    test_df = model_df[model_df["season"] == holdout_season].copy()
    if train_df.empty:
        raise ValueError(f"No training rows found before season {holdout_season}")
    if test_df.empty:
        raise ValueError(f"No test rows found for season {holdout_season}")
    return train_df, test_df


def run_model(model_df, feature_cols: list[str], test_frac: float, seed: int, holdout_season: int | None):
    if holdout_season is None:
        train_df, test_df = time_split_grouped(model_df, test_frac=test_frac)
    else:
        train_df, test_df = split_for_holdout_season(model_df, holdout_season=holdout_season)
    return fit_and_predict(train_df, test_df, feature_cols=feature_cols, seed=seed)


def export_selected_bets(
    run,
    ev_threshold: float,
    output_path: str,
) -> int:
    output_dir = os.path.dirname(output_path)
    odds_cols = [
        "market_away_win_odds_open",
        "market_draw_odds_open",
        "market_home_win_odds_open",
    ]
    odds = run.test_df[odds_cols].to_numpy()
    expected_value = run.proba * odds - 1.0
    chosen = expected_value.argmax(axis=1)
    chosen_ev = expected_value[np.arange(len(expected_value)), chosen]
    valid_mask = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    bet_mask = valid_mask & (chosen_ev > ev_threshold)

    if not bet_mask.any():
        output = run.test_df.iloc[0:0].copy()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output.to_csv(output_path, index=False)
        return 0

    outcome_map = np.array(["away_win", "draw", "home_win"], dtype=object)
    bet_df = run.test_df.loc[bet_mask].copy()
    bet_df["selected_outcome"] = outcome_map[chosen[bet_mask]]
    bet_df["selected_odds"] = odds[np.arange(len(odds)), chosen][bet_mask]
    bet_df["predicted_probability"] = run.proba[np.arange(len(run.proba)), chosen][bet_mask]
    bet_df["expected_value"] = chosen_ev[bet_mask]
    bet_df["won_bet"] = (
        chosen[bet_mask] == bet_df["target"].astype(int).to_numpy()
    )
    bet_df["profit"] = np.where(
        bet_df["won_bet"],
        bet_df["selected_odds"] - 1.0,
        -1.0,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    bet_df.to_csv(output_path, index=False)
    return len(bet_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout-season",
        type=int,
        default=None,
        help="If set, trains on seasons before this value and evaluates only on this season",
    )
    parser.add_argument(
        "--bet-ev-threshold",
        type=float,
        default=0.5,
        help="Only place a flat 1-unit bet when predicted EV exceeds this threshold",
    )
    parser.add_argument(
        "--export-bets",
        default="",
        help="Optional CSV export for the selected bets on the evaluation split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    export_path = resolve_path(args.export_bets) if args.export_bets else ""
    model_df, feature_cols = build_model_df(data_path)

    print(f"feature count={len(feature_cols)} rows after dropna={len(model_df)}")
    run = run_model(
        model_df,
        feature_cols,
        test_frac=args.test_frac,
        seed=args.seed,
        holdout_season=args.holdout_season,
    )

    print_full_report("global", run)

    betting = evaluate_value_bets(run.test_df, run.proba, ev_threshold=args.bet_ev_threshold)
    print(
        "value betting",
        {
            "bets": int(betting["bets"]),
            "hit_rate": None if np.isnan(betting["hit_rate"]) else round(betting["hit_rate"], 4),
            "roi": None if np.isnan(betting["roi"]) else round(betting["roi"], 4),
            "avg_ev": None if np.isnan(betting["avg_ev"]) else round(betting["avg_ev"], 4),
            "avg_edge": None if np.isnan(betting["avg_edge"]) else round(betting["avg_edge"], 4),
        },
    )

    if export_path:
        count = export_selected_bets(run, ev_threshold=args.bet_ev_threshold, output_path=export_path)
        print(f"exported bets {count} -> {export_path}")


if __name__ == "__main__":
    main()
