import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_common import build_xgb_model, get_feature_cols, load_dataset, make_sample_weight


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "dataset_home.csv"


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path.cwd() / candidate).resolve())


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--val-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-league", default="")
    parser.add_argument("--bet-league", default="")
    parser.add_argument("--outcome", default="draw", choices=["home_win", "draw", "away_win"])
    parser.add_argument("--odds-min", type=float, default=2.0)
    parser.add_argument("--odds-max", type=float, default=10.0)
    parser.add_argument("--market-favorite-mode", default="nonfavorite", choices=["all", "favorite", "nonfavorite"])
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--edge-values", default="0.0,0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--min-val-bets", type=int, default=60)
    parser.add_argument("--export-bets", default="")
    return parser.parse_args()


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
    outcome_map = np.array(["away_win", "draw", "home_win"], dtype=object)

    bet_df = eval_df.copy()
    bet_df["selected_outcome"] = outcome_map[chosen]
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
    selected_idx = pd.Series(bet_df["selected_outcome"]).map({"away_win": 0, "draw": 1, "home_win": 2}).to_numpy()
    bet_df["bet_is_market_favorite"] = selected_idx == market_fav_idx
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


def main() -> None:
    args = parse_args()

    df = load_dataset(resolve_path(args.data)).dropna(subset=["target"]).copy()
    if args.train_league:
        df = df[df["league"] == args.train_league].copy()

    feature_cols = get_feature_cols(df)
    train_df = df[df["season"] < args.val_season].copy()
    val_df = df[df["season"] == args.val_season].copy()
    pretest_df = df[df["season"] <= args.val_season].copy()
    test_df = df[df["season"] == args.test_season].copy()

    if train_df.empty or val_df.empty or pretest_df.empty or test_df.empty:
        raise ValueError("One of train/val/pretest/test splits is empty")

    print(
        f"rows: train={len(train_df)} val={len(val_df)} pretest={len(pretest_df)} test={len(test_df)} "
        f"train_league={args.train_league or 'ALL'} bet_league={args.bet_league or 'ALL'}"
    )
    print(
        f"strategy: outcome={args.outcome} odds=[{args.odds_min}, {args.odds_max}) "
        f"market_favorite_mode={args.market_favorite_mode}"
    )

    rng = np.random.default_rng(args.seed)
    thresholds = threshold_values(args.threshold_start, args.threshold_stop, args.threshold_step)
    edge_values = [float(value) for value in args.edge_values.split(",") if value.strip()]

    best: dict | None = None

    for trial_idx in range(args.trials):
        params = sample_params(rng)
        model = build_xgb_model(seed=args.seed, n_estimators=500, **params)
        model.fit(
            train_df[feature_cols],
            train_df["target"].astype(int),
            sample_weight=make_sample_weight(train_df["target"].astype(int)),
        )
        val_base = build_base_bets(val_df, model.predict_proba(val_df[feature_cols]))

        trial_best: dict | None = None
        for threshold in thresholds:
            for edge_min in edge_values:
                val_bets = apply_strategy(
                    val_base,
                    threshold=threshold,
                    edge_min=edge_min,
                    bet_league=args.bet_league,
                    outcome=args.outcome,
                    odds_min=args.odds_min,
                    odds_max=args.odds_max,
                    market_favorite_mode=args.market_favorite_mode,
                )
                if len(val_bets) < args.min_val_bets:
                    continue

                candidate = {
                    "params": params,
                    "threshold": threshold,
                    "edge_min": edge_min,
                    "val_bets": int(len(val_bets)),
                    "val_roi": float(val_bets["profit"].mean()),
                    "val_profit": float(val_bets["profit"].sum()),
                }
                if trial_best is None or (
                    candidate["val_roi"],
                    candidate["val_profit"],
                    candidate["val_bets"],
                ) > (
                    trial_best["val_roi"],
                    trial_best["val_profit"],
                    trial_best["val_bets"],
                ):
                    trial_best = candidate

        if trial_best is None:
            continue

        if best is None or (
            trial_best["val_roi"],
            trial_best["val_profit"],
            trial_best["val_bets"],
        ) > (
            best["val_roi"],
            best["val_profit"],
            best["val_bets"],
        ):
            best = trial_best

        print(
            f"trial {trial_idx+1:03d}/{args.trials} "
            f"val_roi={trial_best['val_roi']:.4f} val_profit={trial_best['val_profit']:.2f} "
            f"val_bets={trial_best['val_bets']} threshold={trial_best['threshold']:.2f} "
            f"edge_min={trial_best['edge_min']:.2f}"
        )

    if best is None:
        raise ValueError("No strategy satisfied min_val_bets on validation")

    print("\nBEST ON VALIDATION")
    print(best)

    final_model = build_xgb_model(seed=args.seed, n_estimators=500, **best["params"])
    final_model.fit(
        pretest_df[feature_cols],
        pretest_df["target"].astype(int),
        sample_weight=make_sample_weight(pretest_df["target"].astype(int)),
    )
    test_base = build_base_bets(test_df, final_model.predict_proba(test_df[feature_cols]))
    test_bets = apply_strategy(
        test_base,
        threshold=best["threshold"],
        edge_min=best["edge_min"],
        bet_league=args.bet_league,
        outcome=args.outcome,
        odds_min=args.odds_min,
        odds_max=args.odds_max,
        market_favorite_mode=args.market_favorite_mode,
    )

    test_roi = float(test_bets["profit"].mean()) if len(test_bets) else float("nan")
    test_profit = float(test_bets["profit"].sum()) if len(test_bets) else 0.0

    print("\nTEST")
    print(
        {
            "test_bets": int(len(test_bets)),
            "test_roi": None if np.isnan(test_roi) else round(test_roi, 4),
            "test_profit": round(test_profit, 4),
            "hit_rate": None if len(test_bets) == 0 else round(float(test_bets["won_bet"].mean()), 4),
        }
    )

    if args.export_bets:
        output_path = Path(resolve_path(args.export_bets))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        test_bets.to_csv(output_path, index=False)
        print(f"exported bets {len(test_bets)} -> {output_path}")


if __name__ == "__main__":
    main()
