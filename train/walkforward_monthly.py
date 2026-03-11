import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_common import build_xgb_model, evaluate_value_bets, get_feature_cols, load_dataset, make_sample_weight
from seasonal_protocol import (
    TrialResult,
    accuracy,
    choose_threshold,
    multiclass_logloss,
    resolve_path,
    sample_params,
    split_by_season,
    threshold_candidates,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "dataset_home.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--val-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--threshold-start", type=float, default=0.0)
    parser.add_argument("--threshold-stop", type=float, default=0.8)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--min-val-bets", type=int, default=75)
    parser.add_argument(
        "--selection-metric",
        default="logloss",
        choices=["logloss", "roi"],
        help="How to choose hyperparameters on the validation season. Threshold is always chosen on validation ROI.",
    )
    parser.add_argument("--export-bets", default="")
    parser.add_argument("--export-summary", default="")
    return parser.parse_args()


def select_best_setup(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> TrialResult:
    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int).to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["target"].astype(int).to_numpy()
    sample_weight = make_sample_weight(train_df["target"].astype(int))
    rng = np.random.default_rng(args.seed)
    thresholds = threshold_candidates(args.threshold_start, args.threshold_stop, args.threshold_step)

    best: TrialResult | None = None

    for trial_idx in range(args.trials):
        params = sample_params(rng)
        model = build_xgb_model(
            seed=args.seed,
            n_estimators=5000,
            early_stopping_rounds=args.early_stopping_rounds,
            **params,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        proba_val = model.predict_proba(X_val)
        pred_val = proba_val.argmax(axis=1)
        threshold, val_betting = choose_threshold(
            val_df,
            proba_val,
            candidates=thresholds,
            min_val_bets=args.min_val_bets,
        )
        result = TrialResult(
            params=params,
            best_iteration=int(getattr(model, "best_iteration", model.n_estimators - 1)),
            val_acc=accuracy(y_val, pred_val),
            val_logloss=multiclass_logloss(y_val, proba_val),
            val_roi=float(val_betting["roi"]),
            val_bets=int(val_betting["bets"]),
            threshold=threshold,
        )

        is_better = False
        if best is None:
            is_better = True
        elif args.selection_metric == "logloss":
            is_better = (
                result.val_logloss,
                -result.val_roi,
                -result.val_bets,
            ) < (
                best.val_logloss,
                -best.val_roi,
                -best.val_bets,
            )
        else:
            is_better = (
                result.val_roi,
                -result.val_logloss,
                result.val_bets,
            ) > (
                best.val_roi,
                -best.val_logloss,
                best.val_bets,
            )

        if is_better:
            best = result

        if (trial_idx + 1) % args.print_every == 0 or trial_idx == 0 or trial_idx == args.trials - 1:
            print(
                f"trial {trial_idx+1:03d}/{args.trials} "
                f"val_roi={result.val_roi:.4f} val_bets={result.val_bets} "
                f"val_acc={result.val_acc:.4f} val_logloss={result.val_logloss:.4f} "
                f"threshold={result.threshold:.2f} best_iter={result.best_iteration}"
            )
            print(f"  params: {params}")

    assert best is not None
    return best


def export_bets(all_bets: pd.DataFrame, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_bets.to_csv(output, index=False)


def build_bets_df(eval_df: pd.DataFrame, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    odds_cols = [
        "market_away_win_odds_open",
        "market_draw_odds_open",
        "market_home_win_odds_open",
    ]
    odds = eval_df[odds_cols].to_numpy()
    expected_value = proba * odds - 1.0
    chosen = expected_value.argmax(axis=1)
    chosen_ev = expected_value[np.arange(len(expected_value)), chosen]
    valid_mask = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    bet_mask = valid_mask & (chosen_ev > threshold)
    if not bet_mask.any():
        return eval_df.iloc[0:0].copy()

    outcome_map = np.array(["away_win", "draw", "home_win"], dtype=object)
    bet_df = eval_df.loc[bet_mask].copy()
    bet_df["selected_outcome"] = outcome_map[chosen[bet_mask]]
    bet_df["selected_odds"] = odds[np.arange(len(odds)), chosen][bet_mask]
    bet_df["predicted_probability"] = proba[np.arange(len(proba)), chosen][bet_mask]
    bet_df["expected_value"] = chosen_ev[bet_mask]
    bet_df["won_bet"] = chosen[bet_mask] == bet_df["target"].astype(int).to_numpy()
    bet_df["profit"] = np.where(bet_df["won_bet"], bet_df["selected_odds"] - 1.0, -1.0)
    return bet_df


def main() -> None:
    args = parse_args()

    df = load_dataset(resolve_path(args.data))
    feature_cols = get_feature_cols(df)
    model_df = df.dropna(subset=["target"]).copy()
    train_df, val_df, test_df = split_by_season(model_df, args.val_season, args.test_season)

    print(
        f"rows: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"val_season={args.val_season} test_season={args.test_season}"
    )
    print(
        f"dates: train_max={train_df['date'].max()} "
        f"val_range=({val_df['date'].min()} -> {val_df['date'].max()}) "
        f"test_range=({test_df['date'].min()} -> {test_df['date'].max()})"
    )
    print(f"feature count={len(feature_cols)}")

    best = select_best_setup(train_df, val_df, feature_cols, args)
    print("\nBEST ON VALIDATION")
    print(
        f"selection_metric={args.selection_metric} "
        f"val_roi={best.val_roi:.4f} val_bets={best.val_bets} "
        f"val_acc={best.val_acc:.4f} val_logloss={best.val_logloss:.4f} "
        f"threshold={best.threshold:.2f} best_iter={best.best_iteration}"
    )
    print(best.params)

    test_df = test_df.copy()
    test_df["month_bucket"] = test_df["date"].dt.to_period("M").astype(str)
    month_buckets = list(test_df["month_bucket"].drop_duplicates())

    monthly_rows: list[dict] = []
    monthly_bets: list[pd.DataFrame] = []
    monthly_eval_frames: list[pd.DataFrame] = []
    monthly_predictions: list[np.ndarray] = []
    monthly_targets: list[np.ndarray] = []

    fixed_estimators = max(best.best_iteration + 1, 50)

    for month_bucket in month_buckets:
        month_df = test_df[test_df["month_bucket"] == month_bucket].copy()
        month_start = month_df["date"].min()
        month_train_df = model_df[model_df["date"] < month_start].copy()

        X_month_train = month_train_df[feature_cols]
        y_month_train = month_train_df["target"].astype(int).to_numpy()
        X_month_test = month_df[feature_cols]
        y_month_test = month_df["target"].astype(int).to_numpy()

        model = build_xgb_model(
            seed=args.seed,
            n_estimators=fixed_estimators,
            **best.params,
        )
        model.fit(
            X_month_train,
            y_month_train,
            sample_weight=make_sample_weight(month_train_df["target"].astype(int)),
        )

        proba = model.predict_proba(X_month_test)
        pred = proba.argmax(axis=1)
        betting = evaluate_value_bets(month_df, proba, ev_threshold=best.threshold)
        bets_df = build_bets_df(month_df, proba, threshold=best.threshold)
        if not bets_df.empty:
            bets_df["month_bucket"] = month_bucket
            monthly_bets.append(bets_df)

        monthly_rows.append(
            {
                "month_bucket": month_bucket,
                "matches": len(month_df),
                "train_rows": len(month_train_df),
                "acc": accuracy(y_month_test, pred),
                "logloss": multiclass_logloss(y_month_test, proba),
                "bets": int(betting["bets"]),
                "hit_rate": None if np.isnan(betting["hit_rate"]) else float(betting["hit_rate"]),
                "roi": None if np.isnan(betting["roi"]) else float(betting["roi"]),
                "avg_ev": None if np.isnan(betting["avg_ev"]) else float(betting["avg_ev"]),
                "avg_edge": None if np.isnan(betting["avg_edge"]) else float(betting["avg_edge"]),
            }
        )
        monthly_eval_frames.append(month_df)
        monthly_predictions.append(proba)
        monthly_targets.append(y_month_test)

    monthly_summary = pd.DataFrame(monthly_rows)
    ordered_test_df = pd.concat(monthly_eval_frames, ignore_index=True)
    all_test_proba = np.vstack(monthly_predictions)
    all_test_targets = np.concatenate(monthly_targets)
    overall_betting = evaluate_value_bets(ordered_test_df, all_test_proba, ev_threshold=best.threshold)

    print("\nMONTHLY TEST")
    print(monthly_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nOVERALL TEST")
    print(
        {
            "acc": round(accuracy(all_test_targets, all_test_proba.argmax(axis=1)), 4),
            "logloss": round(multiclass_logloss(all_test_targets, all_test_proba), 4),
            "bets": int(overall_betting["bets"]),
            "hit_rate": None if np.isnan(overall_betting["hit_rate"]) else round(float(overall_betting["hit_rate"]), 4),
            "roi": None if np.isnan(overall_betting["roi"]) else round(float(overall_betting["roi"]), 4),
            "avg_ev": None if np.isnan(overall_betting["avg_ev"]) else round(float(overall_betting["avg_ev"]), 4),
            "avg_edge": None if np.isnan(overall_betting["avg_edge"]) else round(float(overall_betting["avg_edge"]), 4),
            "threshold": round(float(best.threshold), 4),
        }
    )

    if args.export_summary:
        summary_path = resolve_path(args.export_summary)
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        monthly_summary.to_csv(summary_path, index=False)
        print(f"exported summary -> {summary_path}")

    if args.export_bets:
        bets_path = resolve_path(args.export_bets)
        all_bets = pd.concat(monthly_bets, ignore_index=True) if monthly_bets else ordered_test_df.iloc[0:0].copy()
        export_bets(all_bets, bets_path)
        print(f"exported bets {len(all_bets)} -> {bets_path}")


if __name__ == "__main__":
    main()
