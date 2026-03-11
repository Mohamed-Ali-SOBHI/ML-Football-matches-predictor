import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml_common import (
    build_xgb_model,
    evaluate_value_bets,
    get_feature_cols,
    load_dataset,
    make_sample_weight,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "dataset_home.csv"


@dataclass
class TrialResult:
    params: dict
    best_iteration: int
    val_acc: float
    val_logloss: float
    val_roi: float
    val_bets: int
    threshold: float


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path.cwd() / candidate).resolve())


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
        help="How to choose the model on the validation season. The threshold is always chosen on validation ROI.",
    )
    parser.add_argument("--export-bets", default="")
    return parser.parse_args()


def split_by_season(df, val_season: int, test_season: int):
    train_df = df[df["season"] < val_season].copy()
    val_df = df[df["season"] == val_season].copy()
    test_df = df[df["season"] == test_season].copy()

    if train_df.empty:
        raise ValueError(f"No training rows found before season {val_season}")
    if val_df.empty:
        raise ValueError(f"No validation rows found for season {val_season}")
    if test_df.empty:
        raise ValueError(f"No test rows found for season {test_season}")
    return train_df, val_df, test_df


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


def accuracy(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float((y_true == pred).mean())


def multiclass_logloss(y_true: np.ndarray, proba: np.ndarray) -> float:
    probs = np.clip(proba[np.arange(len(y_true)), y_true], 1e-15, 1.0)
    return float(-np.mean(np.log(probs)))


def threshold_candidates(start: float, stop: float, step: float) -> list[float]:
    values = np.arange(start, stop + step / 2.0, step)
    return [round(float(value), 10) for value in values]


def choose_threshold(
    val_df,
    proba: np.ndarray,
    candidates: list[float],
    min_val_bets: int,
) -> tuple[float, dict[str, float]]:
    scored: list[tuple[float, float, int, float]] = []

    for threshold in candidates:
        betting = evaluate_value_bets(val_df, proba, ev_threshold=threshold)
        bets = int(betting["bets"])
        roi = float(betting["roi"]) if not np.isnan(betting["roi"]) else float("-inf")
        avg_ev = float(betting["avg_ev"]) if not np.isnan(betting["avg_ev"]) else float("-inf")
        if bets >= min_val_bets:
            scored.append((roi, avg_ev, bets, threshold))

    if not scored:
        for threshold in candidates:
            betting = evaluate_value_bets(val_df, proba, ev_threshold=threshold)
            bets = int(betting["bets"])
            roi = float(betting["roi"]) if not np.isnan(betting["roi"]) else float("-inf")
            avg_ev = float(betting["avg_ev"]) if not np.isnan(betting["avg_ev"]) else float("-inf")
            scored.append((roi, avg_ev, bets, threshold))

    best_roi, _, _, best_threshold = max(scored, key=lambda row: (row[0], row[1], row[2], -row[3]))
    best_betting = evaluate_value_bets(val_df, proba, ev_threshold=best_threshold)
    return best_threshold, best_betting


def export_selected_bets(eval_df, proba: np.ndarray, threshold: float, output_path: str) -> int:
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

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not bet_mask.any():
        eval_df.iloc[0:0].copy().to_csv(output, index=False)
        return 0

    outcome_map = np.array(["away_win", "draw", "home_win"], dtype=object)
    bet_df = eval_df.loc[bet_mask].copy()
    bet_df["selected_outcome"] = outcome_map[chosen[bet_mask]]
    bet_df["selected_odds"] = odds[np.arange(len(odds)), chosen][bet_mask]
    bet_df["predicted_probability"] = proba[np.arange(len(proba)), chosen][bet_mask]
    bet_df["expected_value"] = chosen_ev[bet_mask]
    bet_df["won_bet"] = chosen[bet_mask] == bet_df["target"].astype(int).to_numpy()
    bet_df["profit"] = np.where(bet_df["won_bet"], bet_df["selected_odds"] - 1.0, -1.0)
    bet_df.to_csv(output, index=False)
    return len(bet_df)


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

    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int).to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["target"].astype(int).to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int).to_numpy()

    sample_weight = make_sample_weight(train_df["target"].astype(int))
    rng = np.random.default_rng(args.seed)
    thresholds = threshold_candidates(args.threshold_start, args.threshold_stop, args.threshold_step)

    best: TrialResult | None = None
    best_test_proba: np.ndarray | None = None

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
            best_test_proba = model.predict_proba(X_test)

        if (trial_idx + 1) % args.print_every == 0 or trial_idx == 0 or trial_idx == args.trials - 1:
            print(
                f"trial {trial_idx+1:03d}/{args.trials} "
                f"val_roi={result.val_roi:.4f} val_bets={result.val_bets} "
                f"val_acc={result.val_acc:.4f} val_logloss={result.val_logloss:.4f} "
                f"threshold={result.threshold:.2f} best_iter={result.best_iteration}"
            )
            print(f"  params: {params}")

    assert best is not None
    assert best_test_proba is not None

    print("\nBEST ON VALIDATION")
    print(
        f"selection_metric={args.selection_metric} "
        f"val_roi={best.val_roi:.4f} val_bets={best.val_bets} "
        f"val_acc={best.val_acc:.4f} val_logloss={best.val_logloss:.4f} "
        f"threshold={best.threshold:.2f} best_iter={best.best_iteration}"
    )
    print(best.params)

    pred_test = best_test_proba.argmax(axis=1)
    test_acc = accuracy(y_test, pred_test)
    test_logloss = multiclass_logloss(y_test, best_test_proba)
    test_betting = evaluate_value_bets(test_df, best_test_proba, ev_threshold=best.threshold)

    print("\nTEST")
    print(
        {
            "acc": round(test_acc, 4),
            "logloss": round(test_logloss, 4),
            "bets": int(test_betting["bets"]),
            "hit_rate": None if np.isnan(test_betting["hit_rate"]) else round(float(test_betting["hit_rate"]), 4),
            "roi": None if np.isnan(test_betting["roi"]) else round(float(test_betting["roi"]), 4),
            "avg_ev": None if np.isnan(test_betting["avg_ev"]) else round(float(test_betting["avg_ev"]), 4),
            "avg_edge": None if np.isnan(test_betting["avg_edge"]) else round(float(test_betting["avg_edge"]), 4),
        }
    )

    if args.export_bets:
        count = export_selected_bets(test_df, best_test_proba, best.threshold, resolve_path(args.export_bets))
        print(f"exported bets {count} -> {resolve_path(args.export_bets)}")


if __name__ == "__main__":
    main()
