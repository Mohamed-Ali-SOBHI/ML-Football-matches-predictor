import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

from ml_common import (
    build_xgb_model,
    get_feature_cols,
    load_dataset,
    make_sample_weight,
    split_grouped_threeway,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "dataset_home.csv"


@dataclass
class TrialResult:
    params: dict
    best_iteration: int
    val_logloss: float
    val_acc: float


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


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path.cwd() / candidate).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataset(resolve_path(args.data))
    feature_cols = get_feature_cols(df)
    model_df = df.dropna(subset=["target"]).copy()

    train_df, val_df, test_df = split_grouped_threeway(
        model_df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df["target"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int)

    print(f"rows: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    baseline_class = int(y_train.value_counts().idxmax())
    baseline_pred = np.full_like(y_test, baseline_class)
    print("baseline test acc", float(accuracy_score(y_test, baseline_pred)))

    sample_weight = make_sample_weight(y_train)
    rng = np.random.default_rng(args.seed)

    best: TrialResult | None = None

    for t in range(args.trials):
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
        val_ll = float(log_loss(y_val, proba_val, labels=[0, 1, 2]))
        val_acc = float(accuracy_score(y_val, pred_val))

        res = TrialResult(
            params=params,
            best_iteration=int(getattr(model, "best_iteration", model.n_estimators - 1)),
            val_logloss=val_ll,
            val_acc=val_acc,
        )

        if best is None or res.val_logloss < best.val_logloss:
            best = res

        if (t + 1) % args.print_every == 0 or t == 0 or t == args.trials - 1:
            print(
                f"trial {t+1:03d}/{args.trials} val_logloss={val_ll:.4f} val_acc={val_acc:.4f} "
                f"best_iter={res.best_iteration} best_val={best.val_logloss:.4f}"
            )
            print(f"  params: {params}")

    assert best is not None

    print("\nBEST (by val logloss)")
    print(f"val_logloss={best.val_logloss:.4f} val_acc={best.val_acc:.4f} best_iter={best.best_iteration}")
    print(best.params)

    trainval = pd.concat([train_df, val_df], ignore_index=True)
    X_trainval = trainval[feature_cols]
    y_trainval = trainval["target"].astype(int)
    sample_weight_trainval = make_sample_weight(y_trainval)

    final_model = build_xgb_model(
        seed=args.seed,
        n_estimators=max(best.best_iteration + 1, 50),
        **best.params,
    )
    final_model.fit(X_trainval, y_trainval, sample_weight=sample_weight_trainval)

    proba_test = final_model.predict_proba(X_test)
    pred_test = proba_test.argmax(axis=1)

    print("\nTEST")
    print("acc", float(accuracy_score(y_test, pred_test)))
    print("logloss", float(log_loss(y_test, proba_test, labels=[0, 1, 2])))
    print("confusion matrix (rows=true, cols=pred)")
    print(confusion_matrix(y_test, pred_test))
    print("classification report")
    print(classification_report(y_test, pred_test, target_names=["L", "D", "W"], zero_division=0))


if __name__ == "__main__":
    main()
