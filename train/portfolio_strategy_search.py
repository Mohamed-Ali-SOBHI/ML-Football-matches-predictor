from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_common import build_xgb_model, get_feature_cols, load_dataset, make_sample_weight
from strategy_search_common import (
    StrategyFamily,
    apply_strategy,
    build_base_bets,
    parse_list_argument,
    parse_odds_ranges,
    sample_params,
    summarize_bets,
    threshold_values,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "dataset_home.csv"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "output" / "positive_strategy_portfolio_summary.csv"
DEFAULT_BETS_PATH = SCRIPT_DIR / "output" / "positive_strategy_portfolio_bets.csv"


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
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-scopes", default="ALL,LOCAL")
    parser.add_argument("--bet-leagues", default="EPL,Bundesliga,La_liga,Ligue_1,Serie_A")
    parser.add_argument("--outcomes", default="home_win,draw,away_win")
    parser.add_argument("--odds-ranges", default="1.30:2.20,2.20:4.00,4.00:10.00,2.00:10.00")
    parser.add_argument("--market-favorite-modes", default="favorite,nonfavorite")
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--edge-values", default="0.0,0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--min-val-bets", type=int, default=25)
    parser.add_argument("--min-val-roi", type=float, default=0.02)
    parser.add_argument("--max-strategies", type=int, default=4)
    parser.add_argument("--max-val-overlap", type=float, default=0.35)
    parser.add_argument("--portfolio-selection-split", choices=["val", "test"], default="val")
    parser.add_argument("--selection-min-roi", type=float, default=0.0)
    parser.add_argument("--test-fit-scope", choices=["pretest", "train"], default="train")
    parser.add_argument("--export-summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--export-bets", default=str(DEFAULT_BETS_PATH))
    return parser.parse_args()


def normalize_train_scope(scope: str, *, bet_league: str) -> str:
    upper = scope.upper()
    if upper == "ALL":
        return ""
    if upper == "LOCAL":
        return bet_league
    return scope


def build_families(args: argparse.Namespace) -> list[StrategyFamily]:
    train_scopes = parse_list_argument(args.train_scopes)
    bet_leagues = parse_list_argument(args.bet_leagues)
    outcomes = parse_list_argument(args.outcomes)
    market_favorite_modes = parse_list_argument(args.market_favorite_modes)
    odds_ranges = parse_odds_ranges(args.odds_ranges)

    families: list[StrategyFamily] = []
    seen: set[tuple[str, str, str, float, float, str]] = set()
    for bet_league in bet_leagues:
        for scope in train_scopes:
            train_league = normalize_train_scope(scope, bet_league=bet_league)
            for outcome in outcomes:
                for odds_min, odds_max in odds_ranges:
                    for market_favorite_mode in market_favorite_modes:
                        key = (
                            train_league,
                            bet_league,
                            outcome,
                            odds_min,
                            odds_max,
                            market_favorite_mode,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        families.append(
                            StrategyFamily(
                                train_league=train_league,
                                bet_league=bet_league,
                                outcome=outcome,
                                odds_min=odds_min,
                                odds_max=odds_max,
                                market_favorite_mode=market_favorite_mode,
                            )
                        )
    return families


def static_family_filter(base_bets: pd.DataFrame, family: StrategyFamily) -> pd.Series:
    mask = base_bets["selected_outcome"] == family.outcome
    if family.bet_league:
        mask &= base_bets["league"] == family.bet_league
    mask &= (base_bets["selected_odds"] >= family.odds_min) & (base_bets["selected_odds"] < family.odds_max)
    if family.market_favorite_mode == "favorite":
        mask &= base_bets["bet_is_market_favorite"]
    elif family.market_favorite_mode == "nonfavorite":
        mask &= ~base_bets["bet_is_market_favorite"]
    return mask


def candidate_better(candidate: dict, current: dict | None) -> bool:
    if current is None:
        return True
    return (
        candidate["val_roi"],
        candidate["val_profit"],
        candidate["val_bets"],
    ) > (
        current["val_roi"],
        current["val_profit"],
        current["val_bets"],
    )


def overlap_ratio(candidate_keys: set[str], selected_keys: set[str]) -> float:
    if not candidate_keys:
        return 0.0
    return len(candidate_keys & selected_keys) / float(len(candidate_keys))


def conflict_count(match_outcomes: dict[str, str], bets: pd.DataFrame) -> int:
    conflicts = 0
    for row in bets.itertuples(index=False):
        existing = match_outcomes.get(str(row.match_id))
        if existing is not None and existing != row.selected_outcome:
            conflicts += 1
    return conflicts


def append_strategy_tag(bets: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    tagged = bets.copy()
    tagged["strategy_name"] = strategy_name
    return tagged


def combine_portfolio_bets(candidates: list[dict], split_name: str) -> pd.DataFrame:
    tagged_frames = []
    for candidate in candidates:
        bets = candidate[f"{split_name}_bets_df"]
        if bets.empty:
            continue
        tagged_frames.append(append_strategy_tag(bets, candidate["strategy_name"]))

    if not tagged_frames:
        return pd.DataFrame()

    all_bets = pd.concat(tagged_frames, ignore_index=True)
    strategy_names = (
        all_bets.groupby("bet_key", sort=False)["strategy_name"]
        .agg(lambda values: "|".join(dict.fromkeys(values)))
        .rename("strategy_names")
    )
    deduped = (
        all_bets.sort_values(["bet_key", "expected_value", "edge"], ascending=[True, False, False])
        .drop_duplicates(subset=["bet_key"], keep="first")
        .copy()
    )
    deduped = deduped.merge(strategy_names, on="bet_key", how="left", validate="one_to_one")
    return deduped


def select_portfolio(
    candidates: list[dict],
    *,
    split_name: str,
    max_strategies: int,
    max_overlap: float,
    selection_min_roi: float,
) -> list[dict]:
    selected: list[dict] = []
    selected_keys: set[str] = set()
    match_outcomes: dict[str, str] = {}
    bets_key = f"{split_name}_bets_df"
    roi_key = f"{split_name}_roi"
    profit_key = f"{split_name}_profit"
    count_key = f"{split_name}_bets"

    ordered = sorted(
        candidates,
        key=lambda item: (
            -np.inf if item[roi_key] is None else item[roi_key],
            item[profit_key],
            item[count_key],
        ),
        reverse=True,
    )

    for candidate in ordered:
        split_roi = candidate[roi_key]
        if split_roi is None or split_roi < selection_min_roi:
            continue

        split_bets = candidate[bets_key]
        split_keys = set(split_bets["bet_key"].astype(str))
        overlap = overlap_ratio(split_keys, selected_keys)
        conflicts = conflict_count(match_outcomes, split_bets)

        candidate["portfolio_selection_split"] = split_name
        candidate["portfolio_selection_overlap"] = overlap
        candidate["portfolio_selection_conflicts"] = conflicts

        if overlap > max_overlap or conflicts > 0:
            continue

        selected.append(candidate)
        selected_keys |= split_keys
        for row in split_bets.itertuples(index=False):
            match_outcomes[str(row.match_id)] = row.selected_outcome
        if len(selected) >= max_strategies:
            break

    return selected


def train_and_score(train_df: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], params: dict, seed: int) -> pd.DataFrame:
    model = build_xgb_model(seed=seed, n_estimators=500, **params)
    y_train = train_df["target"].astype(int)
    model.fit(
        train_df[feature_cols],
        y_train,
        sample_weight=make_sample_weight(y_train),
    )
    return build_base_bets(eval_df, model.predict_proba(eval_df[feature_cols]))


def build_test_train_frame(
    df: pd.DataFrame,
    *,
    val_season: int,
    train_league: str,
    test_fit_scope: str,
) -> pd.DataFrame:
    if test_fit_scope == "train":
        train_mask = df["season"] < val_season
    else:
        train_mask = df["season"] <= val_season

    if train_league != "ALL":
        train_mask &= df["league"] == train_league
    return df[train_mask].copy()


def main() -> None:
    args = parse_args()

    df = load_dataset(resolve_path(args.data)).dropna(subset=["target"]).copy()
    families = build_families(args)
    thresholds = threshold_values(args.threshold_start, args.threshold_stop, args.threshold_step)
    edge_values = [float(value) for value in parse_list_argument(args.edge_values)]
    rng = np.random.default_rng(args.seed)

    val_df = df[df["season"] == args.val_season].copy()
    test_df = df[df["season"] == args.test_season].copy()
    if val_df.empty or test_df.empty:
        raise ValueError("Validation or test split is empty")

    grouped_families: dict[str, list[StrategyFamily]] = {}
    for family in families:
        grouped_families.setdefault(family.train_league, []).append(family)

    best_by_strategy: dict[str, dict] = {}

    for train_league, family_group in grouped_families.items():
        train_mask = df["season"] < args.val_season
        if train_league:
            train_mask &= df["league"] == train_league
        train_df = df[train_mask].copy()
        if train_df.empty:
            continue

        feature_cols = get_feature_cols(train_df)
        print(
            f"search train_league={train_league or 'ALL'} "
            f"families={len(family_group)} train_rows={len(train_df)}"
        )

        for trial_idx in range(args.trials):
            params = sample_params(rng)
            val_base = train_and_score(train_df, val_df, feature_cols, params, args.seed)

            static_frames = {
                family.name: val_base[static_family_filter(val_base, family)].copy()
                for family in family_group
            }

            improved = 0
            for family in family_group:
                family_base = static_frames[family.name]
                if family_base.empty:
                    continue

                current_best = best_by_strategy.get(family.name)
                for threshold in thresholds:
                    threshold_bets = family_base[family_base["expected_value"] > threshold]
                    if len(threshold_bets) < args.min_val_bets:
                        continue
                    for edge_min in edge_values:
                        val_bets = threshold_bets[threshold_bets["edge"] >= edge_min].copy()
                        if len(val_bets) < args.min_val_bets:
                            continue

                        val_summary = summarize_bets(val_bets, "val")
                        val_roi = val_summary["val_roi"]
                        if val_roi is None or val_roi < args.min_val_roi:
                            continue

                        candidate = {
                            "strategy_name": family.name,
                            "train_league": family.train_league or "ALL",
                            "bet_league": family.bet_league,
                            "outcome": family.outcome,
                            "odds_min": family.odds_min,
                            "odds_max": family.odds_max,
                            "market_favorite_mode": family.market_favorite_mode,
                            "params": params,
                            "threshold": threshold,
                            "edge_min": edge_min,
                            "feature_cols": feature_cols,
                            "val_bets_df": val_bets.copy(),
                            **val_summary,
                        }
                        if candidate_better(candidate, current_best):
                            best_by_strategy[family.name] = candidate
                            current_best = candidate
                            improved += 1

            print(
                f"  trial {trial_idx+1:03d}/{args.trials} "
                f"improved={improved} current_candidates={len(best_by_strategy)}"
            )

    if not best_by_strategy:
        raise ValueError("No candidate strategy satisfied the validation constraints")

    all_candidates = list(best_by_strategy.values())
    print(f"\nvalidation survivors={len(all_candidates)}")

    for candidate in all_candidates:
        train_league = candidate["train_league"]
        test_train_df = build_test_train_frame(
            df,
            val_season=args.val_season,
            train_league=train_league,
            test_fit_scope=args.test_fit_scope,
        )
        test_base = train_and_score(test_train_df, test_df, candidate["feature_cols"], candidate["params"], args.seed)
        test_bets = apply_strategy(
            test_base,
            threshold=candidate["threshold"],
            edge_min=candidate["edge_min"],
            bet_league=candidate["bet_league"],
            outcome=candidate["outcome"],
            odds_min=candidate["odds_min"],
            odds_max=candidate["odds_max"],
            market_favorite_mode=candidate["market_favorite_mode"],
        )
        candidate["test_bets_df"] = test_bets.copy()
        candidate.update(summarize_bets(test_bets, "test"))

    selected = select_portfolio(
        all_candidates,
        split_name=args.portfolio_selection_split,
        max_strategies=args.max_strategies,
        max_overlap=args.max_val_overlap,
        selection_min_roi=args.selection_min_roi,
    )

    portfolio_val = combine_portfolio_bets(selected, "val")
    portfolio_test = combine_portfolio_bets(selected, "test")
    portfolio_val_summary = summarize_bets(portfolio_val, "portfolio_val")
    portfolio_test_summary = summarize_bets(portfolio_test, "portfolio_test")

    summary_rows = []
    selected_names = {candidate["strategy_name"] for candidate in selected}
    for candidate in sorted(
        all_candidates,
        key=lambda item: (item["val_roi"], item["val_profit"], item["val_bets"]),
        reverse=True,
    ):
        row = {
            "selected_for_portfolio": candidate["strategy_name"] in selected_names,
            **{k: v for k, v in candidate.items() if not k.endswith("_df") and k not in {"params", "feature_cols"}},
            "test_fit_scope": args.test_fit_scope,
            "params": str(candidate["params"]),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    output_summary = Path(resolve_path(args.export_summary))
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_summary, index=False)

    output_bets = Path(resolve_path(args.export_bets))
    output_bets.parent.mkdir(parents=True, exist_ok=True)
    portfolio_test.to_csv(output_bets, index=False)

    print("\nSELECTED PORTFOLIO")
    for idx, candidate in enumerate(selected, start=1):
        print(
            f"{idx}. {candidate['strategy_name']} "
            f"val_roi={candidate['val_roi']:.4f} val_bets={candidate['val_bets']} "
            f"test_roi={candidate['test_roi'] if candidate['test_roi'] is not None else 'NA'} "
            f"test_bets={candidate['test_bets']}"
        )

    print("\nPORTFOLIO SUMMARY")
    print(
        {
            "portfolio_selection_split": args.portfolio_selection_split,
            "selection_min_roi": args.selection_min_roi,
            "max_strategies": args.max_strategies,
            "max_overlap": args.max_val_overlap,
            "test_fit_scope": args.test_fit_scope,
        }
    )
    print({**portfolio_val_summary, **portfolio_test_summary})
    print(f"exported summary -> {output_summary}")
    print(f"exported bets -> {output_bets}")


if __name__ == "__main__":
    main()
