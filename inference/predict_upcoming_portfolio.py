import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from inference.portfolio_presets import DEFAULT_PORTFOLIO_NAME, PORTFOLIO_PRESETS
from inference.live_tracking import append_tracking_rows, build_tracking_rows
from inference.upcoming_portfolio_strategy import (
    assign_flat_stakes,
    build_dataset_with_fixtures,
    dedupe_recommended_bets,
    load_historical_team_rows,
    prepare_fixture_frame,
    score_strategy_rows,
    train_frozen_models,
)


DEFAULT_DATA_DIR = REPO_ROOT / "Data"
DEFAULT_FIXTURES_PATH = SCRIPT_DIR / "output" / "sportytrader_upcoming_portfolio_odds.csv"
DEFAULT_OUTPUT_ALL = SCRIPT_DIR / "output" / "upcoming_portfolio_predictions.csv"
DEFAULT_OUTPUT_BETS = SCRIPT_DIR / "output" / "upcoming_portfolio_bets.csv"
DEFAULT_TRACKING_LEDGER = SCRIPT_DIR / "output" / "live_portfolio_bet_log.csv"


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--fixtures-csv", default=str(DEFAULT_FIXTURES_PATH))
    parser.add_argument("--portfolio", default=DEFAULT_PORTFOLIO_NAME)
    parser.add_argument("--bankroll-eur", type=float, default=50.0)
    parser.add_argument("--stake-fraction", type=float, default=0.05)
    parser.add_argument("--max-total-exposure-fraction", type=float, default=0.25)
    parser.add_argument("--output-all", default=str(DEFAULT_OUTPUT_ALL))
    parser.add_argument("--output-bets", default=str(DEFAULT_OUTPUT_BETS))
    parser.add_argument("--tracking-ledger", default=str(DEFAULT_TRACKING_LEDGER))
    return parser.parse_args()


def write_exports(
    scored: pd.DataFrame,
    bets: pd.DataFrame,
    *,
    output_all: Path,
    output_bets: Path,
) -> None:
    output_all.parent.mkdir(parents=True, exist_ok=True)
    output_bets.parent.mkdir(parents=True, exist_ok=True)

    all_cols = [
        "date",
        "league",
        "team_name",
        "opponent_name",
        "strategy_name",
        "train_league",
        "bet_league",
        "selected_outcome",
        "selected_odds",
        "pred_home_win",
        "pred_draw",
        "pred_away_win",
        "predicted_probability",
        "market_probability",
        "edge",
        "expected_value",
        "bet_is_market_favorite",
        "recommended_bet",
    ]
    scored[all_cols].to_csv(output_all, index=False)

    bet_cols = [
        "date",
        "league",
        "team_name",
        "opponent_name",
        "selected_outcome",
        "selected_odds",
        "predicted_probability",
        "market_probability",
        "edge",
        "expected_value",
        "strategy_names",
        "stake_eur",
        "max_total_exposure_eur",
        "potential_profit_eur_if_win",
    ]
    bets[bet_cols].to_csv(output_bets, index=False)


def main() -> None:
    args = parse_args()
    try:
        strategies = PORTFOLIO_PRESETS[args.portfolio]
    except KeyError as exc:
        raise KeyError(f"Unknown portfolio preset: {args.portfolio}") from exc

    team_rows = load_historical_team_rows(str(resolve_path(args.data_dir)))
    fixtures = prepare_fixture_frame(pd.read_csv(resolve_path(args.fixtures_csv)))
    dataset, future_match_ids = build_dataset_with_fixtures(team_rows, fixtures)

    if not future_match_ids:
        print("No future fixtures found in fixtures CSV.")
        return

    bundles = train_frozen_models(dataset, strategies)
    future_df = dataset[dataset["match_id"].isin(future_match_ids)].copy()
    future_df = future_df.sort_values(["date", "league", "team_name"]).reset_index(drop=True)
    scored = score_strategy_rows(future_df, bundles, strategies)
    bets = dedupe_recommended_bets(scored)
    bets = assign_flat_stakes(
        bets,
        bankroll_eur=args.bankroll_eur,
        stake_fraction=args.stake_fraction,
        max_total_exposure_fraction=args.max_total_exposure_fraction,
    )

    output_all = resolve_path(args.output_all)
    output_bets = resolve_path(args.output_bets)
    write_exports(scored, bets, output_all=output_all, output_bets=output_bets)

    tracking_ledger = resolve_path(args.tracking_ledger)
    tracking_rows = build_tracking_rows(bets, portfolio_name=args.portfolio)
    append_tracking_rows(tracking_rows, tracking_ledger)

    print(
        {
            "portfolio": args.portfolio,
            "fixtures_scored": int(future_df["match_id"].nunique()),
            "strategy_rows": int(len(scored)),
            "recommended_bets": int(len(bets)),
            "bankroll_eur": round(args.bankroll_eur, 2),
            "output_all": str(output_all),
            "output_bets": str(output_bets),
            "tracking_ledger": str(tracking_ledger),
        }
    )
    if not bets.empty:
        print(
            bets[
                [
                    "date",
                    "league",
                    "team_name",
                    "opponent_name",
                    "selected_outcome",
                    "selected_odds",
                    "expected_value",
                    "edge",
                    "strategy_names",
                    "stake_eur",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
