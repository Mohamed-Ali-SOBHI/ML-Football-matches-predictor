import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from inference.upcoming_epl_strategy import (
    DrawStrategy,
    assign_flat_stakes,
    build_dataset_with_fixtures,
    load_historical_team_rows,
    prepare_fixture_frame,
    score_upcoming_matches,
    train_frozen_model,
)


DEFAULT_DATA_DIR = REPO_ROOT / "Data"
DEFAULT_FIXTURES_PATH = SCRIPT_DIR / "output" / "sportytrader_upcoming_epl_odds.csv"
DEFAULT_OUTPUT_ALL = SCRIPT_DIR / "output" / "upcoming_epl_predictions.csv"
DEFAULT_OUTPUT_BETS = SCRIPT_DIR / "output" / "upcoming_epl_draw_bets.csv"


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--fixtures-csv", default=str(DEFAULT_FIXTURES_PATH))
    parser.add_argument("--bankroll-eur", type=float, default=50.0)
    parser.add_argument("--stake-fraction", type=float, default=0.05)
    parser.add_argument("--max-total-exposure-fraction", type=float, default=0.25)
    parser.add_argument("--output-all", default=str(DEFAULT_OUTPUT_ALL))
    parser.add_argument("--output-bets", default=str(DEFAULT_OUTPUT_BETS))
    return parser.parse_args()


def write_exports(scored: pd.DataFrame, bets: pd.DataFrame, *, output_all: Path, output_bets: Path) -> None:
    output_all.parent.mkdir(parents=True, exist_ok=True)
    output_bets.parent.mkdir(parents=True, exist_ok=True)

    all_cols = [
        "date",
        "league",
        "team_name",
        "opponent_name",
        "market_home_win_odds_open",
        "market_draw_odds_open",
        "market_away_win_odds_open",
        "pred_home_win",
        "pred_draw",
        "pred_away_win",
        "draw_market_probability",
        "draw_edge",
        "draw_expected_value",
        "draw_is_market_favorite",
        "recommended_bet",
    ]
    scored[all_cols].to_csv(output_all, index=False)

    bet_cols = all_cols + [
        "stake_eur",
        "max_total_exposure_eur",
        "potential_profit_eur_if_win",
    ]
    bets[bet_cols].to_csv(output_bets, index=False)


def main() -> None:
    args = parse_args()

    team_rows = load_historical_team_rows(str(resolve_path(args.data_dir)))
    fixtures = prepare_fixture_frame(pd.read_csv(resolve_path(args.fixtures_csv)))
    dataset, future_match_ids = build_dataset_with_fixtures(team_rows, fixtures)

    if not future_match_ids:
        print("No future fixtures found in fixtures CSV.")
        return

    model, feature_cols = train_frozen_model(dataset)
    strategy = DrawStrategy()
    scored = score_upcoming_matches(dataset, future_match_ids, feature_cols, model, strategy)
    bets = assign_flat_stakes(
        scored[scored["recommended_bet"] == "draw"].copy(),
        bankroll_eur=args.bankroll_eur,
        stake_fraction=args.stake_fraction,
        max_total_exposure_fraction=args.max_total_exposure_fraction,
    )

    output_all = resolve_path(args.output_all)
    output_bets = resolve_path(args.output_bets)
    write_exports(scored, bets, output_all=output_all, output_bets=output_bets)

    print(
        {
            "fixtures_scored": int(len(scored)),
            "recommended_bets": int(len(bets)),
            "bankroll_eur": round(args.bankroll_eur, 2),
            "output_all": str(output_all),
            "output_bets": str(output_bets),
        }
    )
    if not bets.empty:
        print(
            bets[
                [
                    "date",
                    "team_name",
                    "opponent_name",
                    "market_draw_odds_open",
                    "pred_draw",
                    "draw_edge",
                    "draw_expected_value",
                    "stake_eur",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
