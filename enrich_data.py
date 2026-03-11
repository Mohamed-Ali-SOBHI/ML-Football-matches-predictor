import argparse
import glob

import pandas as pd

from market_data import enrich_team_rows_with_market_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Data")
    parser.add_argument("--max-date-diff-days", type=int, default=14)
    args = parser.parse_args()

    paths = sorted(glob.glob(f"{args.data_dir}/**/*.csv", recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {args.data_dir!r}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["_source_path"] = path
        frame["_row_order"] = range(len(frame))
        frames.append(frame)

    all_rows = pd.concat(frames, ignore_index=True)
    enriched, match_market = enrich_team_rows_with_market_data(
        all_rows,
        max_date_diff_days=args.max_date_diff_days,
    )

    for path, group in enriched.groupby("_source_path", sort=False):
        output = group.sort_values("_row_order").drop(columns=["_source_path", "_row_order"])
        output.to_csv(path, index=False)

    diff_counts = match_market["market_date_diff_days"].value_counts().sort_index().to_dict()
    missing_opening_odds = int(
        match_market[["home_win_odds_open", "draw_odds_open", "away_win_odds_open"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    print(f"Enriched {len(paths)} files and {len(match_market)} matches")
    print(f"Market date diff distribution: {diff_counts}")
    print(f"Matches missing complete opening odds: {missing_opening_odds}")


if __name__ == "__main__":
    main()
