import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from inference.sportytrader_client import fetch_upcoming_epl_fixtures


DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "sportytrader_upcoming_epl_odds.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-from", required=True, help="Inclusive start date, YYYY-MM-DD")
    parser.add_argument("--date-to", required=True, help="Inclusive end date, YYYY-MM-DD")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--wait-seconds", type=float, default=8.0)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def main() -> None:
    args = parse_args()
    date_from = pd.Timestamp(args.date_from)
    date_to = pd.Timestamp(args.date_to)
    if date_to < date_from:
        raise ValueError("--date-to must be >= --date-from")

    fixtures = fetch_upcoming_epl_fixtures(
        date_from=date_from,
        date_to=date_to,
        wait_seconds=args.wait_seconds,
        timeout_seconds=args.timeout_seconds,
    )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(output_path, index=False)

    print(
        {
            "fixtures_found": int(len(fixtures)),
            "date_from": date_from.strftime("%Y-%m-%d"),
            "date_to": date_to.strftime("%Y-%m-%d"),
            "output": str(output_path),
        }
    )
    if not fixtures.empty:
        print(fixtures.to_string(index=False))


if __name__ == "__main__":
    main()
