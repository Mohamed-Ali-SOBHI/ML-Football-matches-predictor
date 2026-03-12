from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TRACKING_COLUMNS = [
    "snapshot_key",
    "prediction_generated_at_utc",
    "portfolio_name",
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
    "result_status",
    "closing_selected_odds",
    "realized_profit",
]


def build_tracking_rows(bets: pd.DataFrame, *, portfolio_name: str) -> pd.DataFrame:
    if bets.empty:
        return bets.copy()

    tracked = bets.copy()
    tracked["prediction_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    tracked["portfolio_name"] = portfolio_name
    tracked["snapshot_key"] = (
        tracked["portfolio_name"].astype(str)
        + "|"
        + tracked["date"].astype(str)
        + "|"
        + tracked["league"].astype(str)
        + "|"
        + tracked["team_name"].astype(str)
        + "|"
        + tracked["opponent_name"].astype(str)
        + "|"
        + tracked["selected_outcome"].astype(str)
        + "|"
        + tracked["strategy_names"].astype(str)
    )
    tracked["result_status"] = "pending"
    tracked["closing_selected_odds"] = pd.NA
    tracked["realized_profit"] = pd.NA
    return tracked[TRACKING_COLUMNS].copy()


def append_tracking_rows(tracking_rows: pd.DataFrame, ledger_path: Path) -> None:
    if tracking_rows.empty:
        return

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        existing = pd.read_csv(ledger_path)
        combined = pd.concat([existing, tracking_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["snapshot_key"], keep="last")
    else:
        combined = tracking_rows.copy()
    combined.to_csv(ledger_path, index=False)
