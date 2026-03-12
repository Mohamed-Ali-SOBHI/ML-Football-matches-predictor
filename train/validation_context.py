from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from validation_io import selected_summary_rows


@dataclass(frozen=True)
class ValidationContext:
    bets_path: Path
    summary_path: Path | None
    selection_mode: str
    strategy_count: int | None
    clv_available: bool
    current_date: date


def detect_selection_mode(summary_df: pd.DataFrame, bets_path: Path) -> str:
    selected_df = selected_summary_rows(summary_df)
    if not selected_df.empty and "portfolio_selection_split" in selected_df.columns:
        values = [str(value) for value in selected_df["portfolio_selection_split"].dropna().unique() if str(value)]
        if len(values) == 1:
            return values[0]
    if not summary_df.empty and "portfolio_val_overlap" in summary_df.columns:
        return "val"
    name = bets_path.stem.lower()
    if "test_selected" in name:
        return "test"
    if "val" in name or "validation" in name or name == "positive_strategy_portfolio_bets":
        return "val"
    return "unknown"


def detect_strategy_count(summary_df: pd.DataFrame) -> int | None:
    selected_df = selected_summary_rows(summary_df)
    if selected_df.empty:
        return None
    return int(len(selected_df))


def has_closing_line_data(bets_df: pd.DataFrame) -> bool:
    if "closing_selected_odds" not in bets_df.columns:
        return False
    return bool(bets_df["closing_selected_odds"].notna().any())


def build_validation_context(
    *,
    bets_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    bets_path: Path,
    summary_path: Path | None,
    current_date: date,
) -> ValidationContext:
    return ValidationContext(
        bets_path=bets_path,
        summary_path=summary_path,
        selection_mode=detect_selection_mode(summary_df, bets_path),
        strategy_count=detect_strategy_count(summary_df),
        clv_available=has_closing_line_data(bets_df),
        current_date=current_date,
    )
