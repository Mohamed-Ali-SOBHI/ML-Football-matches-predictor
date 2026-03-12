from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def load_bets(path: str | Path) -> pd.DataFrame:
    csv_path = resolve_path(str(path))
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    if "won_bet" in df.columns:
        df["won_bet"] = df["won_bet"].astype(bool)
    return df


def load_summary(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    csv_path = resolve_path(str(path))
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def selected_summary_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    if "selected_for_portfolio" not in summary_df.columns:
        return summary_df.copy()
    selected = summary_df[summary_df["selected_for_portfolio"].fillna(False)].copy()
    if selected.empty:
        return summary_df.copy()
    return selected
