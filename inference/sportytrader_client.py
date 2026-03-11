from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd


SPORTYTRADER_LEAGUE_CONFIGS = {
    "EPL": {
        "url": "https://www.sportytrader.com/en/odds/football/england/premier-league-49/",
        "title_contains": "Premier League",
        "section_title": "Upcoming Premier League matches",
    },
    "Bundesliga": {
        "url": "https://www.sportytrader.com/en/odds/football/germany/bundesliga-65/",
        "title_contains": "Bundesliga",
        "section_title": "Upcoming Bundesliga matches",
    },
    "Serie_A": {
        "url": "https://www.sportytrader.com/en/odds/football/italy/serie-a-79/",
        "title_contains": "Serie A",
        "section_title": "Upcoming Serie A matches",
    },
    "Ligue_1": {
        "url": "https://www.sportytrader.com/en/odds/football/france/ligue-1-123/",
        "title_contains": "Ligue 1",
        "section_title": "Upcoming Ligue 1 matches",
    },
}
NPX_EXECUTABLE = shutil.which("npx") or shutil.which("npx.cmd") or "npx"
PLAYWRIGHT_CLI = [
    NPX_EXECUTABLE,
    "--yes",
    "--package",
    "@playwright/cli",
    "playwright-cli",
]
DATE_LINE_RE = re.compile(r"^\d{1,2}\s+[A-Z][a-z]{2}\s+-\s+\d{2}:\d{2}$")
MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def run_playwright(args: list[str], *, timeout: float = 30.0, check: bool = True) -> str:
    proc = subprocess.run(
        PLAYWRIGHT_CLI + args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"playwright-cli {' '.join(args)} failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def _reset_playwright_state() -> None:
    run_playwright(["close-all"], timeout=20.0, check=False)
    run_playwright(["kill-all"], timeout=20.0, check=False)


def extract_result_block(output: str):
    match = re.search(r"### Result\s*(.*?)\s*### Ran Playwright code", output, flags=re.S)
    if not match:
        raise ValueError(f"Unable to parse Playwright result block from output:\n{output}")
    payload = match.group(1).strip()
    return json.loads(payload)


def wait_until_ready(*, wait_seconds: float, timeout_seconds: float, title_contains: str, section_title: str) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(wait_seconds)
        try:
            result = extract_result_block(
                run_playwright(
                    [
                        "eval",
                        "() => ({title: document.title, head: document.body.innerText.slice(0, 400)})",
                    ],
                    timeout=30.0,
                )
            )
        except Exception:
            continue

        if title_contains in result.get("title", "") and section_title in result.get("head", ""):
            return
    raise TimeoutError(f"Timed out waiting for Sportytrader page to become readable: {title_contains}")


def fetch_league_page_text(
    league: str,
    *,
    wait_seconds: float,
    timeout_seconds: float,
) -> str:
    if league not in SPORTYTRADER_LEAGUE_CONFIGS:
        raise KeyError(f"Unsupported Sportytrader league: {league}")

    config = SPORTYTRADER_LEAGUE_CONFIGS[league]
    _reset_playwright_state()
    try:
        run_playwright(["open", config["url"], "--headed"], timeout=40.0)
    except RuntimeError as exc:
        if "EADDRINUSE" not in str(exc):
            raise
        _reset_playwright_state()
        run_playwright(["open", config["url"], "--headed"], timeout=40.0)
    try:
        wait_until_ready(
            wait_seconds=wait_seconds,
            timeout_seconds=timeout_seconds,
            title_contains=config["title_contains"],
            section_title=config["section_title"],
        )
        result = extract_result_block(
            run_playwright(["eval", "() => document.body.innerText"], timeout=30.0)
        )
        if not isinstance(result, str):
            raise ValueError("Expected page innerText to be a string")
        return result
    finally:
        _reset_playwright_state()


def choose_year(day: int, month: int, date_from: pd.Timestamp) -> int:
    candidates = [date_from.year - 1, date_from.year, date_from.year + 1]
    target = date_from.normalize()
    best_year = date_from.year
    best_distance = None
    for year in candidates:
        try:
            candidate = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue
        distance = abs((candidate - target).days)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_year = year
    return best_year


def parse_fixture_timestamp(raw: str, date_from: pd.Timestamp) -> pd.Timestamp:
    date_part, time_part = raw.split(" - ", maxsplit=1)
    day_str, month_abbrev = date_part.split()
    month = MONTHS[month_abbrev]
    year = choose_year(int(day_str), month, date_from)
    return pd.Timestamp(f"{year:04d}-{month:02d}-{int(day_str):02d} {time_part}:00")


def parse_upcoming_fixtures(
    page_text: str,
    *,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
    league: str,
) -> pd.DataFrame:
    if league not in SPORTYTRADER_LEAGUE_CONFIGS:
        raise KeyError(f"Unsupported Sportytrader league: {league}")

    section_title = SPORTYTRADER_LEAGUE_CONFIGS[league]["section_title"]
    raw_lines = [line.strip().replace("\xa0", " ") for line in page_text.splitlines()]
    lines = [line for line in raw_lines if line]

    try:
        start = lines.index(section_title) + 1
    except ValueError as exc:
        raise ValueError(f"Could not find {section_title!r} section in Sportytrader page text") from exc

    fixtures: list[dict[str, object]] = []
    i = start
    while i + 7 < len(lines):
        if not DATE_LINE_RE.match(lines[i]):
            break
        if " - " not in lines[i + 1]:
            break
        if lines[i + 2] != "1" or lines[i + 4] != "X" or lines[i + 6] != "2":
            break

        home_team, away_team = [part.strip() for part in lines[i + 1].split(" - ", maxsplit=1)]
        fixtures.append(
            {
                "date": parse_fixture_timestamp(lines[i], date_from),
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "home_win_odds_open": float(lines[i + 3]),
                "draw_odds_open": float(lines[i + 5]),
                "away_win_odds_open": float(lines[i + 7]),
                "source": "sportytrader_playwright",
            }
        )
        i += 8

    frame = pd.DataFrame(fixtures)
    if frame.empty:
        return frame

    end_of_day = date_to + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return (
        frame[(frame["date"] >= date_from) & (frame["date"] <= end_of_day)]
        .sort_values(["date", "home_team", "away_team"])
        .reset_index(drop=True)
    )


def fetch_upcoming_epl_fixtures(*, date_from: pd.Timestamp, date_to: pd.Timestamp, wait_seconds: float, timeout_seconds: float) -> pd.DataFrame:
    return fetch_upcoming_league_fixtures(
        "EPL",
        date_from=date_from,
        date_to=date_to,
        wait_seconds=wait_seconds,
        timeout_seconds=timeout_seconds,
    )


def fetch_upcoming_league_fixtures(
    league: str,
    *,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
    wait_seconds: float,
    timeout_seconds: float,
) -> pd.DataFrame:
    page_text = fetch_league_page_text(
        league,
        wait_seconds=wait_seconds,
        timeout_seconds=timeout_seconds,
    )
    return parse_upcoming_fixtures(
        page_text,
        date_from=date_from,
        date_to=date_to,
        league=league,
    )


def fetch_upcoming_fixtures_for_leagues(
    leagues: list[str],
    *,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
    wait_seconds: float,
    timeout_seconds: float,
) -> pd.DataFrame:
    frames = [
        fetch_upcoming_league_fixtures(
            league,
            date_from=date_from,
            date_to=date_to,
            wait_seconds=wait_seconds,
            timeout_seconds=timeout_seconds,
        )
        for league in leagues
    ]
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "league",
                "home_team",
                "away_team",
                "home_win_odds_open",
                "draw_odds_open",
                "away_win_odds_open",
                "source",
            ]
        )
    return pd.concat(frames, ignore_index=True).sort_values(["date", "league", "home_team"]).reset_index(drop=True)
