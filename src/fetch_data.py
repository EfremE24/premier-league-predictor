"""Download Premier League match results + odds from football-data.co.uk.

Pulls one CSV per season (2015/16 - 2025/26), keeps a fixed set of columns,
and writes a single combined, date-sorted CSV to data/processed/matches.csv.

Odds columns: football-data has renamed its "market average" odds column
over time. We prefer, in order:
  1. AvgH/AvgD/AvgA   - average across all tracked bookmakers (2019/20+)
  2. BbAvH/BbAvD/BbAvA - same concept, old "Betbrain average" name (2015/16-2018/19)
  3. B365H/B365D/B365A - single bookmaker (Bet365), only if no average exists
Whichever source is used is recorded per-row in the OddsSource column so it's
possible to audit whether a season is on a cross-bookmaker average or a
single book's price.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

FIRST_SEASON_START_YEAR = 2015
LAST_SEASON_START_YEAR = 2025  # 2025/26

ODDS_FALLBACK_CHAIN = [
    ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"),
    ("B365H", "B365D", "B365A"),
]

REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def season_codes(start_year: int, end_year: int) -> list[str]:
    return [
        f"{y % 100:02d}{(y + 1) % 100:02d}"
        for y in range(start_year, end_year + 1)
    ]


def season_label(code: str) -> str:
    return f"20{code[:2]}/{code[2:]}"


def download_season(season: str, retries: int = 3, timeout: int = 20) -> Path | None:
    dest = RAW_DIR / f"{season}_E0.csv"
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    url = BASE_URL.format(season=season)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            if len(resp.content) < 100:
                raise ValueError("response too small, likely an empty/error page")
            dest.write_bytes(resp.content)
            return dest
        except (requests.RequestException, ValueError) as exc:
            print(f"    attempt {attempt}/{retries} failed: {exc}", file=sys.stderr)
            if attempt < retries:
                time.sleep(1.5)
    return None


def read_raw_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def pick_odds_columns(df: pd.DataFrame) -> tuple[str, str, str] | None:
    for h, d, a in ODDS_FALLBACK_CHAIN:
        if h in df.columns and d in df.columns and a in df.columns:
            return h, d, a
    return None


def load_season(path: Path, season: str) -> pd.DataFrame | None:
    df = read_raw_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"    missing required columns {missing}, skipping season", file=sys.stderr)
        return None

    df = df.dropna(subset=REQUIRED_COLS).copy()
    if df.empty:
        return None

    odds_cols = pick_odds_columns(df)
    if odds_cols is None:
        print("    no odds columns of any kind found, filling odds with NaN", file=sys.stderr)
        df["AvgH"], df["AvgD"], df["AvgA"] = pd.NA, pd.NA, pd.NA
        df["OddsSource"] = "none"
    else:
        h, d, a = odds_cols
        df["AvgH"] = pd.to_numeric(df[h], errors="coerce")
        df["AvgD"] = pd.to_numeric(df[d], errors="coerce")
        df["AvgA"] = pd.to_numeric(df[a], errors="coerce")
        df["OddsSource"] = h[:-1]  # "AvgH" -> "Avg", "BbAvH" -> "BbAv", "B365H" -> "B365"

    # football-data.co.uk mixes DD/MM/YYYY and DD/MM/YY across seasons/files.
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df["Season"] = season_label(season)

    out = df[[
        "Date", "Season", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR", "AvgH", "AvgD", "AvgA", "OddsSource",
    ]].copy()
    out["FTHG"] = out["FTHG"].astype(int)
    out["FTAG"] = out["FTAG"].astype(int)
    return out


def print_class_balance(df: pd.DataFrame) -> None:
    counts = df["FTR"].value_counts()
    pcts = df["FTR"].value_counts(normalize=True) * 100
    label_map = {"H": "Home win", "D": "Draw", "A": "Away win"}
    print("\nFTR class balance (all seasons combined):")
    for cls in ["H", "D", "A"]:
        n = int(counts.get(cls, 0))
        p = float(pcts.get(cls, 0.0))
        print(f"  {label_map[cls]:10s} ({cls}): {n:5d}  ({p:5.2f}%)")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    codes = season_codes(FIRST_SEASON_START_YEAR, LAST_SEASON_START_YEAR)
    frames = []

    for code in codes:
        label = season_label(code)
        print(f"Fetching {label} ({code})...")
        path = download_season(code)
        if path is None:
            print(f"  FAILED to download {label} after retries, skipping.", file=sys.stderr)
            continue

        df = load_season(path, code)
        if df is None or df.empty:
            print(f"  no usable rows for {label}, skipping.", file=sys.stderr)
            continue

        frames.append(df)
        print(f"  {len(df)} matches (odds source: {df['OddsSource'].iloc[0]})")

    if not frames:
        print("No seasons downloaded successfully, aborting.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("Date", kind="stable").reset_index(drop=True)

    out_path = PROCESSED_DIR / "matches.csv"
    combined.to_csv(out_path, index=False)

    print(f"\nSaved {len(combined)} matches across {len(frames)} seasons to {out_path}")
    print(f"Date range: {combined['Date'].min().date()} to {combined['Date'].max().date()}")
    print_class_balance(combined)


if __name__ == "__main__":
    main()
