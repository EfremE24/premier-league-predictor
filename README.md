# Premier League Match Outcome Predictor

Predicts full-time result (home win / draw / away win) for English Premier
League matches using historical results and pre-match market odds.

Structural reference: an NFL win-probability project (features -> train ->
model selection -> predict), adapted for a 3-class outcome problem with a
much stronger class-imbalance and calibration story than a binary NFL win/
loss target.

## Status

Data pipeline complete (`src/fetch_data.py`). Feature engineering and
modeling are in progress — see commit history / project notes for the
current design decisions and why they were made.

## Project layout

```
data/
  raw/         one CSV per season, as downloaded (gitignored)
  processed/   combined matches.csv (gitignored)
src/
  fetch_data.py   download + combine PL seasons from football-data.co.uk
  features.py     feature engineering (WIP)
  train.py        model training + selection (WIP)
  predict.py      CLI prediction (WIP)
tests/
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/fetch_data.py
```

## Data source

[football-data.co.uk](https://www.football-data.co.uk/englandm.php) - free
historical match results and closing/average bookmaker odds for the
English Premier League (`E0`), seasons 2015/16 through 2025/26.

Columns kept: `Date, Season, HomeTeam, AwayTeam, FTHG, FTAG, FTR, AvgH,
AvgD, AvgA, OddsSource`.

`AvgH/AvgD/AvgA` are the market-average odds for home win / draw / away
win. football-data renamed this column over the years, so `fetch_data.py`
falls back through `AvgH/D/A` -> `BbAvH/D/A` (older "Betbrain average"
name) -> `B365H/D/A` (single-bookmaker odds, last resort). `OddsSource`
records which one was actually used for each row so this is auditable
rather than silently mixed.
