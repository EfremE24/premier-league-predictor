"""CLI: predict H/D/A outcome probabilities for one upcoming PL fixture.

Loads the persisted final model (models/rf_final_model.joblib) and the
feature-engineering state (models/feature_state.joblib, team Elo/form/
rest/PPG as of the end of the processed dataset) and computes the same 15
features pre_match_features() produces during training, for a fixture that
hasn't been played yet -- then runs the model on them.

Cold start for an unrecognized team (e.g. newly promoted, zero rows in
training history): pre_match_features() already falls back to the same
neutral priors used for a team's first-ever match in the training walk
(Elo 1500, 1.5 PPG for form/home-ppg/away-ppg, 7-day rest) rather than
raising -- verified directly against the persisted state before writing
this CLI, see tests/test_features.py::test_pre_match_features_handles_team_never_seen_before.
Predicting a fixture for a newly promoted club is normal input, not an
error case, so this CLI prints a note rather than failing when it happens,
and otherwise proceeds. This does mean an unrecognized team's strength is
almost certainly overstated (see MODEL_NOTES.md known limitations) -- the
note exists so that's visible, not hidden.

Usage:
    python src/predict.py "Arsenal" "Man City" 2026-09-13 \\
        --avg-h 2.60 --avg-d 3.70 --avg-a 2.65
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURE_COLUMNS, pre_match_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "rf_final_model.joblib"
STATE_PATH = MODELS_DIR / "feature_state.joblib"

CLASS_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}
NEUTRAL_ELO_LABEL = "1500 Elo, 1.5 PPG form, 7-day rest"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict H/D/A probabilities for a Premier League fixture.")
    parser.add_argument("home_team", help="Home team name, football-data.co.uk style (e.g. 'Man City')")
    parser.add_argument("away_team", help="Away team name")
    parser.add_argument("date", help="Fixture date, YYYY-MM-DD")
    parser.add_argument("--avg-h", type=float, required=True, help="Current average market odds, home win (decimal)")
    parser.add_argument("--avg-d", type=float, required=True, help="Current average market odds, draw (decimal)")
    parser.add_argument("--avg-a", type=float, required=True, help="Current average market odds, away win (decimal)")
    parser.add_argument("--season", default=None, help="Season label e.g. '2026/27'; inferred from date if omitted")
    args = parser.parse_args(argv)

    for flag, value in [("--avg-h", args.avg_h), ("--avg-d", args.avg_d), ("--avg-a", args.avg_a)]:
        if value <= 1.0:
            parser.error(f"{flag} must be a decimal odds value > 1.0, got {value}")
    return args


def infer_season(date: pd.Timestamp) -> str:
    """PL seasons run Aug-May; a date before July counts as part of the
    season that started the previous calendar year."""
    start_year = date.year if date.month >= 7 else date.year - 1
    return f"{start_year}/{str(start_year + 1)[2:]}"


def load_artifacts():
    if not MODEL_PATH.exists() or not STATE_PATH.exists():
        sys.exit(
            f"Missing model artifacts (expected {MODEL_PATH} and {STATE_PATH}). "
            "Run `python src/train.py` first to produce them."
        )
    model = joblib.load(MODEL_PATH)
    state = joblib.load(STATE_PATH)
    return model, state


def predict(model, state, home_team: str, away_team: str, date: pd.Timestamp,
            avg_h: float, avg_d: float, avg_a: float, season: str | None = None) -> tuple[dict, dict]:
    """Returns (proba_by_class, features) -- the features dict is the exact
    15 values sent to the model, returned alongside the prediction so a
    caller (the API, in particular) can show its work rather than just the
    output, without recomputing anything or duplicating pre_match_features
    logic itself."""
    season = season or infer_season(date)

    for team, role in [(home_team, "home"), (away_team, "away")]:
        if team not in state.elo:
            print(
                f"Note: '{team}' has no prior match history in feature_state -- "
                f"predicting with neutral cold-start priors ({NEUTRAL_ELO_LABEL}) for this team ({role})."
            )

    feats = pre_match_features(state, home_team, away_team, date, season, avg_h=avg_h, avg_d=avg_d, avg_a=avg_a)
    X = pd.DataFrame([feats])[FEATURE_COLUMNS]
    proba = model.predict_proba(X)[0]
    return dict(zip(model.classes_, proba)), feats


def main(argv=None) -> None:
    args = parse_args(argv)
    date = pd.Timestamp(args.date)
    season = args.season or infer_season(date)

    model, state = load_artifacts()
    proba_by_class, _feats = predict(model, state, args.home_team, args.away_team, date,
                                      args.avg_h, args.avg_d, args.avg_a, season=season)

    print(f"\n{args.home_team} vs {args.away_team} -- {date.date()} ({season})")
    print(f"Market odds: H {args.avg_h}  D {args.avg_d}  A {args.avg_a}")
    print("\nPredicted outcome probabilities:")
    for cls in ["H", "D", "A"]:
        print(f"  {CLASS_LABELS[cls]:10s} ({cls}): {proba_by_class[cls] * 100:5.1f}%")

    predicted = max(proba_by_class, key=proba_by_class.get)
    print(f"\nMost likely outcome: {CLASS_LABELS[predicted]} ({predicted})")


if __name__ == "__main__":
    main()
