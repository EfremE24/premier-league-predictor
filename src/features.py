"""Feature engineering for Premier League match outcome prediction.

Reads data/processed/matches.csv and writes data/processed/features.csv,
one row per match, with only information that was actually available
before kickoff.

Design note on leakage: every feature below is computed by a single pass
over matches in chronological order, maintaining per-team running state
(Elo rating, recent form, home/away scoring history, last match date).
For each match we READ the state (that's the feature), THEN update it with
the match's actual result. This makes it mechanically impossible for a
feature to see its own match's outcome or any later match. A groupby/
rolling approach was considered and rejected: each team plays both home
and away across a season, and getting a rolling window right across both
roles without an off-by-one leak is easy to get subtly wrong and hard to
verify by inspection. A sequential pass is slower but its correctness is
obvious by construction, which matters more here than speed at ~4k rows.

Feature groups:
  - Elo rating (elo_home, elo_away, elo_gap): long-run team strength,
    updated after every match, with mild regression to the mean at each
    team's first match of a new season (transfers/squad turnover mean a
    team's rating shouldn't carry over at full strength).
  - Form (form_home, form_away, form_gap): points-per-game over each
    team's last 5 matches, any venue. Captures short-term hot/cold streaks
    that a slow-moving Elo rating won't pick up.
  - Home/away split PPG (home_ppg, away_ppg, home_advantage_gap): the
    home team's historical points-per-game specifically in home matches,
    vs. the away team's historical PPG specifically in away matches. Some
    teams are much stronger at home than their overall record suggests;
    this isolates that rather than relying on the model to infer it.
  - Rest days (rest_days_home, rest_days_away, rest_days_gap): days since
    each team's previous match.
  - Market-implied probability (imp_prob_h, imp_prob_d, imp_prob_a): the
    average market odds converted to probabilities and normalized to
    remove the bookmaker's overround (raw 1/odds implied probabilities
    sum to >1; that excess is the bookmaker's margin, not signal).

Cold-start convention: a team's first-ever match in the dataset (or first
home/away match, or first match of the whole window) has no history to
compute a feature from. Rather than impute from the data itself (which
would leak a whole-dataset average into early rows), each feature falls
back to a fixed, domain-chosen neutral prior documented next to the
constant. This trades a small amount of accuracy on a handful of edge-of-
history rows for a fallback that is trivially leakage-free.

Reusable state: the per-team running state (Elo, form, PPG splits, last
match date) that this module accumulates while walking matches.csv is
exactly what's needed to compute the same 15 features for a fixture that
HASN'T been played yet -- that's how predict.py produces inference-time
features rather than just training-time ones. So that state lives in an
explicit, picklable FeatureState object rather than as loop-local dict
variables, and the per-match logic is split into two pure steps:
pre_match_features() (read state -> feature dict, the thing both training
and inference need) and update_state() (apply an actual result -> state,
only relevant once a match has actually been played). run_feature_pipeline()
drives both across a full match log and hands back the final FeatureState
alongside the features DataFrame, so a caller (train.py, eventually
predict.py) can persist "team state as of right now" for later use.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "matches.csv"
OUTPUT_PATH = PROCESSED_DIR / "features.csv"

# Elo parameters (fixed, literature-typical values -- not fit to this
# data; tuning these is a reasonable future step, not done here).
INITIAL_ELO = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0  # points added to home team's rating in the expected-score calc only
SEASON_REGRESSION = 0.25  # fraction of the gap to 1500 erased at each team's first match of a new season

FORM_WINDOW = 5
NEUTRAL_PPG = 1.5  # midpoint of the 0-3 points-per-game scale; used when a team has no relevant history yet
NEUTRAL_REST_DAYS = 7  # typical one-week gap; used for a team's first match in the dataset

# A team continuously in the Premier League never goes longer than the summer
# break (~90-105 days in this data) without a match. A gap larger than that
# means the team was relegated and spent one or more seasons in the
# Championship, which this dataset doesn't cover -- e.g. Sunderland's 2025/26
# return shows a raw gap of 3009 days. That's "team was out of the dataset",
# not "team is well-rested", so it's treated as a cold start (neutral prior)
# rather than fed to the model as a genuine rest-days value.
MAX_MEANINGFUL_REST_DAYS = 120

FEATURE_COLUMNS = [
    "elo_home", "elo_away", "elo_gap",
    "form_home", "form_away", "form_gap",
    "home_ppg", "away_ppg", "home_advantage_gap",
    "rest_days_home", "rest_days_away", "rest_days_gap",
    "imp_prob_h", "imp_prob_d", "imp_prob_a",
]

PASSTHROUGH_COLUMNS = ["Date", "Season", "HomeTeam", "AwayTeam", "FTR"]


def match_points(ftr: str, venue: str) -> int:
    """Points earned by the team on `venue` ('H' or 'A') given the result."""
    if ftr == "D":
        return 1
    return 3 if ftr == venue else 0


def implied_probabilities(avg_h: float, avg_a: float, avg_d: float) -> tuple[float, float, float]:
    raw_h, raw_d, raw_a = 1.0 / avg_h, 1.0 / avg_d, 1.0 / avg_a
    overround = raw_h + raw_d + raw_a
    return raw_h / overround, raw_d / overround, raw_a / overround


@dataclass
class FeatureState:
    """All per-team running state the feature pipeline needs. Read (not
    mutated) by pre_match_features() to produce a feature row; mutated by
    update_state() once that row's actual result is known. Picklable as-is,
    so this is what gets persisted for inference-time feature generation."""

    elo: dict = field(default_factory=dict)
    elo_season: dict = field(default_factory=dict)
    form: dict = field(default_factory=dict)
    home_points_sum: dict = field(default_factory=dict)
    home_matches: dict = field(default_factory=dict)
    away_points_sum: dict = field(default_factory=dict)
    away_matches: dict = field(default_factory=dict)
    last_played: dict = field(default_factory=dict)


def pre_match_features(state: FeatureState, home: str, away: str, date, season: str,
                        avg_h: float, avg_d: float, avg_a: float) -> dict:
    """The 15 FEATURE_COLUMNS values for a match about to be played, from
    state as of right before kickoff. Does not require the match to have
    happened -- this is the exact function predict.py will call for a
    genuinely upcoming fixture, given a loaded FeatureState."""
    for team in (home, away):
        state.elo.setdefault(team, INITIAL_ELO)
        if state.elo_season.get(team) is not None and state.elo_season[team] != season:
            state.elo[team] = INITIAL_ELO + (1 - SEASON_REGRESSION) * (state.elo[team] - INITIAL_ELO)
        state.elo_season[team] = season

    elo_home_pre, elo_away_pre = state.elo[home], state.elo[away]

    home_form_hist = state.form.get(home, [])
    away_form_hist = state.form.get(away, [])
    form_home = sum(home_form_hist) / len(home_form_hist) if home_form_hist else NEUTRAL_PPG
    form_away = sum(away_form_hist) / len(away_form_hist) if away_form_hist else NEUTRAL_PPG

    home_ppg = (
        state.home_points_sum[home] / state.home_matches[home]
        if state.home_matches.get(home) else NEUTRAL_PPG
    )
    away_ppg = (
        state.away_points_sum[away] / state.away_matches[away]
        if state.away_matches.get(away) else NEUTRAL_PPG
    )

    rest_home = (date - state.last_played[home]).days if home in state.last_played else NEUTRAL_REST_DAYS
    rest_away = (date - state.last_played[away]).days if away in state.last_played else NEUTRAL_REST_DAYS
    if rest_home > MAX_MEANINGFUL_REST_DAYS:
        rest_home = NEUTRAL_REST_DAYS
    if rest_away > MAX_MEANINGFUL_REST_DAYS:
        rest_away = NEUTRAL_REST_DAYS

    imp_h, imp_d, imp_a = implied_probabilities(avg_h=avg_h, avg_a=avg_a, avg_d=avg_d)

    return {
        "elo_home": elo_home_pre, "elo_away": elo_away_pre, "elo_gap": elo_home_pre - elo_away_pre,
        "form_home": form_home, "form_away": form_away, "form_gap": form_home - form_away,
        "home_ppg": home_ppg, "away_ppg": away_ppg, "home_advantage_gap": home_ppg - away_ppg,
        "rest_days_home": rest_home, "rest_days_away": rest_away, "rest_days_gap": rest_home - rest_away,
        "imp_prob_h": imp_h, "imp_prob_d": imp_d, "imp_prob_a": imp_a,
    }


def update_state(state: FeatureState, home: str, away: str, date, ftr: str) -> None:
    """Apply a match's ACTUAL result to state. Only ever called once that
    result is known -- never for the fixture whose features were just read."""
    elo_home_pre, elo_away_pre = state.elo[home], state.elo[away]
    s_home = {"H": 1.0, "D": 0.5, "A": 0.0}[ftr]
    expected_home = 1.0 / (1.0 + 10 ** (-((elo_home_pre + ELO_HOME_ADVANTAGE) - elo_away_pre) / 400))
    delta = ELO_K * (s_home - expected_home)
    state.elo[home] = elo_home_pre + delta
    state.elo[away] = elo_away_pre - delta

    pts_home = match_points(ftr, "H")
    pts_away = match_points(ftr, "A")
    state.form.setdefault(home, []).append(pts_home)
    state.form.setdefault(away, []).append(pts_away)
    state.form[home] = state.form[home][-FORM_WINDOW:]
    state.form[away] = state.form[away][-FORM_WINDOW:]

    state.home_points_sum[home] = state.home_points_sum.get(home, 0) + pts_home
    state.home_matches[home] = state.home_matches.get(home, 0) + 1
    state.away_points_sum[away] = state.away_points_sum.get(away, 0) + pts_away
    state.away_matches[away] = state.away_matches.get(away, 0) + 1

    state.last_played[home] = date
    state.last_played[away] = date


def run_feature_pipeline(matches: pd.DataFrame) -> tuple[pd.DataFrame, FeatureState]:
    """Drive pre_match_features()/update_state() across a full match log in
    chronological order. Returns the features DataFrame (one row per match)
    AND the final FeatureState (team ratings/form/etc. as of the last row) --
    the latter is what a caller persists to compute features for matches
    that come after this log ends."""
    matches = matches.sort_values("Date", kind="stable").reset_index(drop=True)
    state = FeatureState()
    rows = []

    for row in matches.itertuples(index=False):
        home, away = row.HomeTeam, row.AwayTeam
        date, season = row.Date, row.Season

        feats = pre_match_features(state, home, away, date, season, row.AvgH, row.AvgD, row.AvgA)
        rows.append({"Date": date, "Season": season, "HomeTeam": home, "AwayTeam": away, "FTR": row.FTR, **feats})
        update_state(state, home, away, date, row.FTR)

    features_df = pd.DataFrame(rows, columns=PASSTHROUGH_COLUMNS + FEATURE_COLUMNS)
    return features_df, state


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    features_df, _ = run_feature_pipeline(matches)
    return features_df


def main() -> None:
    matches = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    features = build_features(matches)

    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0, "unexpected NaNs in feature columns"

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(features)} rows x {len(FEATURE_COLUMNS)} features to {OUTPUT_PATH}")
    print("\nFeature columns:", FEATURE_COLUMNS)
    print("\nSummary stats:")
    print(features[FEATURE_COLUMNS].describe().transpose().to_string())


if __name__ == "__main__":
    main()
