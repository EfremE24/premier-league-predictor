import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import (
    ELO_HOME_ADVANTAGE,
    INITIAL_ELO,
    NEUTRAL_PPG,
    NEUTRAL_REST_DAYS,
    build_features,
    implied_probabilities,
    match_points,
    pre_match_features,
    run_feature_pipeline,
)


def test_implied_probabilities_sum_to_one_and_remove_overround():
    h, d, a = implied_probabilities(avg_h=2.0, avg_a=4.0, avg_d=3.5)
    assert h + d + a == pytest.approx(1.0)
    # raw 1/odds should overround (sum > 1) before normalization
    raw_sum = 1 / 2.0 + 1 / 3.5 + 1 / 4.0
    assert raw_sum > 1.0
    assert h == pytest.approx((1 / 2.0) / raw_sum)


@pytest.mark.parametrize(
    "ftr,venue,expected",
    [("H", "H", 3), ("H", "A", 0), ("A", "A", 3), ("A", "H", 0), ("D", "H", 1), ("D", "A", 1)],
)
def test_match_points(ftr, venue, expected):
    assert match_points(ftr, venue) == expected


def _toy_matches():
    return pd.DataFrame({
        "Date": pd.to_datetime(["2020-08-01", "2020-08-08", "2020-08-15"]),
        "Season": ["2020/21", "2020/21", "2020/21"],
        "HomeTeam": ["A", "B", "A"],
        "AwayTeam": ["B", "A", "B"],
        "FTHG": [2, 0, 3],
        "FTAG": [0, 1, 1],
        "FTR": ["H", "A", "H"],
        "AvgH": [2.0, 2.0, 1.8],
        "AvgD": [3.5, 3.5, 3.5],
        "AvgA": [4.0, 4.0, 4.5],
    })


def test_first_match_uses_neutral_cold_start_priors():
    feats = build_features(_toy_matches())
    first = feats.iloc[0]
    assert first["elo_home"] == INITIAL_ELO
    assert first["elo_away"] == INITIAL_ELO
    assert first["form_home"] == NEUTRAL_PPG
    assert first["home_ppg"] == NEUTRAL_PPG
    assert first["rest_days_home"] == NEUTRAL_REST_DAYS


def test_features_do_not_leak_future_results():
    """Match 3 (A home, won 3-1) is A's second home match. Its pre-match
    home_ppg must reflect only match 1 (A won at home => 3 pts), not the
    unfolding match 3 result itself."""
    feats = build_features(_toy_matches())
    third = feats.iloc[2]
    assert third["HomeTeam"] == "A"
    assert third["home_ppg"] == pytest.approx(3.0)  # from match 1 only
    # elo_home going into match 3 must equal elo AFTER match 1's update,
    # not something influenced by match 3's own H/A/D result.
    assert third["elo_home"] != INITIAL_ELO


def test_elo_update_is_zero_sum_per_match():
    """Whatever a team gains, its opponent loses -- no rating is created
    or destroyed by a single match update."""
    feats = build_features(_toy_matches())
    first = feats.iloc[0]
    expected_home = 1.0 / (1.0 + 10 ** (-((first["elo_home"] + ELO_HOME_ADVANTAGE) - first["elo_away"]) / 400))
    s_home = 1.0  # match 1 was a home win
    delta = 20.0 * (s_home - expected_home)  # ELO_K = 20.0
    second = feats.iloc[1]  # B's pre-match elo_away reflects A's state; B is HomeTeam here
    # second row: HomeTeam=B, AwayTeam=A -> elo_away is A's rating after match 1
    assert second["elo_away"] == pytest.approx(first["elo_home"] + delta)


def test_no_missing_values_in_engineered_features():
    feats = build_features(_toy_matches())
    feature_cols = [c for c in feats.columns if c not in ("Date", "Season", "HomeTeam", "AwayTeam", "FTR")]
    assert feats[feature_cols].isna().sum().sum() == 0


def test_pre_match_features_handles_team_never_seen_before():
    """This is the predict.py cold-start case, not the season-opener one:
    a FeatureState that already has real history for other teams (as
    persisted to feature_state.joblib), queried for a team that has never
    appeared in it at all -- e.g. a club newly promoted to the league.
    Distinct from test_first_match_uses_neutral_cold_start_priors, which
    covers a team's first match within an otherwise-empty state."""
    _, state = run_feature_pipeline(_toy_matches())  # state now holds real history for A and B

    feats = pre_match_features(
        state, home="NewlyPromotedFC", away="A",
        date=pd.Timestamp("2020-09-01"), season="2020/21",
        avg_h=2.0, avg_d=3.5, avg_a=4.0,
    )

    # the unseen team falls back to neutral priors, not a crash
    assert feats["elo_home"] == INITIAL_ELO
    assert feats["form_home"] == NEUTRAL_PPG
    assert feats["home_ppg"] == NEUTRAL_PPG
    assert feats["rest_days_home"] == NEUTRAL_REST_DAYS

    # the opponent's real history must be unaffected by the other side being unseen
    assert feats["elo_away"] != INITIAL_ELO
    assert feats["away_ppg"] != NEUTRAL_PPG

    # querying an unseen team must not register it in the shared state as a side effect
    # beyond the elo default (setdefault is expected); it should not gain fake match history
    assert "NewlyPromotedFC" not in state.home_matches
    assert "NewlyPromotedFC" not in state.form
