"""API-level tests: request/response shapes, validation, mode routing.

Complements tests/test_features.py (feature-engineering correctness) with
service-layer coverage. These hit the real FastAPI app through TestClient,
which runs the actual lifespan startup -- so they load the real committed
model artifacts from models/ (small, local files, no network), exercising
the real startup path, not just individual functions in isolation.

GET /fixtures is the one endpoint with a real external dependency
(football-data.org). api.get_season_fixtures is patched out in the two
tests that touch it, so this file never makes a network call and never
needs a real FOOTBALL_DATA_API_KEY -- deterministic and fast in CI.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from api import app  # noqa: E402

MODES = {"market_only", "team_stat_only", "combined"}


@pytest.fixture
def client():
    with TestClient(app) as c:  # runs the real lifespan startup (loads models)
        yield c


def test_health_check(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert set(body["models_loaded"]) == MODES


def test_teams_returns_current_pl_roster_not_historical_one(client):
    res = client.get("/teams")
    assert res.status_code == 200
    teams = res.json()["teams"]
    assert len(teams) == 20
    assert "Arsenal" in teams
    # Cardiff is in the model's 11-season historical roster but not the
    # current Premier League -- GET /teams should only ever offer current
    # teams (see api.py's list_teams docstring for why).
    assert "Cardiff" not in teams


def test_model_info_covers_all_three_modes(client):
    res = client.get("/model-info")
    assert res.status_code == 200
    body = res.json()
    assert set(body["modes"].keys()) == MODES
    assert body["best_mode"] in body["modes"]
    for mode_info in body["modes"].values():
        assert mode_info["feature_importances"]
        for metric in ("test_log_loss", "test_accuracy", "test_roc_auc", "test_brier_score"):
            assert isinstance(mode_info[metric], float)


@pytest.mark.parametrize("mode", sorted(MODES))
def test_predict_each_mode_returns_valid_probabilities(client, mode):
    res = client.post("/predict", json={
        "home_team": "Arsenal", "away_team": "Liverpool", "date": "2026-09-20",
        "avg_h": 2.45, "avg_d": 3.60, "avg_a": 2.85, "mode": mode,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == mode
    probs = body["probabilities"]
    assert set(probs.keys()) == {"Home win", "Draw", "Away win"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=2e-3)
    assert body["predicted_outcome"] in probs


def test_predict_defaults_to_combined_mode_when_omitted(client):
    res = client.post("/predict", json={
        "home_team": "Arsenal", "away_team": "Liverpool", "date": "2026-09-20",
        "avg_h": 2.45, "avg_d": 3.60, "avg_a": 2.85,
    })
    assert res.status_code == 200
    assert res.json()["mode"] == "combined"


def test_predict_rejects_invalid_mode(client):
    res = client.post("/predict", json={
        "home_team": "Arsenal", "away_team": "Liverpool", "date": "2026-09-20",
        "avg_h": 2.45, "avg_d": 3.60, "avg_a": 2.85, "mode": "not_a_real_mode",
    })
    assert res.status_code == 422


def test_predict_rejects_invalid_odds(client):
    res = client.post("/predict", json={
        "home_team": "Arsenal", "away_team": "Liverpool", "date": "2026-09-20",
        "avg_h": 0.5, "avg_d": 3.60, "avg_a": 2.85,
    })
    assert res.status_code == 422


def test_predict_cold_start_team_does_not_error(client):
    """Coventry has zero rows in feature_state (see fixtures.py's
    TEAM_NAME_MAP comment) -- this is the real cold-start path, exercised
    here through the actual HTTP endpoint rather than simulated, alongside
    the features.py-level coverage in test_features.py."""
    res = client.post("/predict", json={
        "home_team": "Coventry", "away_team": "Arsenal", "date": "2026-09-20",
        "avg_h": 5.50, "avg_d": 4.20, "avg_a": 1.55,
    })
    assert res.status_code == 200
    assert sum(res.json()["probabilities"].values()) == pytest.approx(1.0, abs=2e-3)


def test_fixtures_returns_mocked_data_with_known_flags(client):
    fake_result = {
        "fixtures": [{
            "id": 1, "kickoff_utc": "2026-09-20T15:00:00Z", "date": "2026-09-20",
            "matchday": 4, "home_team": "Arsenal", "away_team": "Chelsea",
            "status": "SCHEDULED", "home_score": None, "away_score": None,
        }],
        "fetched_at": "2026-09-20T00:00:00+00:00", "error": None,
    }
    with patch("api.get_season_fixtures", return_value=fake_result):
        res = client.get("/fixtures")
    assert res.status_code == 200
    fixture = res.json()["fixtures"][0]
    assert fixture["home_team_known"] is True
    assert fixture["away_team_known"] is True


def test_fixtures_degrades_gracefully_without_api_key(client):
    """Mirrors what fixtures.get_season_fixtures() itself returns when
    FOOTBALL_DATA_API_KEY isn't set -- the endpoint should pass that
    through as a 200 with an empty list and an error message, not a 500."""
    missing_key_result = {"fixtures": [], "fetched_at": None, "error": "FOOTBALL_DATA_API_KEY not configured"}
    with patch("api.get_season_fixtures", return_value=missing_key_result):
        res = client.get("/fixtures")
    assert res.status_code == 200
    body = res.json()
    assert body["fixtures"] == []
    assert body["error"] is not None
