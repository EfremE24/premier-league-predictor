"""FastAPI backend for the Premier League match predictor.

This module contains NO prediction logic of its own -- it only adapts
HTTP request/response shapes onto predict.load_artifacts() and
predict.predict(), the same functions the predict.py CLI uses. That way
there is exactly one place that knows how to go from raw fixture inputs to
H/D/A probabilities, and the CLI and the API can never quietly drift apart.

Three modes, not one model: market_only, team_stat_only, and combined are
three genuinely different fitted models (see train.py/predict.py module
docstrings), all loaded at startup so switching modes is just picking which
already-loaded model to call, not a retrain or a reload.

Model + feature state are loaded once at process startup (FastAPI lifespan
context, stored on app.state) rather than per-request -- both are
effectively read-only after loading (predict() calls pre_match_features(),
which does mutate a couple of dict entries on the state object as a side
effect of computing Elo-regression, but never in a way that depends on
which request triggered it -- see features.py), and re-reading three
models from disk on every request would be wasted work, which matters more
on Render's free tier where the instance is already resource-limited and
spins down between requests.

Run locally (from the project root, with the venv active):
    uvicorn api:app --app-dir src --reload --port 8000

--app-dir src puts src/ on sys.path before importing api, which is what
lets `from predict import ...` above resolve -- this project uses flat,
sibling-style imports across src/ (no src/__init__.py, matching
fetch_data.py/features.py/train.py/predict.py already), so this keeps api.py
consistent with the rest instead of being the one module that needs a
different import style.
"""
import os
from contextlib import asynccontextmanager
from datetime import date as date_type
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from fixtures import CURRENT_PL_TEAMS, get_season_fixtures
from predict import CLASS_LABELS, infer_season, load_artifacts, predict

load_dotenv()  # local dev only -- reads .env if present; a no-op when it isn't
# (Render). Production sets FOOTBALL_DATA_API_KEY as a real dashboard env var.

Mode = Literal["market_only", "team_stat_only", "combined"]


def compute_feature_importances(model, feature_columns: list[str]) -> list[dict]:
    """Feature importance comes straight off the loaded model -- an
    attribute it already has, not something recomputed -- so it can never
    drift from what the model actually learned. Random Forest exposes
    .feature_importances_ natively; Logistic Regression (market_only and
    team_stat_only, see train.py) doesn't, so this falls back to the mean
    absolute coefficient across the 3 classes, which is a standard proxy
    and specifically comparable across features here because the pipeline
    standardizes inputs first (StandardScaler) -- the coefficients are
    already on a common scale."""
    clf = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    raw = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_).mean(axis=0)
    return sorted(
        (
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in zip(feature_columns, raw)
        ),
        key=lambda row: -row["importance"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load_artifacts() exits the process if any model/state file is
    # missing -- fine here too: an API that can't predict shouldn't start.
    app.state.models, app.state.feature_state, app.state.metadata = load_artifacts()
    app.state.feature_importances = {
        mode: compute_feature_importances(app.state.models[mode], info["feature_columns"])
        for mode, info in app.state.metadata["modes"].items()
    }
    yield


app = FastAPI(
    title="Premier League Match Predictor API",
    description="Predicts home win / draw / away win probabilities for a Premier League fixture.",
    lifespan=lifespan,
)

# Narrowed once the frontend actually had a fixed deployed URL -- open to
# "*" only made sense before that existed. Keeps localhost too, since that's
# still the real local-dev loop (Vite's default port; --port 5174 is what
# this project's README/launch config actually uses).
ALLOWED_ORIGINS = [
    "https://premier-league-predictor-six.vercel.app",
    "http://localhost:5173",
    "http://localhost:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    date: date_type
    avg_h: float = Field(gt=1.0, description="Current average decimal market odds, home win")
    avg_d: float = Field(gt=1.0, description="Current average decimal market odds, draw")
    avg_a: float = Field(gt=1.0, description="Current average decimal market odds, away win")
    season: str | None = Field(default=None, description="e.g. '2026/27'; inferred from date if omitted")
    mode: Mode = Field(default="combined", description="Which trained model to use -- see GET /model-info")


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    date: date_type
    season: str
    mode: Mode
    probabilities: dict[str, float]
    predicted_outcome: str
    features: dict[str, float]


@app.post("/predict", response_model=PredictResponse)
def predict_match(req: PredictRequest, request: Request) -> PredictResponse:
    season = req.season or infer_season(pd.Timestamp(req.date))
    try:
        proba_by_class, feats = predict(
            request.app.state.models, request.app.state.feature_state, request.app.state.metadata,
            req.home_team, req.away_team, pd.Timestamp(req.date),
            req.avg_h, req.avg_d, req.avg_a, mode=req.mode, season=season,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    predicted = max(proba_by_class, key=proba_by_class.get)
    return PredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        date=req.date,
        season=season,
        mode=req.mode,
        probabilities={CLASS_LABELS[cls]: round(float(p), 4) for cls, p in proba_by_class.items()},
        predicted_outcome=CLASS_LABELS[predicted],
        features={k: round(float(v), 4) for k, v in feats.items()},
    )


@app.get("/model-info")
def model_info(request: Request) -> dict:
    """Static facts about all three trained modes -- dataset size, each
    mode's held-out test performance, and each mode's feature importances
    -- for a frontend to show its work (and let a user compare modes)
    rather than presenting predictions as an unexplained black box.
    Nothing here is computed per-request; it's the same for every caller
    until the models are retrained."""
    meta = request.app.state.metadata
    modes = {
        mode_key: {**info, "feature_importances": request.app.state.feature_importances[mode_key]}
        for mode_key, info in meta["modes"].items()
    }
    return {
        "class_weight": meta["class_weight"],
        "train_seasons": meta["train_seasons"],
        "test_seasons": meta["test_seasons"],
        "train_row_count": meta["train_row_count"],
        "test_row_count": meta["test_row_count"],
        "best_mode": meta["best_mode"],
        "modes": modes,
    }


@app.get("/fixtures")
def list_fixtures(request: Request) -> dict:
    """Premier League fixtures for the season -- both already-played
    (status FINISHED, with a score) and not-yet-played (status SCHEDULED)
    -- so the frontend can offer a full-season picker instead of a list
    that loses whichever month just happened. Each fixture is annotated
    with whether each team has training history in feature_state -- a team
    without it (e.g. freshly promoted) will still predict fine via the
    cold-start fallback in pre_match_features(), but the frontend can use
    this to show that upfront rather than as a surprise after submitting."""
    result = get_season_fixtures(os.environ.get("FOOTBALL_DATA_API_KEY"))
    known_teams = set(request.app.state.feature_state.elo.keys())
    for fixture in result["fixtures"]:
        fixture["home_team_known"] = fixture["home_team"] in known_teams
        fixture["away_team_known"] = fixture["away_team"] in known_teams
    return result


@app.get("/teams")
def list_teams() -> dict:
    """The 20 teams in this season's Premier League (fixtures.py's
    CURRENT_PL_TEAMS), not feature_state.elo.keys() -- the trained model's
    historical roster has 34 team names accumulated across 11 seasons and
    includes plenty of teams now in the Championship. This is what the
    frontend dropdown should offer as real options, not everything the
    model has ever seen."""
    return {"teams": CURRENT_PL_TEAMS}


@app.get("/")
def health_check(request: Request) -> dict:
    return {"status": "ok", "models_loaded": sorted(getattr(request.app.state, "models", {}).keys())}
