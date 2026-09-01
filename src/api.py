"""FastAPI backend for the Premier League match predictor.

This module contains NO prediction logic of its own -- it only adapts
HTTP request/response shapes onto predict.load_artifacts() and
predict.predict(), the same functions the predict.py CLI uses. That way
there is exactly one place that knows how to go from raw fixture inputs to
H/D/A probabilities, and the CLI and the API can never quietly drift apart.

Model + feature state are loaded once at process startup (FastAPI lifespan
context, stored on app.state) rather than per-request -- both are
effectively read-only after loading (predict() calls pre_match_features(),
which does mutate a couple of dict entries on the state object as a side
effect of computing Elo-regression, but never in a way that depends on
which request triggered it -- see features.py), and re-reading a ~3MB
RandomForest from disk on every request would be wasted work, which matters
more on Render's free tier where the instance is already resource-limited
and spins down between requests.

Run locally (from the project root, with the venv active):
    uvicorn api:app --app-dir src --reload --port 8000

--app-dir src puts src/ on sys.path before importing api, which is what
lets `from predict import ...` above resolve -- this project uses flat,
sibling-style imports across src/ (no src/__init__.py, matching
fetch_data.py/features.py/train.py/predict.py already), so this keeps api.py
consistent with the rest instead of being the one module that needs a
different import style.
"""
from contextlib import asynccontextmanager
from datetime import date as date_type

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict import CLASS_LABELS, infer_season, load_artifacts, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load_artifacts() exits the process if the model/state files are
    # missing -- fine here too: an API that can't predict shouldn't start.
    app.state.model, app.state.feature_state = load_artifacts()
    yield


app = FastAPI(
    title="Premier League Match Predictor API",
    description="Predicts home win / draw / away win probabilities for a Premier League fixture.",
    lifespan=lifespan,
)

# No frontend deployed yet, so there's no real origin to restrict to. Open
# for local development; narrow this to the deployed frontend's origin
# once one exists, before this goes further than local testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    date: date_type
    season: str
    probabilities: dict[str, float]
    predicted_outcome: str


@app.post("/predict", response_model=PredictResponse)
def predict_match(req: PredictRequest, request: Request) -> PredictResponse:
    season = req.season or infer_season(pd.Timestamp(req.date))
    try:
        proba_by_class = predict(
            request.app.state.model, request.app.state.feature_state,
            req.home_team, req.away_team, pd.Timestamp(req.date),
            req.avg_h, req.avg_d, req.avg_a, season=season,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    predicted = max(proba_by_class, key=proba_by_class.get)
    return PredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        date=req.date,
        season=season,
        probabilities={CLASS_LABELS[cls]: round(float(p), 4) for cls, p in proba_by_class.items()},
        predicted_outcome=CLASS_LABELS[predicted],
    )


@app.get("/teams")
def list_teams(request: Request) -> dict:
    """All team names the persisted feature_state has history for -- for a
    frontend to populate dropdowns without hardcoding a team list that
    would drift out of date as data/features.py get re-run."""
    return {"teams": sorted(request.app.state.feature_state.elo.keys())}


@app.get("/")
def health_check(request: Request) -> dict:
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model")}
