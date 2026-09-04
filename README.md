# Full Time — Premier League Match Predictor

Predicts full-time result (home win / draw / away win) for English Premier
League matches. A resume/portfolio data science project — full pipeline
from raw historical data through a deployed, live-clickable web app, not
just a notebook.

**Live demo:** backend and frontend deployment in progress — links go here
once both are up.

## The headline result, up front

51.1% accuracy sounds unremarkable in isolation. It isn't, once you see
what it's actually being compared against:

| Approach | Accuracy |
|---|---|
| Random guess (3 classes) | 33.3% |
| Always predict "home win" (naive baseline) | 44.3% |
| **This model** | **51.1%** |
| Market odds alone (professional bookmakers) | 51.8% |

Landing within 0.7 points of the market itself is a strong, credible
result — betting markets aggregate huge amounts of information (injuries,
team news, sharp money) and are notoriously hard to beat. Football also
has a real accuracy ceiling regardless of model quality: matches involve
enough randomness (a post hit, a bad call) that even professional models
rarely clear the mid-50s on 3-way outcomes.

Full reasoning, every ablation, and the numbers behind them are in
[`MODEL_NOTES.md`](MODEL_NOTES.md) — written for interview prep, not just
as a lab notebook.

## What it actually found (not just what it predicts)

Two findings drove real design decisions, not just observations:

1. **`class_weight='balanced'` made the model overconfident about draws.**
   The intuitive fix for a 24%-minority class (reweight the loss) actively
   hurt calibration — with `balanced`, the model pushed ~550 test matches
   into a "30–50% chance of a draw" bucket where a draw actually happened
   only ~27% of the time. Log loss confirmed it: 1.0084 with `balanced`
   vs. **0.9981** without. Switched to `class_weight=None` for the final
   model.
2. **Team-strength features (Elo, form, rest, home/away PPG) add almost
   nothing once market odds are in the model.** A market-only model
   (just the bookmakers' implied probabilities) scored within 0.0008 log
   loss of the full 15-feature model. The market is doing nearly all the
   work — an honest, unglamorous result, and the actual headline finding
   of this project rather than something to spin.

The live site doesn't just state finding #2 — it lets you check it.
There's a **prediction mode switcher** (Market-aware / Team-stat /
Combined), and each mode is a genuinely different trained model, not the
same model with inputs zeroed out. Switch to Team-stat on any matchup and
watch the market-derived features gray out in the "what the model
actually saw" panel, the prediction shift, and the live metrics confirm
it's the weaker mode — see [`MODEL_NOTES.md`](MODEL_NOTES.md#4-the-site-lets-you-switch-between-all-three-models-not-just-read-about-them)
for the full per-mode comparison (accuracy, log loss, ROC AUC, Brier score).

## Project layout

```
data/
  raw/            one CSV per season, as downloaded (gitignored)
  processed/      combined matches.csv + features.csv (gitignored)
models/
  model_market_only.joblib     fitted model, imp_prob_h/d/a only
  model_team_stat_only.joblib  fitted model, Elo/form/rest/PPG only
  model_combined.joblib        fitted model, all 15 features (default mode)
  feature_state.joblib         per-team Elo/form/rest/PPG state, shared across modes
  model_metadata.json          per-mode hyperparams, split, confirmed test metrics
src/
  fetch_data.py   download + combine 11 PL seasons from football-data.co.uk
  features.py     15 engineered features via a leak-free chronological pass
  fixtures.py     live upcoming fixtures from football-data.org
  train.py        time-based split, class-weight + feature-set ablations,
                   persists one model per prediction mode
  predict.py      CLI: raw fixture inputs -> H/D/A probabilities, any mode
  api.py          FastAPI wrapper around predict.py (no duplicated logic)
frontend/         Vite + React app: prediction mode switcher, fixture picker,
                  live prediction, model transparency (feature values used,
                  per-mode feature importances)
tests/
MODEL_NOTES.md    full ablation write-up, in plain language, for interviews
```

Model artifacts (`models/*.joblib`) are committed, unlike `data/` — Render
has no access to football-data.co.uk at deploy time, so the API needs a
pre-trained model already in the repo to serve predictions at all.

## How it works

1. **`fetch_data.py`** pulls 2015/16–2025/26 from football-data.co.uk,
   normalizing odds columns across a mid-history naming change.
2. **`features.py`** builds 15 features (Elo rating, 5-match form,
   home/away-specific points-per-game, rest days, market-implied
   probability) via a single chronological pass — not a pandas
   rolling/groupby, since getting a leak-free window right across a team's
   dual home/away role is easy to get subtly wrong. Verified leak-free by
   tests that check a feature for match *N* only reflects matches 1..N-1.
3. **`train.py`** splits on time (train 2015/16–2023/24, test
   2024/25–2025/26 — never random, since features are inherently
   sequential), runs the class-weight and feature-set ablations above, and
   selects by log loss rather than accuracy: log loss penalizes confident
   wrong answers, which is exactly the failure mode that matters most for
   draws.
4. **`predict.py`** / **`api.py`** take a fixture (teams, date, odds) and
   reproduce the same 15 features for inference, including a tested
   cold-start fallback for teams with no training history (e.g. newly
   promoted clubs) — neutral priors instead of an error.
5. **`fixtures.py`** pulls live upcoming fixtures from football-data.org
   so the frontend can offer a picker instead of manual entry, cached on a
   6-hour TTL to stay under the free-tier rate limit.

## Setup

Backend:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add a free football-data.org API key for live fixtures
uvicorn api:app --app-dir src --reload --port 8000
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

Full pipeline from scratch (only needed to retrain, not to run the app —
committed model artifacts already work out of the box):
```bash
python src/fetch_data.py
python src/features.py
python src/train.py
```

## Tech stack

Python (pandas, scikit-learn, FastAPI) for the pipeline and API; React +
Vite for the frontend; football-data.co.uk (historical results/odds) and
football-data.org (live fixtures) as data sources; deployed on Render
(backend) and Vercel (frontend).

## Known limitations

Documented rather than hidden — see [`MODEL_NOTES.md`](MODEL_NOTES.md) for
full detail on each:

- Newly promoted teams start at a neutral Elo, likely overstating their
  strength until they've played enough matches to move it.
- **The model's team-strength features don't update as the current season
  is played** — `feature_state.joblib` is a snapshot from the last
  training run; nothing yet feeds finished-match results back into it.
  This is the most significant open item, not a deliberate tradeoff.
- Live odds (vs. the current "look it up yourself" link) was deliberately
  not built — real-time odds APIs add cost or fragility disproportionate
  to the convenience of typing in three numbers.
