"""Premier League fixtures (scheduled and already-played) from football-data.org.

Separate from features.py/predict.py on purpose: this module only knows
how to talk to football-data.org and normalize its team names to match
ours -- it has no idea what a model or a FeatureState is. api.py composes
this with feature_state.elo (to flag which fixture teams have training
history) when building the /fixtures response.

Caching: a single in-process cache with a TTL, refreshed lazily on request
rather than on a background schedule. football-data.org's free tier is
rate-limited (10 req/min) and this project's own choice, documented to the
user, is to refresh every few hours rather than per-request -- fixtures
don't change minute to minute, and Render's free tier spins the process
down between periods of inactivity anyway, which would kill any background
scheduler. A cache that's just "stale after N hours, refetch on next
request" survives that restart pattern correctly since it has no state to
lose that a fresh fetch doesn't reconstruct.

Team-name mapping: football-data.org's team names (full legal name, e.g.
"Nottingham Forest FC") don't match football-data.co.uk's naming, which
is what our historical data and feature_state use ("Nott'm Forest"). This
map was built by hand against a live API response for the current 20
Premier League teams (see conversation/commit history), not derived
programmatically -- there's no reliable automatic transform between "Man
United" and "Manchester United FC". A team missing from this map (e.g.
after a season's promotion/relegation changes the league) falls back to
football-data.org's shortName rather than raising, since a fixture with an
unmapped name is still useful to show -- worst case it doesn't match an
existing feature_state entry and gets treated as a cold-start team, which
is a real, already-handled outcome, not a crash.
"""
import time
from datetime import datetime, timezone

import requests

BASE_URL = "https://api.football-data.org/v4"
COMPETITION = "PL"
REQUEST_TIMEOUT = 10
CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours -- see module docstring
# The full season (~380 matches, one competition so nowhere near
# football-data.org's own response-size limits) is returned -- no artificial
# cap. Both not-yet-played and already-played matches are fetched in one
# call (status=SCHEDULED,FINISHED, comma-separated -- verified this works
# as a single request rather than needing two round trips). Originally this
# only fetched SCHEDULED, but that meant a match dropped off the list the
# moment it kicked off, so a user could never see e.g. September's fixtures
# again once September had been played -- the whole month just vanished.
# Already-played matches are still returned here; api.py/the frontend
# decide what to do with them (show the score, don't offer them as
# something to predict -- feature_state.joblib is a training-time snapshot,
# see MODEL_NOTES.md, so "predicting" an already-played match wouldn't
# reflect the team's actual form at that point in the season anyway).
# football-data.org's own per-match status is more granular than our two
# buckets here (TIMED/SCHEDULED both mean "hasn't kicked off"); normalized
# to just "SCHEDULED" or "FINISHED" in what this module returns.

# football-data.org full team name -> our (football-data.co.uk-style) name.
# Hand-verified against a live /v4/competitions/PL/teams response.
TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Chelsea FC": "Chelsea",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "Hull City AFC": "Hull",
    "Leeds United FC": "Leeds",
    "Ipswich Town FC": "Ipswich",
    "Nottingham Forest FC": "Nott'm Forest",
    "Crystal Palace FC": "Crystal Palace",
    "Brighton & Hove Albion FC": "Brighton",
    "Brentford FC": "Brentford",
    "AFC Bournemouth": "Bournemouth",
    "Coventry City FC": "Coventry",  # not in our historical data -- cold start, not a bug
}

# The 20 teams actually in this season's Premier League -- distinct from
# feature_state.elo.keys() in the trained model, which has 34 team names
# accumulated across 11 seasons of history and includes plenty of teams
# now in the Championship (Cardiff, Watford, etc). api.py's GET /teams uses
# this list, not the model's historical roster, so the frontend dropdown
# only ever offers real current-season matchups. A team here with no
# training history (e.g. Coventry above) still predicts fine via the
# cold-start fallback in pre_match_features() -- this list is about which
# teams are real options to pick, not which ones the model has seen before.
CURRENT_PL_TEAMS = sorted(set(TEAM_NAME_MAP.values()))

_cache: dict = {"fixtures": None, "fetched_at": 0.0, "error": None}


def normalize_team_name(team: dict) -> str:
    return TEAM_NAME_MAP.get(team["name"], team["shortName"])


def _fetch_from_api(api_key: str) -> list[dict]:
    # No status filter, deliberately -- status=SCHEDULED as a query param
    # turned out to be a literal match against football-data.org's status
    # enum, and most not-yet-played matches actually carry status TIMED,
    # not SCHEDULED (SCHEDULED,FINISHED as a filter silently dropped every
    # TIMED match, which is most of a live season -- caught by checking the
    # raw API response directly, not by guessing). One competition's season
    # is small enough (380 matches) to just fetch everything and classify
    # client-side below, which also means IN_PLAY/PAUSED/POSTPONED/etc.
    # degrade to "not FINISHED" instead of silently vanishing again if
    # football-data.org's status enum has more values than we've seen.
    resp = requests.get(
        f"{BASE_URL}/competitions/{COMPETITION}/matches",
        headers={"X-Auth-Token": api_key},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    matches = resp.json()["matches"]
    matches.sort(key=lambda m: m["utcDate"])

    fixtures = []
    for m in matches:
        is_finished = m["status"] == "FINISHED"
        full_time = m["score"]["fullTime"]
        fixtures.append({
            "id": m["id"],
            "kickoff_utc": m["utcDate"],
            "date": m["utcDate"][:10],
            "matchday": m["matchday"],
            "home_team": normalize_team_name(m["homeTeam"]),
            "away_team": normalize_team_name(m["awayTeam"]),
            "status": "FINISHED" if is_finished else "SCHEDULED",
            "home_score": full_time["home"] if is_finished else None,
            "away_score": full_time["away"] if is_finished else None,
        })
    return fixtures


def get_season_fixtures(api_key: str | None, force_refresh: bool = False) -> dict:
    """Returns {"fixtures": [...], "fetched_at": iso str or None, "error": str or None}.
    Never raises -- fixtures are a convenience feature, not load-bearing;
    a football-data.org outage or missing key should degrade to an empty
    or stale list, not take the whole API down with it."""
    if api_key is None:
        return {"fixtures": [], "fetched_at": None, "error": "FOOTBALL_DATA_API_KEY not configured"}

    is_stale = (time.time() - _cache["fetched_at"]) > CACHE_TTL_SECONDS
    if force_refresh or _cache["fixtures"] is None or is_stale:
        try:
            fixtures = _fetch_from_api(api_key)
            _cache["fixtures"] = fixtures
            _cache["fetched_at"] = time.time()
            _cache["error"] = None
        except (requests.RequestException, KeyError, ValueError) as exc:
            _cache["error"] = str(exc)
            # keep serving whatever's already cached, even if stale, rather
            # than returning nothing just because a refresh attempt failed

    fetched_at = (
        datetime.fromtimestamp(_cache["fetched_at"], tz=timezone.utc).isoformat()
        if _cache["fetched_at"] else None
    )
    return {"fixtures": _cache["fixtures"] or [], "fetched_at": fetched_at, "error": _cache["error"]}
