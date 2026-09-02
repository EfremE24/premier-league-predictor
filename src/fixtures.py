"""Upcoming Premier League fixtures from football-data.org.

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
MAX_FIXTURES = 20  # roughly the next 2 matchdays; the season has ~380 total

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

_cache: dict = {"fixtures": None, "fetched_at": 0.0, "error": None}


def normalize_team_name(team: dict) -> str:
    return TEAM_NAME_MAP.get(team["name"], team["shortName"])


def _fetch_from_api(api_key: str) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/competitions/{COMPETITION}/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": "SCHEDULED"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    matches = resp.json()["matches"]
    matches.sort(key=lambda m: m["utcDate"])

    fixtures = []
    for m in matches[:MAX_FIXTURES]:
        fixtures.append({
            "id": m["id"],
            "kickoff_utc": m["utcDate"],
            "date": m["utcDate"][:10],
            "matchday": m["matchday"],
            "home_team": normalize_team_name(m["homeTeam"]),
            "away_team": normalize_team_name(m["awayTeam"]),
        })
    return fixtures


def get_upcoming_fixtures(api_key: str | None, force_refresh: bool = False) -> dict:
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
