"""Tonight's games from ESPN (tip-off, spread, total, injury report).

ESPN event ids are the same game ids the hoopR box scores use. The injury
report's "Out" and "Doubtful" players are treated as not playing; everyone
else on a team's recent rotation is treated as available.
"""

import json
import re
import urllib.request

import numpy as np
import pandas as pd

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
OUT_STATUSES = {"out", "doubtful"}
SEASON_TYPES = {2: "regular", 3: "playoffs", 5: "play-in"}  # ESPN season types
SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "season_type",
    "tip_utc",
    "home",
    "away",
    "home_score",
    "away_score",
    "completed",
    "home_spread",
    "total",
]


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "nba-prop-model"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def home_spread_from(details, home, away):
    """'DET -1.5' with DET at home -> -1.5; with DET away -> +1.5."""
    if not details:
        return np.nan
    text = str(details).strip().upper()
    if text in {"EVEN", "PK", "PICK"}:
        return 0.0
    match = re.match(r"([A-Z]+)\s+([-+]?\d+(?:\.\d+)?)$", text)
    if not match:
        return np.nan
    team, line = match.group(1), float(match.group(2))
    if team == home:
        return line
    if team == away:
        return -line
    return np.nan


def parse_scoreboard(payload, season):
    rows = []
    for event in payload.get("events", []):
        kind = SEASON_TYPES.get((event.get("season") or {}).get("type", 2))
        if kind is None:
            continue  # preseason and exhibitions
        competition = event["competitions"][0]
        teams = {
            c["homeAway"]: c["team"]["abbreviation"] for c in competition["competitors"]
        }
        odds = (competition.get("odds") or [{}])[0]
        rows.append(
            {
                "game_id": int(event["id"]),
                "season": season,
                "season_type": kind,
                "tip_utc": pd.Timestamp(event["date"]).tz_convert("UTC"),
                "home": teams.get("home"),
                "away": teams.get("away"),
                "home_score": np.nan,
                "away_score": np.nan,
                "completed": bool(competition["status"]["type"].get("completed")),
                "home_spread": home_spread_from(
                    odds.get("details"), teams.get("home"), teams.get("away")
                ),
                "total": float(odds["overUnder"]) if odds.get("overUnder") else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def scoreboard(date, season):
    """Games on a US-Eastern date ('YYYYMMDD') with tip-off, spread and total."""
    return parse_scoreboard(get_json(f"{ESPN}/scoreboard?dates={date}"), season)


def _athlete_id(athlete):
    for href in [athlete.get("headshot", {}).get("href", "")] + [
        link.get("href", "") for link in athlete.get("links", [])
    ]:
        match = re.search(r"/full/(\d+)\.png|/id/(\d+)", href or "")
        if match:
            return int(match.group(1) or match.group(2))
    return None


def parse_injuries(payload):
    rows = []
    for team in payload.get("injuries", []):
        for item in team.get("injuries", []):
            athlete = item.get("athlete", {})
            rows.append(
                {
                    "team": athlete.get("team", {}).get("abbreviation"),
                    "player_id": _athlete_id(athlete),
                    "player": athlete.get("displayName"),
                    "status": item.get("status"),
                    "comment": item.get("shortComment"),
                    "updated": item.get("date"),
                }
            )
    frame = pd.DataFrame(
        rows, columns=["team", "player_id", "player", "status", "comment", "updated"]
    )
    frame["out"] = frame.status.str.lower().isin(OUT_STATUSES)
    return frame


def injuries():
    return parse_injuries(get_json(f"{ESPN}/injuries"))


def upcoming_rows(players, schedule, report, listed, lookback=3):
    """Player rows for games not yet played, in the box-score format.

    For each team: everyone who played in its last ``lookback`` games this
    season, plus players Kalshi lists for the game (``listed``: game_id, team,
    player_id). Players the injury report has Out or Doubtful get
    did_not_play; everyone else is expected to play.
    """
    played = players[(~players.did_not_play) & (players.minutes > 0)]
    latest = players.sort_values("tip_utc").drop_duplicates("player_id", keep="last")
    names = latest.set_index("player_id")[["player", "position"]]
    out_ids = set(report.loc[report.out, "player_id"].dropna().astype("int64"))
    rows = []
    for game in schedule.itertuples():
        for team, opp, home in [
            (game.home, game.away, True),
            (game.away, game.home, False),
        ]:
            recent = played[(played.team == team) & (played.season == game.season)]
            recent_games = recent.drop_duplicates("game_id").nlargest(
                lookback, "tip_utc"
            )
            ids = set(recent[recent.game_id.isin(recent_games.game_id)].player_id)
            mine = listed[(listed.game_id == game.game_id) & (listed.team == team)]
            ids |= set(mine.player_id.dropna().astype("int64"))
            for pid in sorted(ids):
                if pid not in names.index:
                    continue
                out = pid in out_ids
                rows.append(
                    {
                        "game_id": game.game_id,
                        "season": game.season,
                        "season_type": game.season_type,
                        "tip_utc": game.tip_utc,
                        "game_date": game.tip_utc.tz_convert("America/New_York").date(),
                        "team": team,
                        "opp": opp,
                        "home": home,
                        "player_id": pid,
                        "player": names.at[pid, "player"],
                        "position": names.at[pid, "position"],
                        "starter": False,
                        "did_not_play": out,
                        "reason": "injury report" if out else None,
                        "expected": not out,
                    }
                )
    return pd.DataFrame(rows)
