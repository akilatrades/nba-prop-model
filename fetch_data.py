"""Download and trim the public NBA data the model uses.

Source: the sportsdataverse hoopR data repository (ESPN box scores and
schedules, plus consensus closing lines from The Odds API).

Writes (not committed; rerun to refresh):
    data/player_games.parquet  one row per player per game, including players
                               listed as not playing, regular season, play-in
                               and playoffs
    data/games.parquet         schedule with tip-off times, scores and closing
                               spread and total, including upcoming games
    data/retrieval.json        what was downloaded, when, and checksums

Run from the repository root: python fetch_data.py [--local PATH]
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

FIRST_SEASON, CURRENT_SEASON = 2018, 2027  # seasons are named by the year they end
DATA = Path("data")
REMOTE = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main"
PLAYER_PATH = "nba/player_box/parquet/player_box_{season}.parquet"
SCHEDULE_PATH = "nba/schedules/parquet/nba_schedule_{season}.parquet"
LINES_PATH = "nba/betting_lines/closing_lines_odds_api.parquet"
SEASON_TYPES = {2: "regular", 3: "playoffs", 5: "play-in"}
STATS = {
    "minutes": "minutes",
    "points": "points",
    "rebounds": "rebounds",
    "offensive_rebounds": "oreb",
    "defensive_rebounds": "dreb",
    "assists": "assists",
    "three_point_field_goals_made": "threes",
    "three_point_field_goals_attempted": "fg3a",
    "field_goals_made": "fgm",
    "field_goals_attempted": "fga",
    "free_throws_made": "ftm",
    "free_throws_attempted": "fta",
    "turnovers": "tov",
    "steals": "steals",
    "blocks": "blocks",
    "fouls": "fouls",
}
# The 30 teams, in ESPN's abbreviations; drops All-Star and exhibition teams.
NBA_TEAMS = {
    "ATL",
    "BKN",
    "BOS",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GS",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NO",
    "NY",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SA",
    "SAC",
    "TOR",
    "UTAH",
    "WSH",
}
POSITION = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}


def read(root, path):
    return pd.read_parquet(f"{root}/{path}")


def player_games(root, seasons):
    frames = []
    for season in seasons:
        try:
            box = read(root, PLAYER_PATH.format(season=season))
        except (FileNotFoundError, OSError):
            continue  # a season with no games yet
        box = box[
            box.season_type.isin(SEASON_TYPES)
            & box.athlete_id.notna()
            & box.team_abbreviation.isin(NBA_TEAMS)
            & box.opponent_team_abbreviation.isin(NBA_TEAMS)
        ]
        out = pd.DataFrame(
            {
                "game_id": box.game_id.astype("int64"),
                "season": box.season.astype("int16"),
                "season_type": box.season_type.map(SEASON_TYPES),
                "tip_utc": pd.to_datetime(box.game_date_time, utc=True),
                "game_date": pd.to_datetime(box.game_date).dt.date,
                "team": box.team_abbreviation,
                "opp": box.opponent_team_abbreviation,
                "home": box.home_away.eq("home"),
                "player_id": box.athlete_id.astype("int64"),
                "player": box.athlete_display_name,
                "position": box.athlete_position_abbreviation.map(POSITION),
                "starter": box.starter.fillna(False).astype(bool),
                "did_not_play": box.did_not_play.fillna(False).astype(bool),
                "reason": box.reason,
                "team_score": box.team_score,
                "opp_score": box.opponent_team_score,
            }
        )
        for source, name in STATS.items():
            out[name] = box[source].astype("float32")
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def games(root, seasons):
    frames = []
    for season in seasons:
        try:
            sched = read(root, SCHEDULE_PATH.format(season=season))
        except (FileNotFoundError, OSError):
            continue
        sched = sched[
            sched.season_type.isin(SEASON_TYPES)
            & sched.home_abbreviation.isin(NBA_TEAMS)
            & sched.away_abbreviation.isin(NBA_TEAMS)
        ]
        frames.append(
            pd.DataFrame(
                {
                    "game_id": sched.game_id.astype("int64"),
                    "season": sched.season.astype("int16"),
                    "season_type": sched.season_type.map(SEASON_TYPES),
                    "tip_utc": pd.to_datetime(sched.date, utc=True),
                    "home": sched.home_abbreviation,
                    "away": sched.away_abbreviation,
                    "home_score": pd.to_numeric(sched.home_score, errors="coerce"),
                    "away_score": pd.to_numeric(sched.away_score, errors="coerce"),
                    "completed": sched.status_type_completed.astype(bool),
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("game_id")
    lines = read(root, LINES_PATH)[["game_id", "home_point", "over_under"]]
    lines = lines.rename(columns={"home_point": "home_spread", "over_under": "total"})
    return frame.merge(lines, on="game_id", how="left")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="path to a local copy of hoopR-nba-data")
    args = parser.parse_args()
    root = args.local or REMOTE
    seasons = range(FIRST_SEASON, CURRENT_SEASON + 1)
    DATA.mkdir(exist_ok=True)
    players = player_games(root, seasons)
    schedule = games(root, seasons)
    players.to_parquet(DATA / "player_games.parquet", index=False)
    schedule.to_parquet(DATA / "games.parquet", index=False)
    played = schedule[schedule.completed]
    retrieval = {
        "retrieved": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": REMOTE if not args.local else "local copy of " + REMOTE,
        "seasons": [FIRST_SEASON, CURRENT_SEASON],
        "last_completed_game": str(played.tip_utc.max()),
        "rows": {"player_games": len(players), "games": len(schedule)},
        "games_with_lines": int(schedule.total.notna().sum()),
        "sha256": {
            "player_games.parquet": sha256(DATA / "player_games.parquet"),
            "games.parquet": sha256(DATA / "games.parquet"),
        },
    }
    (DATA / "retrieval.json").write_text(json.dumps(retrieval, indent=2) + "\n")
    print(json.dumps(retrieval, indent=2))


if __name__ == "__main__":
    main()
