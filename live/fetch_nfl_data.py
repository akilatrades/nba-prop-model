"""Download and trim the public nflverse data the model uses.

Writes:
    data/games.csv            regular-season schedule, results and lines,
                              including upcoming games of the current season
    data/player_week.csv.gz   weekly box scores for QB, RB, FB, WR and TE
    data/injuries.csv.gz      weekly injury reports (game status) for those positions

Run from the repository root: python -m live.fetch_nfl_data
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

FIRST_SEASON, CURRENT_SEASON = 2012, 2026
# Fixed gzip timestamp: unchanged data produces an identical file.
GZIP = {"method": "gzip", "mtime": 0}
DATA = Path("data/nfl")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PLAYER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)
GAME_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "gametime",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "result",
    "total",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "total_line",
    "away_qb_id",
    "home_qb_id",
    "away_qb_name",
    "home_qb_name",
]
INJURY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "injuries/injuries_{season}.csv"
)
INJURY_COLUMNS = [
    "season",
    "week",
    "team",
    "gsis_id",
    "position",
    "full_name",
    "report_status",
]
PLAYER_COLUMNS = [
    "player_id",
    "player_display_name",
    "position",
    "season",
    "week",
    "game_id",
    "team",
    "opponent_team",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    games = pd.read_csv(GAMES_URL)
    games = games[
        (games.game_type == "REG") & games.season.between(FIRST_SEASON, CURRENT_SEASON)
    ]
    games[GAME_COLUMNS].to_csv(DATA / "games.csv", index=False)

    frames = []
    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        weekly = pd.read_csv(PLAYER_URL.format(season=season), low_memory=False)
        weekly = weekly[
            (weekly.season_type == "REG")
            & weekly.position.isin(["QB", "RB", "FB", "WR", "TE"])
        ]
        frames.append(weekly[PLAYER_COLUMNS])
    players = pd.concat(frames, ignore_index=True)
    players.to_csv(DATA / "player_week.csv.gz", index=False, compression=GZIP)

    injuries = []
    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        report = pd.read_csv(INJURY_URL.format(season=season), low_memory=False)
        report = report[
            (report.game_type == "REG")
            & report.position.isin(["QB", "RB", "FB", "WR", "TE"])
        ]
        injuries.append(report[INJURY_COLUMNS])
    injuries = pd.concat(injuries, ignore_index=True)
    injuries.to_csv(DATA / "injuries.csv.gz", index=False, compression=GZIP)

    played = games[games.home_score.notna()]
    retrieval = {
        "retrieved": datetime.now(UTC).date().isoformat(),
        "seasons": [FIRST_SEASON, CURRENT_SEASON],
        "last_completed_week": {
            "season": int(played.season.max()),
            "week": int(played[played.season == played.season.max()].week.max()),
        },
        "sources": {
            "games": GAMES_URL,
            "player_week": PLAYER_URL,
            "injuries": INJURY_URL,
        },
        "rows": {
            "games": len(games),
            "player_week": len(players),
            "injuries": len(injuries),
        },
        "sha256": {
            "games.csv": sha256(DATA / "games.csv"),
            "player_week.csv.gz": sha256(DATA / "player_week.csv.gz"),
            "injuries.csv.gz": sha256(DATA / "injuries.csv.gz"),
        },
    }
    (DATA / "retrieval.json").write_text(json.dumps(retrieval, indent=2) + "\n")
    print(json.dumps(retrieval["rows"]), retrieval["last_completed_week"])


if __name__ == "__main__":
    main()
