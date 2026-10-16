"""Log the sportsbooks' NBA prop lines (RESEARCH_LOG.md, 2026-10-16).

Pulls every US book's prop lines for games in the next 36 hours from The
Odds API (NBA: points, rebounds, assists, threes, main and alternate; NFL:
receiving yards), removes the vig within each book, averages across books,
and writes:

    outputs/books/<sport>_latest.csv       one row per player, stat and line
    outputs/books/<sport>_<day>.csv.gz     every snapshot that day, appended

Only the de-vigged consensus is kept (the feed's terms do not allow
redistributing bookmakers' odds). About 8 credits per NBA game, 2 per NFL
game. Needs ODDS_API_KEY. Run: python -m live.log_books [nba|nfl]
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kalshi.books import SPORTS, BooksError, api_key, consensus, props

OUT = Path("outputs/books")


def main(sport="nba"):
    now = datetime.now(UTC)
    lines, usage = props(api_key(), sport=sport)
    table = consensus(lines)
    table["snapshot_utc"] = now.isoformat(timespec="minutes")
    if len(lines):
        starts = lines.drop_duplicates("event_id").set_index("event_id").commence_utc
        games = lines.drop_duplicates(["player_key", "stat"])[
            ["player_key", "stat", "event_id", "home", "away"]
        ]
        games["commence_utc"] = games.event_id.map(starts)
        table = table.merge(
            games.drop_duplicates(["player_key", "stat"]),
            on=["player_key", "stat"],
            how="left",
        )
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / SPORTS[sport][2], index=False)
    day = OUT / f"{sport}_{now:%Y-%m-%d}.csv.gz"
    old = pd.read_csv(day) if day.exists() else pd.DataFrame()
    pd.concat([old, table], ignore_index=True).to_csv(day, index=False)
    out = {
        "sport": sport,
        "snapshot_utc": table.snapshot_utc.iloc[0] if len(table) else None,
        "games": int(lines.event_id.nunique()) if len(lines) else 0,
        "books": sorted(lines.book.unique().tolist()) if len(lines) else [],
        "lines_quoted_both_sides": int(len(table)),
        "players": int(table.player_key.nunique()) if len(table) else 0,
        "credits_remaining": usage.get("remaining"),
    }
    print(json.dumps(out))
    print("::notice::" + json.dumps(out))


if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else "nba")
    except BooksError as error:
        print(f"::error::{error}")
        sys.exit(1)
