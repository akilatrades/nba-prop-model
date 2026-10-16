"""Read-only check of The Odds API key: pull NFL player props for the next games
and report how many of Kalshi's open NFL prop markets have a matching book line.

Needs ODDS_API_KEY. Spends a few credits. Run: python -m live.check_odds
"""

import json
import os
import sys
import urllib.parse
import urllib.request

import pandas as pd

from kalshi.api import snapshot
from markets.nfl.kalshi import normalize_name, parse

BASE = "https://api.the-odds-api.com/v4"
MARKETS = {"player_anytime_td": "skill_tds", "player_reception_yds": "receiving_yards"}


def get(path, params):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        remaining = response.headers.get("x-requests-remaining")
        return json.loads(response.read()), remaining


def main():
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        print("::error::ODDS_API_KEY is empty")
        sys.exit(1)
    events, remaining = get("/sports/americanfootball_nfl/events", {"apiKey": key})
    events = events[:4]
    rows = []
    for event in events:
        odds, remaining = get(
            f"/sports/americanfootball_nfl/events/{event['id']}/odds",
            {
                "apiKey": key,
                "regions": "us",
                "markets": ",".join(MARKETS),
                "oddsFormat": "american",
            },
        )
        for book in odds.get("bookmakers", []):
            for market in book.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "event": f"{event['away_team']} @ {event['home_team']}",
                            "book": book["key"],
                            "stat": MARKETS.get(market["key"], market["key"]),
                            "player": outcome.get("description"),
                            "side": outcome.get("name"),
                            "line": outcome.get("point", 0.5),
                            "odds": outcome.get("price"),
                        }
                    )
    lines = pd.DataFrame(rows)
    kalshi = parse(snapshot(["KXNFLANYTD", "KXNFLRECYDS"]))
    kalshi["key"] = kalshi.player_key + "|" + kalshi.stat
    if len(lines):
        lines["player_key"] = lines.player.map(normalize_name)
        lines["key"] = lines.player_key + "|" + lines.stat
        covered = kalshi.key.isin(set(lines.key))
    else:
        covered = pd.Series(False, index=kalshi.index)
    out = {
        "events_checked": len(events),
        "book_lines": len(lines),
        "books": sorted(lines.book.unique().tolist()) if len(lines) else [],
        "kalshi_open_nfl_markets": len(kalshi),
        "kalshi_markets_with_a_book_line_for_the_player_and_stat": int(covered.sum()),
        "coverage": round(float(covered.mean()), 3) if len(kalshi) else None,
        "credits_remaining": remaining,
    }
    print(json.dumps(out))
    print("::notice::" + json.dumps(out))


if __name__ == "__main__":
    main()
