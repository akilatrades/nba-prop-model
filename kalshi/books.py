"""Sportsbook player-prop lines from The Odds API, and the book probability of
the over at each Kalshi line (RESEARCH_LOG.md, 2026-10-16, "sportsbook-line
veto").

Current odds: ``props(key, sport)`` pulls every US book's main and alternate
lines for the next games. Historical odds: ``historical_event_odds`` returns
the feed's snapshot at or just before a timestamp. ``consensus`` turns raw
quotes into one probability per player, stat and line (vig removed within
each book, averaged across books); ``attach`` merges that onto a priced
Kalshi frame as ``p_book`` and ``book_n``.

Credits: current event odds cost 1 per market per region per event (8 per
game here); historical cost 10 per market per region per event (80 per game).
"""

import gzip
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from markets.nba.kalshi import normalize_name

BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
REGIONS = "us"
# The Odds API market key -> our stat name. Alternates carry the extra lines.
MARKETS = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_points_alternate": "points",
    "player_rebounds_alternate": "rebounds",
    "player_assists_alternate": "assists",
    "player_threes_alternate": "threes",
}
# NFL: receiving yards is quoted both sides and can be de-vigged; anytime
# touchdown is quoted "Yes" only, so it has no book probability here.
NFL_SPORT = "americanfootball_nfl"
NFL_MARKETS = {
    "player_reception_yds": "receiving_yards",
    "player_reception_yds_alternate": "receiving_yards",
}
SPORTS = {
    "nba": (SPORT, MARKETS, "nba_latest.csv"),
    "nfl": (NFL_SPORT, NFL_MARKETS, "nfl_latest.csv"),
}
# Full team names as the feed gives them -> Kalshi's three-letter codes.
TEAMS = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}
LINE_COLUMNS = [
    "event_id",
    "commence_utc",
    "home",
    "away",
    "book",
    "market",
    "stat",
    "player",
    "player_key",
    "line",
    "side",
    "odds",
    "snapshot_utc",
]


class BooksError(Exception):
    pass


def _get(path, params, timeout=30):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            remaining = response.headers.get("x-requests-remaining")
            used = response.headers.get("x-requests-used")
            return json.loads(response.read()), {"remaining": remaining, "used": used}
    except urllib.error.HTTPError as error:
        body = error.read(300).decode("utf-8", "replace")
        raise BooksError(f"odds api {path}: HTTP {error.code} {body}") from error
    except urllib.error.URLError as error:
        raise BooksError(f"odds api {path}: {error.reason}") from error


def api_key():
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise BooksError("ODDS_API_KEY is empty")
    return key


# --- requests -----------------------------------------------------------------


def events(key, sport=SPORT):
    """Upcoming events: id, commence_time, home_team, away_team."""
    data, usage = _get(f"/sports/{sport}/events", {"apiKey": key})
    return data, usage


def event_odds(key, event_id, sport=SPORT, markets=MARKETS, regions=REGIONS):
    data, usage = _get(
        f"/sports/{sport}/events/{event_id}/odds",
        {
            "apiKey": key,
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "american",
        },
    )
    return data, usage


def historical_events(key, date_iso, sport=SPORT):
    """Events as the feed listed them at ``date_iso`` (ISO 8601, UTC)."""
    data, usage = _get(
        f"/historical/sports/{sport}/events", {"apiKey": key, "date": date_iso}
    )
    return data.get("data", []), data.get("timestamp"), usage


def historical_event_odds(
    key, event_id, date_iso, sport=SPORT, markets=MARKETS, regions=REGIONS
):
    """The snapshot at or just before ``date_iso``; returns (odds, snapshot ts)."""
    data, usage = _get(
        f"/historical/sports/{sport}/events/{event_id}/odds",
        {
            "apiKey": key,
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "date": date_iso,
        },
    )
    return data.get("data", {}), data.get("timestamp"), usage


# --- parsing -------------------------------------------------------------------


def parse_event(odds, snapshot_utc=None, markets=MARKETS):
    """One row per book, market, player, line and side from an event-odds reply."""
    rows = []
    home, away = odds.get("home_team"), odds.get("away_team")
    for book in odds.get("bookmakers", []) or []:
        for market in book.get("markets", []) or []:
            stat = markets.get(market.get("key"))
            if stat is None:
                continue
            for outcome in market.get("outcomes", []) or []:
                player = outcome.get("description")
                if not player or outcome.get("point") is None:
                    continue
                rows.append(
                    {
                        "event_id": odds.get("id"),
                        "commence_utc": odds.get("commence_time"),
                        "home": TEAMS.get(home, home),
                        "away": TEAMS.get(away, away),
                        "book": book.get("key"),
                        "market": market.get("key"),
                        "stat": stat,
                        "player": player,
                        "player_key": normalize_name(player),
                        "line": float(outcome["point"]),
                        "side": outcome.get("name"),
                        "odds": outcome.get("price"),
                        "snapshot_utc": snapshot_utc or market.get("last_update"),
                    }
                )
    return pd.DataFrame(rows, columns=LINE_COLUMNS)


def implied(american):
    """American odds -> implied probability including the vig."""
    a = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def consensus(lines):
    """Book probability of the over per (player_key, stat, line).

    Within one book, a line quoted on both sides has its vig removed
    proportionally (over / (over + under)). A book quoting one side only is
    skipped: its number alone cannot be de-vigged. Books are then averaged
    with equal weight. ``book_n`` is how many books went into the average.
    """
    if lines is None or lines.empty:
        return pd.DataFrame(columns=["player_key", "stat", "line", "p_book", "book_n"])
    q = lines.copy()
    q["p_raw"] = implied(q.odds)
    q["side"] = q.side.str.lower()
    keys = ["book", "player_key", "stat", "line"]
    # One quote per book, line and side (alternate and main markets overlap).
    q = q.drop_duplicates(keys + ["side"])
    wide = q.pivot_table(
        index=keys, columns="side", values="p_raw", aggfunc="first"
    ).reset_index()
    if "over" not in wide or "under" not in wide:
        return pd.DataFrame(columns=["player_key", "stat", "line", "p_book", "book_n"])
    both = wide.dropna(subset=["over", "under"]).copy()
    both["p_over"] = both.over / (both.over + both.under)
    out = (
        both.groupby(["player_key", "stat", "line"])
        .agg(p_book=("p_over", "mean"), book_n=("p_over", "size"))
        .reset_index()
    )
    out["book_n"] = out.book_n.astype(int)
    return out


def attach(priced, book):
    """Add ``p_book`` and ``book_n`` to a priced Kalshi frame by player, stat, line.

    ``priced`` needs player (or player_key), stat and line. Lines are matched
    exactly (Kalshi's 24.5 against the books' 24.5); a Kalshi line the books
    do not quote gets NaN.
    """
    p = priced.copy()
    if "player_key" not in p:
        p["player_key"] = p.player.map(normalize_name)
    p["line"] = pd.to_numeric(p.line, errors="coerce").round(1)
    if book is None or book.empty:
        p["p_book"] = np.nan
        p["book_n"] = 0
        return p
    b = book.copy()
    b["line"] = pd.to_numeric(b.line, errors="coerce").round(1)
    b = b.drop_duplicates(["player_key", "stat", "line"])
    p = p.drop(columns=["p_book", "book_n"], errors="ignore").merge(
        b[["player_key", "stat", "line", "p_book", "book_n"]],
        on=["player_key", "stat", "line"],
        how="left",
    )
    p["book_n"] = p.book_n.fillna(0).astype(int)
    return p


LATEST = Path("outputs/books/nba_latest.csv")


def attach_latest(priced, path=LATEST, max_age_hours=4):
    """``attach`` with the last book log (live/log_books.py), if it is fresh.

    A log older than ``max_age_hours`` is ignored: a stale book line is worse
    than none, since the model veto then takes over for that line.
    """
    path = Path(path)
    if not path.exists() or priced.empty:
        return attach(priced, None)
    book = pd.read_csv(path)
    if len(book) and "snapshot_utc" in book:
        age = pd.Timestamp.now(tz="UTC") - pd.to_datetime(
            book.snapshot_utc.iloc[0], utc=True
        )
        if age > pd.Timedelta(hours=max_age_hours):
            print(f"Book lines are {age} old; ignored")
            return attach(priced, None)
    out = attach(priced, book)
    print(f"Book lines: {int(out.p_book.notna().sum())} of {len(out)} markets covered")
    return out


# --- convenience ------------------------------------------------------------------


def props(key, sport="nba", max_events=None, hours_ahead=36):
    """Current lines for upcoming games of one sport (1 credit per market per game)."""
    sport_key, markets, _ = SPORTS[sport]
    listed, usage = events(key, sport=sport_key)
    now = pd.Timestamp.now(tz="UTC")
    frames = []
    for event in listed:
        start = pd.Timestamp(event["commence_time"])
        if start < now - pd.Timedelta(hours=1) or start > now + pd.Timedelta(
            hours=hours_ahead
        ):
            continue
        odds, usage = event_odds(key, event["id"], sport=sport_key, markets=markets)
        frames.append(parse_event(odds, markets=markets))
        if max_events and len(frames) >= max_events:
            break
    lines = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=LINE_COLUMNS)
    )
    return lines, usage


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def load_json(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)
