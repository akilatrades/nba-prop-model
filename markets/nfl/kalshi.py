"""Read public Kalshi NFL player-prop prices and line them up with the model.

Kalshi lists each prop as a ladder of yes/no contracts, for example
"Mike Evans: 80+ receiving yards" (strike 79.5). A yes contract pays $1 if
it happens, so its price is the market's probability. No login is needed
for market data.
"""

import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
# Kalshi series -> the model's stat name.
SERIES = {
    "KXNFLPASSYDS": "passing_yards",
    "KXNFLPASSTDS": "passing_tds",
    "KXNFLRECYDS": "receiving_yards",
    "KXNFLREC": "receptions",
    "KXNFLRSHYDS": "rushing_yards",
    "KXNFLANYTD": "skill_tds",
}
FIELDS = [
    "ticker",
    "event_ticker",
    "title",
    "yes_sub_title",
    "floor_strike",
    "strike_type",
    "status",
    "yes_bid_dollars",
    "yes_ask_dollars",
    "last_price_dollars",
    "volume_fp",
    "open_interest_fp",
    "occurrence_datetime",
    "close_time",
]
FEE_RATE = 0.07  # Kalshi general trading fee: 0.07 x contracts x P x (1 - P)
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def get_json(path, params, retries=3):
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def fetch_series(series_ticker, status="open"):
    """Every market in a series, following Kalshi's cursor pagination."""
    markets, cursor = [], ""
    while True:
        params = {"series_ticker": series_ticker, "status": status, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        page = get_json("/markets", params)
        markets.extend(page.get("markets", []))
        cursor = page.get("cursor") or ""
        if not cursor:
            return markets


def snapshot(series=SERIES, fetch=fetch_series, now=None):
    """One row per open market across the prop series, with a timestamp."""
    now = now or datetime.now().astimezone()
    rows = []
    for series_ticker in series:
        for market in fetch(series_ticker):
            row = {field: market.get(field) for field in FIELDS}
            row["series"] = series_ticker
            rows.append(row)
    frame = pd.DataFrame(rows, columns=["series", *FIELDS])
    frame.insert(0, "snapshot_utc", pd.Timestamp(now).tz_convert("UTC").isoformat())
    return frame


def normalize_name(name):
    """'Deebo Samuel Sr.' and 'Deebo Samuel' -> 'deebo samuel'."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    words = re.sub(r"[^a-z ]", " ", text.lower().replace("'", "")).split()
    return " ".join(w for w in words if w not in SUFFIXES)


def parse(snapshot_frame):
    """Add player, stat, line, game date and prices to a raw snapshot."""
    frame = snapshot_frame.copy()
    frame["stat"] = frame.series.map(SERIES)
    frame["player"] = frame.title.str.split(":").str[0].str.strip()
    frame["player_key"] = frame.player.map(normalize_name)
    plus = frame.yes_sub_title.str.extract(r"(\d+)\+$")[0].astype(float)
    frame["line"] = pd.to_numeric(frame.floor_strike, errors="coerce").fillna(
        plus - 0.5
    )
    frame.loc[frame.stat.eq("skill_tds") & frame.line.isna(), "line"] = 0.5
    # Event tickers look like KXNFLRECYDS-26SEP10SFLAR: game date 2026-09-10.
    date = frame.event_ticker.str.split("-").str[1].str[:7]
    frame["game_date"] = pd.to_datetime(date, format="%y%b%d", errors="coerce").dt.date
    for column in ["yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["mid"] = (frame.yes_bid_dollars + frame.yes_ask_dollars) / 2
    return frame


def fee_per_contract(price):
    """Kalshi's general fee per contract before rounding up to the cent."""
    price = np.asarray(price, dtype=float)
    return FEE_RATE * price * (1 - price)
