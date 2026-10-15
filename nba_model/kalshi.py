"""Read public Kalshi NBA player-prop prices: live snapshots and settled history.

Kalshi lists each prop as a ladder of yes/no contracts, for example
"Victor Wembanyama: 25+ points" (strike 24.5). A yes contract pays $1 if it
happens, so its price is the market's probability. No login is needed for
market data. Markets that settled before Kalshi's historical cutoff move to
the /historical endpoints, which still serve their hourly bid/ask candles.
"""

import json
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
# Kalshi series -> the model's stat name.
SERIES = {
    "KXNBAPTS": "points",
    "KXNBAREB": "rebounds",
    "KXNBAAST": "assists",
    "KXNBA3PT": "threes",
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
    "open_time",
    "occurrence_datetime",
    "close_time",
]
SETTLED_FIELDS = FIELDS + ["result", "settlement_value_dollars"]
FEE_RATE = 0.07  # Kalshi general trading fee: 0.07 x contracts x P x (1 - P)
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


class RateLimiter:
    """At most ``per_second`` requests per second, shared across threads."""

    def __init__(self, per_second):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_time)
            self.next_time = start + self.interval
        time.sleep(max(0.0, start - now))


LIMITER = RateLimiter(10)


def get_json(path, params, retries=6):
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        LIMITER.wait()
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == retries - 1:
                raise
            # 429 means too many requests: back off harder.
            time.sleep((5 if error.code == 429 else 2) * (attempt + 1))
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def fetch_markets(series_ticker, status="open", historical=False):
    """Every market in a series, following Kalshi's cursor pagination."""
    path = "/historical/markets" if historical else "/markets"
    markets, cursor = [], ""
    while True:
        params = {"series_ticker": series_ticker, "limit": 1000}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        page = get_json(path, params) or {}
        markets.extend(page.get("markets", []))
        cursor = page.get("cursor") or ""
        if not cursor or not page.get("markets"):
            return markets


def fetch_series(series_ticker, status="open"):
    return fetch_markets(series_ticker, status=status)


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


def settled_frame(markets, series_ticker):
    rows = [{field: m.get(field) for field in SETTLED_FIELDS} for m in markets]
    frame = pd.DataFrame(rows, columns=SETTLED_FIELDS)
    frame.insert(0, "series", series_ticker)
    return frame


def candles(ticker, start_ts, end_ts, period=60, historical=True):
    """Hourly (``period`` minutes) candles for one market: bid, ask, last, volume."""
    prefix = (
        "/historical/markets"
        if historical
        else f"/series/{ticker.split('-')[0]}/markets"
    )
    payload = get_json(
        f"{prefix}/{ticker}/candlesticks",
        {"start_ts": int(start_ts), "end_ts": int(end_ts), "period_interval": period},
    )
    rows = []
    for c in (payload or {}).get("candlesticks", []):
        rows.append(
            {
                "ticker": ticker,
                "end_ts": int(c["end_period_ts"]),
                "yes_bid": _close(c.get("yes_bid")),
                "yes_ask": _close(c.get("yes_ask")),
                "price": _close(c.get("price")),
                "volume": _number(c.get("volume")),
                "open_interest": _number(c.get("open_interest")),
            }
        )
    return rows


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _close(block):
    return _number((block or {}).get("close"))


def normalize_name(name):
    """'Jaren Jackson Jr.' and 'Jaren Jackson' -> 'jaren jackson'."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    words = re.sub(r"[^a-z ]", " ", text.lower().replace("'", "")).split()
    return " ".join(w for w in words if w not in SUFFIXES)


def parse(snapshot_frame):
    """Add player, team, stat, line, game date and prices to raw market rows."""
    frame = snapshot_frame.copy()
    frame["stat"] = frame.series.map(SERIES)
    # Titles were "Anthony Edwards records 40+ points" until January 2026 and
    # "Anthony Edwards: 40+ points" after; the yes subtitle is always
    # "Anthony Edwards: 40+".
    name = frame.yes_sub_title.fillna(frame.title).str.split(":").str[0]
    frame["player"] = name.str.replace(r" records .*$", "", regex=True).str.strip()
    frame["player_key"] = frame.player.map(normalize_name)
    plus = frame.yes_sub_title.str.extract(r"(\d+)\+$")[0].astype(float)
    frame["line"] = pd.to_numeric(frame.floor_strike, errors="coerce").fillna(
        plus - 0.5
    )
    # Event tickers look like KXNBAPTS-26MAY30SASOKC: game date 2026-05-30,
    # San Antonio (away) at Oklahoma City (home). Market tickers add the
    # player's team, e.g. ...-SASVWEMBANYAMA1-25.
    event = frame.event_ticker.str.split("-").str[1]
    frame["game_date"] = pd.to_datetime(
        event.str[:7], format="%y%b%d", errors="coerce"
    ).dt.date
    frame["away"] = event.str[7:10]
    frame["home"] = event.str[10:13]
    frame["team"] = frame.ticker.str.split("-").str[2].str[:3]
    for column in ["yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["mid"] = (frame.yes_bid_dollars + frame.yes_ask_dollars) / 2
    return frame


def fee_per_contract(price):
    """Kalshi's general fee per contract before rounding up to the cent."""
    price = np.asarray(price, dtype=float)
    return FEE_RATE * price * (1 - price)
