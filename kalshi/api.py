"""Read public Kalshi NBA player-prop prices: live snapshots and settled history.

Kalshi lists each prop as a ladder of yes/no contracts, for example
"Victor Wembanyama: 25+ points" (strike 24.5). A yes contract pays $1 if it
happens, so its price is the market's probability. No login is needed for
market data. Markets that settled before Kalshi's historical cutoff move to
the /historical endpoints, which still serve their hourly bid/ask candles.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"


def market_data_base_url():
    """Use the same Kalshi venue as the trader when KALSHI_ENV is explicit.

    Unconfigured research tasks retain historical production-market behavior.
    The demo environment has its own markets and order books; production
    quotes must not be used to decide prices for demo orders.
    """
    env = os.environ.get("KALSHI_ENV", "prod").strip().lower()
    if env == "demo":
        return DEMO_BASE_URL
    if env == "prod":
        return BASE_URL
    raise ValueError(f"Invalid KALSHI_ENV {env!r}; expected demo or prod")


# Kalshi series -> the model's stat name.
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
    url = f"{market_data_base_url()}{path}?{urllib.parse.urlencode(params)}"
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


def snapshot(series, fetch=fetch_series, now=None):
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


def trades(ticker, min_ts, max_ts, historical=True):
    """Every trade print for a market between two Unix times."""
    path = "/historical/trades" if historical else "/markets/trades"
    rows, cursor = [], ""
    while True:
        params = {"ticker": ticker, "min_ts": int(min_ts), "max_ts": int(max_ts)}
        params["limit"] = 1000
        if cursor:
            params["cursor"] = cursor
        page = get_json(path, params) or {}
        for t in page.get("trades", []):
            rows.append(
                {
                    "ticker": t.get("ticker", ticker),
                    "created_time": t.get("created_time"),
                    "yes_price": _number(t.get("yes_price_dollars")),
                    "count": _number(t.get("count_fp")),
                    "taker_outcome_side": t.get("taker_outcome_side")
                    or t.get("taker_side"),
                    "is_block_trade": bool(t.get("is_block_trade", False)),
                }
            )
        cursor = page.get("cursor") or ""
        if not cursor or not page.get("trades"):
            return rows


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _close(block):
    return _number((block or {}).get("close"))


def fee_per_contract(price):
    """Kalshi's general fee per contract before rounding up to the cent."""
    price = np.asarray(price, dtype=float)
    return FEE_RATE * price * (1 - price)


def fetch_market(ticker):
    """One market, from the live endpoint or the historical one."""
    for path in (f"/markets/{ticker}", f"/historical/markets/{ticker}"):
        payload = get_json(path, {})
        if payload and payload.get("market"):
            return payload["market"]
    return None


def fetch_trades(ticker, min_ts, max_ts):
    """Trade prints from the live endpoint, else the historical one."""
    rows = trades(ticker, min_ts, max_ts, historical=False)
    return rows or trades(ticker, min_ts, max_ts, historical=True)
