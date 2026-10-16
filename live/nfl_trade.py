"""NFL prop markets for the live trader (RESEARCH_LOG.md, 2026-10-16).

Builds the priced frame the trader's rules take (the same columns the NBA
projection run writes) from Kalshi's open anytime-touchdown and
receiving-yards markets: fresh quotes, kickoff from the market's
occurrence time, the NFL model's over-probability where the latest ladder
has one, and the sportsbooks' consensus where they quote the line.

Run through the trader: TRADE_MARKET=nfl python -m live.nba_trade
"""

from pathlib import Path

import numpy as np
import pandas as pd

from kalshi.api import fetch_markets
from kalshi.books import attach_latest
from markets.nfl.data import load as load_nfl
from markets.nfl.kalshi import parse
from markets.nfl.paper import MODEL_SERIES, model_probabilities

SERIES = sorted(MODEL_SERIES)  # KXNFLANYTD, KXNFLRECYDS
BOOKS = Path("outputs/books/nfl_latest.csv")
COLUMNS = [
    "snapshot_utc",
    "tip_utc",
    "event_ticker",
    "ticker",
    "player",
    "player_id",
    "team",
    "stat",
    "line",
    "p_model",
    "yes_bid_dollars",
    "yes_ask_dollars",
    "hours_to_tip",
    "injury_status",
    "game_id",
]


def priced_frame(markets, p_model, now):
    """Open markets as one priced frame; ``p_model`` maps ticker to P(over)."""
    m = pd.DataFrame(markets)
    if m.empty:
        return pd.DataFrame(columns=COLUMNS)
    m["series"] = m.ticker.str.split("-").str[0]
    m = parse(m)
    m["tip_utc"] = pd.to_datetime(
        m.occurrence_datetime, utc=True, format="ISO8601", errors="coerce"
    )
    m = m[m.tip_utc.notna()].copy()
    m["hours_to_tip"] = (m.tip_utc - pd.Timestamp(now)).dt.total_seconds() / 3600
    m["team"] = m.ticker.str.split("-").str[2].str[:3]
    m["game_id"] = m.event_ticker.str.split("-").str[1]  # e.g. 26OCT18CHIATL
    m["player_id"] = -1  # no roster ids here; the injury filter is NBA-only
    m["p_model"] = m.ticker.map(p_model).astype(float)
    m["injury_status"] = ""
    m["snapshot_utc"] = pd.Timestamp(now).isoformat(timespec="minutes")
    m = m[COLUMNS]
    m["line"] = pd.to_numeric(m.line, errors="coerce")
    return m


def candidates(now):
    """Priced NFL prop markets with model and book probabilities attached."""
    markets = []
    for series in SERIES:
        markets += fetch_markets(series, status="open", historical=False)
    if not markets:
        print("No open NFL prop markets")
        return pd.DataFrame(columns=COLUMNS)
    try:
        games, _ = load_nfl()
        p_model = model_probabilities(markets, games, now)
    except (OSError, ValueError, KeyError) as error:
        print(f"NFL model unavailable ({error}); book veto only")
        p_model = {}
    priced = priced_frame(markets, p_model, now)
    priced = attach_latest(priced, path=BOOKS)
    priced["p_model"] = priced.p_model.replace({None: np.nan})
    print(
        f"NFL: {len(priced)} open markets, {int(priced.p_model.notna().sum())} priced "
        f"by the model, {int(priced.p_book.notna().sum())} with a book line"
    )
    return priced
