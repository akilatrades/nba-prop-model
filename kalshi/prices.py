"""Saved Kalshi price history: markets, hourly candles, the quote at a chosen time."""

import pandas as pd


def price_at_tip(matched, candles, hours_before=0):
    """Last candle ending at or before tip-off (minus ``hours_before``)."""
    c = candles.copy()
    m = matched[["ticker", "tip_utc"]].dropna()
    cutoff = m.tip_utc - pd.Timedelta(hours=hours_before)
    m = m.assign(cutoff_ts=cutoff.map(lambda t: int(t.timestamp())))
    c = c.merge(m[["ticker", "cutoff_ts"]], on="ticker")
    c = c[c.end_ts <= c.cutoff_ts].sort_values("end_ts")
    last = c.drop_duplicates("ticker", keep="last")
    return last[["ticker", "end_ts", "yes_bid", "yes_ask", "price", "volume"]]


def two_sided(frame):
    return (frame.yes_bid > 0) & (frame.yes_ask < 1) & (frame.yes_ask > frame.yes_bid)
