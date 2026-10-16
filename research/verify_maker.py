"""Recompute the limit-order test's headline with separate code.

Rebuilds orders and fills from the raw files without importing
kalshi.maker or kalshi.strategy, and compares with
outputs/new_ideas/results.json.

Run from the repository root: python scripts/verify_maker.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path("data/kalshi_history")
SERIES = ["KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT"]
priced = pd.read_csv("outputs/market_test/priced_markets.csv.gz", low_memory=False)
priced["tip"] = pd.to_datetime(priced.tip_utc, utc=True).map(
    lambda t: int(t.timestamp())
)
candles = pd.concat([pd.read_parquet(H / f"candles_{s}.parquet") for s in SERIES])
trades = pd.concat([pd.read_parquet(H / f"trades_{s}.parquet") for s in SERIES])
trades = trades[~trades.is_block_trade]
trades["ts"] = pd.to_datetime(trades.created_time, utc=True, format="ISO8601").map(
    lambda t: int(t.timestamp())
)
by_ticker = dict(tuple(trades.groupby("ticker")))
candles = candles.merge(priced[["ticker", "tip"]], on="ticker").sort_values("end_ts")


def quotes(hours):
    c = candles[candles.end_ts <= candles.tip - 3600 * hours]
    return c.groupby("ticker").tail(1).set_index("ticker")


taken, orders = set(), []
for hours in (5, 2):
    q = quotes(hours)
    rows = []
    for r in priced.itertuples():
        if r.ticker not in q.index:
            continue
        c = q.loc[r.ticker]
        bid, ask = c.yes_bid, c.yes_ask
        if not (bid > 0 and ask < 1 and ask > bid):
            continue
        improve = round(ask - bid, 4) >= 0.02
        if r.p_model >= (bid + ask) / 2:
            side, price, p = "yes", bid + (0.01 if improve else 0), r.p_model
        else:
            side, price, p = "no", (1 - ask) + (0.01 if improve else 0), 1 - r.p_model
        edge = p - price - 0.25 * 0.07 * price * (1 - price)
        if edge >= 0.03:
            rows.append(
                (
                    r.game_id,
                    r.player_id,
                    r.stat,
                    edge,
                    r.ticker,
                    side,
                    price,
                    improve,
                    int(c.end_ts),
                    r.tip,
                    r.over,
                    r.season_type,
                )
            )
    rows.sort(key=lambda x: -x[3])
    for row in rows:
        key = row[:3]
        if key in taken:
            continue
        taken.add(key)
        orders.append(row)

results = []
for game, _, _, _, ticker, side, price, improve, start, tip, over, kind in orders:
    t = by_ticker.get(ticker)
    filled = False
    if t is not None:
        t = t[(t.ts > start) & (t.ts <= tip)]
        if side == "yes":
            hit = (t.taker_outcome_side == "no") & (
                (t.yes_price < price - 1e-6)
                | (improve & ((t.yes_price - price).abs() < 1e-6))
            )
        else:
            level = 1 - price
            hit = (t.taker_outcome_side == "yes") & (
                (t.yes_price > level + 1e-6)
                | (improve & ((t.yes_price - level).abs() < 1e-6))
            )
        filled = bool(hit.any())
    value = over if side == "yes" else 1 - over
    pnl = value - price - 0.25 * 0.07 * price * (1 - price)
    results.append((game, filled, pnl, kind))
r = pd.DataFrame(results, columns=["game", "filled", "pnl", "kind"])
f = r[r.filled]
mean = f.pnl.mean()
by_game = (f.pnl - mean).groupby(f.game).sum()
se = np.sqrt((by_game**2).sum() * len(by_game) / (len(by_game) - 1)) / len(f)
reported = json.loads(Path("outputs/new_ideas/results.json").read_text())
reported = reported["idea1_maker"]["maker_fee_quarter"]
print(
    f"orders {len(r)}, fill rate {r.filled.mean():.3f}, filled {len(f)}, "
    f"{100 * mean:+.2f}c per contract, t {mean / se:.2f}"
)
print(
    f"reported: orders {reported['orders']}, fill rate {reported['fill_rate']:.3f}, "
    f"filled {reported['filled']['positions']}, "
    f"{reported['filled']['cents_per_contract']:+.2f}c, t {reported['filled']['t']:.2f}"
)
