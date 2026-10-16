"""Idea 1: rest limit orders instead of paying the ask (RESEARCH_LOG.md).

On the side the model favours, an order rests one cent above that side's
best bid (alone at the best price) when the spread is at least 2 cents, or
at the bid (behind a queue of unknown size) when the spread is 1 cent. It is
cancelled at tip-off. Whether it filled is read from Kalshi's trade prints:

- a resting buy of yes at q is hit when a taker sells yes, which Kalshi
  prints as taker outcome side "no", at a yes price of q or less (strictly
  less for an order that joined the queue);
- a resting buy of no at q is the same as selling yes at 1 - q, and is hit
  when a taker buys yes (outcome side "yes") at a yes price of 1 - q or more
  (strictly more if it joined the queue).

Price priority makes this safe: a taker who traded at a worse price would
have traded with this order first.
"""

import numpy as np
import pandas as pd

from kalshi.strategy import one_per_player

MIN_EDGE = 0.03
TICK = 0.01


def maker_fee(price, share=0.25):
    """Kalshi's maker fee as a share of the taker fee 0.07 x P x (1 - P)."""
    price = np.asarray(price, dtype=float)
    return share * 0.07 * price * (1 - price)


def orders(quotes, min_edge=MIN_EDGE, fee_share=0.25):
    """One resting order per market where the model's edge clears min_edge.

    ``quotes`` has p_model, yes_bid, yes_ask, end_ts (when the quote was seen).
    """
    q = quotes.copy()
    spread = (q.yes_ask - q.yes_bid).round(4)
    ok = (q.yes_bid > 0) & (q.yes_ask < 1) & (spread > 0)
    improve = spread >= 2 * TICK - 1e-9
    yes_price = np.where(improve, q.yes_bid + TICK, q.yes_bid)
    no_bid = 1 - q.yes_ask
    no_price = np.where(improve, no_bid + TICK, no_bid)
    yes_edge = q.p_model - yes_price - maker_fee(yes_price, fee_share)
    no_edge = (1 - q.p_model) - no_price - maker_fee(no_price, fee_share)
    yes_side = q.p_model >= (q.yes_bid + q.yes_ask) / 2
    q["side"] = np.where(yes_side, "yes", "no")
    q["price"] = np.where(yes_side, yes_price, no_price).round(4)
    q["edge"] = np.where(yes_side, yes_edge, no_edge)
    q["improved"] = improve
    q = q[ok & (q.edge >= min_edge)].copy()
    # The same order seen from the yes side, for matching trade prints.
    q["yes_level"] = np.where(q.side == "yes", q.price, 1 - q.price).round(4)
    return q


def first_orders(quotes_by_hours, **kwargs):
    """First qualifying order per player, stat and game across the looks."""
    taken, out = set(), []
    for hours, quotes in quotes_by_hours:
        o = one_per_player(orders(quotes, **kwargs))
        keys = list(zip(o.game_id, o.player_id, o.stat))
        o = o[np.array([k not in taken for k in keys], dtype=bool)]
        taken.update(zip(o.game_id, o.player_id, o.stat))
        out.append(o.assign(hours_before=hours))
    return pd.concat(out, ignore_index=True)


def fills(placed, trades):
    """Contracts that traded through each order's price before tip-off.

    ``placed`` needs ticker, end_ts, tip_ts, side, yes_level, improved.
    ``trades`` needs ticker, ts, yes_price, count, taker_outcome_side.
    """
    t = trades.merge(
        placed[["ticker", "end_ts", "tip_ts", "side", "yes_level", "improved"]],
        on="ticker",
    )
    t = t[(t.ts > t.end_ts) & (t.ts <= t.tip_ts)]
    level = t.yes_level
    same = (t.yes_price - level).abs() < 1e-6
    hit_yes = (t.side == "yes") & (t.taker_outcome_side == "no")
    hit_yes &= (t.yes_price < level - 1e-6) | (same & t.improved)
    hit_no = (t.side == "no") & (t.taker_outcome_side == "yes")
    hit_no &= (t.yes_price > level + 1e-6) | (same & t.improved)
    hit = t[hit_yes | hit_no]
    return hit.groupby("ticker")["count"].sum()


def settle_orders(placed, filled_contracts, fee_share=0.25):
    p = placed.copy()
    p["filled_contracts"] = p.ticker.map(filled_contracts).fillna(0.0)
    p["filled"] = p.filled_contracts > 0
    won = np.where(p.side == "yes", p.over, 1 - p.over)
    p["pnl_if_filled"] = won - p.price - maker_fee(p.price, fee_share)
    p["pnl_no_fee"] = won - p.price
    return p


def maker_days(settled):
    """Per-day arrays for the bankroll simulation: price, traded, value."""
    s = settled.sort_values("tip_utc")
    day = pd.to_datetime(s.tip_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    price = s.price.to_numpy(float)
    traded = s.filled_contracts.to_numpy(float)
    value = np.where(s.side.to_numpy() == "yes", s.over, 1 - s.over).astype(float)
    groups = day.groupby(day.to_numpy()).indices
    return [(d, price[i], traded[i], value[i]) for d, i in groups.items()]


def simulate_maker(
    days,
    bankroll=500.0,
    stake=0.01,
    day_cap=0.30,
    fee_share=0.25,
    roi_shift=0.0,
    max_dollars=25.0,
    max_contracts=100,
):
    """Replay resting orders with a bankroll.

    Every order reserves its full cost while it rests, so a day's orders
    together reserve at most ``day_cap`` of the bankroll (each is scaled down
    if needed). An order fills only as many contracts as traded through its
    price. ``roi_shift`` changes the return per dollar of every filled
    contract.
    """
    rows = []
    for day, price, traded, value in days:
        start = bankroll
        scale = min(1.0, day_cap / (stake * len(price)))
        budget = np.minimum(stake * start * scale, max_dollars)
        size = np.minimum(np.floor(budget / price + 1e-9), max_contracts)
        filled = np.minimum(size, np.floor(traded))
        raw = 100 * fee_share * 0.07 * filled * price * (1 - price)
        fee = np.ceil(np.round(raw, 6)) / 100
        profit = float(np.sum(filled * (value + roi_shift * price - price) - fee))
        bankroll = start + profit
        rows.append(
            {
                "day": day,
                "orders": int((size >= 1).sum()),
                "filled_orders": int((filled >= 1).sum()),
                "contracts": int(filled.sum()),
                "profit": profit,
                "bankroll": bankroll,
            }
        )
        if bankroll <= 1:
            break
    return pd.DataFrame(rows)
