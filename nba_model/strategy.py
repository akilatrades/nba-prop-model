"""The trading rule for 2026-27, its staking, and a bankroll simulation.

The rule came from looking at 2025-26 after the pre-registered market test
failed (RESEARCH_LOG.md, "Exploratory findings"), so its 2025-26 numbers are
in-sample. It is fixed before the 2026-27 season and judged on that season.

Rule, for each Kalshi points, rebounds, assists or threes market:

1. Price taken 2 to 6 hours before the scheduled tip-off.
2. Two-sided quote with a spread of at most 2 cents (cheap to trade).
3. Buy the side whose model probability beats its cost (ask for yes,
   1 - bid for no) plus Kalshi's taker fee by at least 5 cents.
4. One position per player, stat and game: the first time any line
   qualifies, the line with the largest edge.
5. Live only: skip players the injury report lists with any status
   (questionable, doubtful, day-to-day), since a player who sits settles at
   a value set by Kalshi rather than by the result.

Staking: each position risks 1% of the bankroll at the start of the day,
in whole contracts, capped at $25 and 100 contracts (these markets are thin:
the median market traded about 650 contracts in its life). A day's positions
together risk at most 30% of the bankroll; if more qualify, each is scaled
down. The model's claimed edges are far larger than what it realised in
2025-26, so stakes are flat rather than sized by edge.
"""

import numpy as np
import pandas as pd

from nba_model.kalshi import fee_per_contract

MAX_SPREAD = 0.02
MIN_EDGE = 0.05
WINDOW_HOURS = (2, 6)  # live: any projection run in this window before tip-off
# Replays of past seasons look at two moments, about as often as the live
# schedule sees each game inside the window: 5 and 2 hours before tip-off.
HOURS_BEFORE = (5, 2)  # earliest first
STAKE = 0.01
DAY_CAP = 0.30
# In 2025-26 the rule broke even in the regular season (+0.02 cents per
# contract over 2,493 positions) and made its money in the play-in and
# playoffs (+4.8 cents, t 3.2, 1,435 positions in 91 games). Regular-season
# positions are paper only; the 2027 play-in and playoffs are the test.
MONEY_SEASON_TYPES = ("play-in", "playoffs")
MAX_ORDER_DOLLARS = 25.0  # Kalshi prop markets are thin: median volume ~650
MAX_CONTRACTS = 100
EPS = 1e-9


def qualifying(frame, max_spread=MAX_SPREAD, min_edge=MIN_EDGE):
    """Markets that pass the spread and edge tests, with side, cost and edge."""
    f = frame.copy()
    spread = f.yes_ask - f.yes_bid
    ok = (f.yes_bid > 0) & (f.yes_ask < 1) & (spread > 0) & (spread <= max_spread + EPS)
    yes_cost, no_cost = f.yes_ask, 1 - f.yes_bid
    yes_edge = f.p_model - yes_cost - fee_per_contract(yes_cost)
    no_edge = (1 - f.p_model) - no_cost - fee_per_contract(no_cost)
    f["side"] = np.select(
        [ok & (yes_edge >= min_edge), ok & (no_edge >= min_edge)], ["yes", "no"], ""
    )
    f = f[f.side != ""].copy()
    yes = f.side == "yes"
    f["cost"] = np.where(yes, yes_cost.loc[f.index], no_cost.loc[f.index])
    f["edge"] = np.where(yes, yes_edge.loc[f.index], no_edge.loc[f.index])
    f["spread"] = spread.loc[f.index]
    return f


def one_per_player(trades, keys=("game_id", "player_id", "stat")):
    """Keep the largest-edge line for each player, stat and game."""
    keys = list(keys)
    return trades.sort_values("edge", ascending=False).drop_duplicates(keys)


def settle(trades):
    """Per-contract profit after fees (taker fee, unrounded)."""
    t = trades.copy()
    won = np.where(t.side == "yes", t.over, 1 - t.over)
    t["fee"] = fee_per_contract(t.cost)
    t["pnl"] = won - t.cost - t.fee
    t["gross_at_mid"] = np.where(t.side == "yes", t.over - t.mid, t.mid - t.over)
    return t


def historical_trades(priced, quotes_at):
    """Apply the rule to past markets.

    ``priced`` has one row per market with p_model, over, game_id, player_id,
    stat and tip_utc. ``quotes_at(hours)`` returns ticker, yes_bid, yes_ask
    as of that many hours before tip-off. A market is traded at the first
    hour (6 hours before, then 5, ...) at which any line for that player,
    stat and game qualifies.
    """
    taken, out = set(), []
    for hours in HOURS_BEFORE:
        q = priced.drop(columns=["yes_bid", "yes_ask"], errors="ignore").merge(
            quotes_at(hours)[["ticker", "yes_bid", "yes_ask"]], on="ticker"
        )
        q["mid"] = (q.yes_bid + q.yes_ask) / 2
        t = one_per_player(qualifying(q))
        key = list(zip(t.game_id, t.player_id, t.stat))
        t = t[[k not in taken for k in key]]
        taken.update(zip(t.game_id, t.player_id, t.stat))
        out.append(t.assign(hours_before=hours))
    return settle(pd.concat(out, ignore_index=True))


def order_fee(contracts, price):
    """Kalshi's taker fee for each order, rounded up to the cent."""
    raw = (
        100 * 0.07 * np.asarray(contracts) * np.asarray(price) * (1 - np.asarray(price))
    )
    return np.ceil(np.round(raw, 6)) / 100


def _days(trades):
    t = trades.sort_values("tip_utc")
    day = pd.to_datetime(t.tip_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    cost = t.cost.to_numpy(float)
    value = np.where(t.side.to_numpy() == "yes", t.over, 1 - t.over).astype(float)
    return [
        (d, cost[idx], value[idx])
        for d, idx in day.groupby(day.to_numpy()).indices.items()
    ]


def contracts_for(budget, cost):
    n = np.floor(np.minimum(budget, MAX_ORDER_DOLLARS) / np.asarray(cost) + 1e-9)
    return np.minimum(n, MAX_CONTRACTS)


def simulate(
    trades, bankroll=500.0, stake=STAKE, day_cap=DAY_CAP, roi_shift=0.0, days=None
):
    """Replay trades day by day with the staking rule.

    ``roi_shift`` changes every position's return per dollar risked (for
    example -0.03 takes three cents of profit from every dollar staked), to
    ask what happens if the edge is smaller than it was. Returns one row per
    day.
    """
    rows = []
    for day, cost, value in days if days is not None else _days(trades):
        start = bankroll
        scale = min(1.0, day_cap / (stake * len(cost)))
        n = contracts_for(stake * start * scale, cost)
        fee = order_fee(n, cost)
        profit = float(np.sum(n * (value + roi_shift * cost - cost) - fee))
        bankroll = start + profit
        rows.append(
            {
                "day": day,
                "trades": int((n >= 1).sum()),
                "contracts": int(n.sum()),
                "risked": float(np.sum(n * cost + fee)),
                "profit": profit,
                "bankroll": bankroll,
            }
        )
        if bankroll <= 1:
            break
    return pd.DataFrame(rows)


def summarize_path(days, start=500.0):
    peak = np.maximum.accumulate(np.r_[start, days.bankroll.to_numpy()])
    drawdown = (np.r_[start, days.bankroll.to_numpy()] - peak) / peak
    return {
        "end": float(days.bankroll.iloc[-1]) if len(days) else start,
        "profit": float(days.profit.sum()),
        "max_drawdown_pct": float(-drawdown.min() * 100),
        "worst_day": float(days.profit.min()) if len(days) else 0.0,
        "days": len(days),
        "trades": int(days.trades.sum()),
        "contracts": int(days.contracts.sum()),
    }


def bootstrap(trades, paths=2000, bankroll=500.0, roi_shift=0.0, seed=0):
    """Resample whole days with replacement to get a range of seasons."""
    days = _days(trades)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(paths):
        picks = rng.integers(0, len(days), len(days))
        season = [(i, days[j][1], days[j][2]) for i, j in enumerate(picks)]
        path = simulate(None, bankroll, roi_shift=roi_shift, days=season)
        out.append(summarize_path(path, bankroll))
    return pd.DataFrame(out)
