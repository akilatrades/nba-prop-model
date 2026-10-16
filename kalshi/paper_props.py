"""Live paper tests of the two resting-order rules (RESEARCH_LOG.md, 2026-10-15).

Each projection run posts simulated orders for games 2 to 6 hours away:

- ``model_side``: on the side the model favours, if its edge after the maker
  fee is at least 3 cents; one order per player, stat and game.
- ``no_seller``: a "no" order on every line, unless the model rates it below
  zero after the maker fee.

Orders rest until tip-off. Afterwards each is checked against Kalshi's public
trade prints (it fills with as many contracts as traded through its price,
with the backtest's price-priority rules) and filled positions are settled
from Kalshi's settlement. Each rule has its own $500 paper bankroll.
"""

import math

import numpy as np
import pandas as pd

from kalshi.maker import fills, maker_fee, orders
from kalshi.stats import clustered_mean_t
from kalshi.strategy import (
    DAY_CAP,
    MAX_CONTRACTS,
    MAX_ORDER_DOLLARS,
    STAKE,
    WINDOW_HOURS,
    one_per_player,
)

RULES = ("model_side", "no_seller")

# v2 variants of the "no" seller (RESEARCH_LOG.md, 2026-10-16), each its own
# $500 paper bankroll, run beside v1 and judged against it in February:
#   late:       posted only in the last 2 hours before tip-off (0.5 to 2 h)
#   edge_sized: stake scales with the model's edge (0.5% to 1.5%)
#   veto2:      skipped unless the model's edge is at least 2 cents
#   stat_veto:  rebounds need 2 cents of edge; other stats zero
V2_RULES = {
    "no_seller_late": {"window": (0.5, 2.0), "min_edge": 0.0, "sizing": "flat"},
    "no_seller_edge_sized": {"window": WINDOW_HOURS, "min_edge": 0.0, "sizing": "edge"},
    "no_seller_veto2": {"window": WINDOW_HOURS, "min_edge": 0.02, "sizing": "flat"},
    "no_seller_stat_veto": {
        "window": WINDOW_HOURS,
        "min_edge": {"rebounds": 0.02},
        "sizing": "flat",
    },
    # bookline: on lines the sportsbooks quote, the books' de-vigged
    # probability is the bouncer (sell "no" only if the "yes" ask is above it,
    # after the maker fee); lines the books do not quote fall back to the
    # model veto, as in v1. Needs p_book on the priced frame (kalshi/books.py).
    "no_seller_bookline": {
        "window": WINDOW_HOURS,
        "min_edge": 0.0,
        "sizing": "flat",
        "book": True,
    },
}
ALL_RULES = RULES + tuple(V2_RULES)


def stake_fraction(edge, sizing):
    """Flat 1%, or 0.5% to 1.5% rising with the model's edge (full at 5 cents)."""
    if sizing != "edge":
        return STAKE
    return 0.5 * STAKE + min(max(float(edge), 0.0), 0.05) / 0.05 * STAKE


START_BANKROLL = 500.0
FEE_SHARE = 0.25
COLUMNS = [
    "rule",
    "placed_utc",
    "placed_ts",
    "tip_utc",
    "tip_ts",
    "day",
    "game_id",
    "player_id",
    "player",
    "team",
    "stat",
    "line",
    "ticker",
    "side",
    "price",
    "improved",
    "yes_level",
    "p_model",
    "edge",
    "contracts",
    "reserved",
    "status",
    "filled_contracts",
    "result",
    "settlement_value",
    "fee",
    "profit",
    "p_book",
    "book_n",
]


def empty_ledger():
    return pd.DataFrame(columns=COLUMNS)


def _day(times):
    return (
        pd.to_datetime(times, utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.astype(str)
    )


def bankroll(ledger, rule, start=START_BANKROLL):
    done = ledger[(ledger.rule == rule) & ledger.status.isin(["settled", "unfilled"])]
    return start + float(pd.to_numeric(done.profit).fillna(0).sum())


def candidate_orders(priced, rule):
    """Orders a rule would post from one run's prices (before sizing)."""
    spec = V2_RULES.get(
        rule, {"window": WINDOW_HOURS, "min_edge": 0.0, "sizing": "flat"}
    )
    q = priced.rename(
        columns={"yes_bid_dollars": "yes_bid", "yes_ask_dollars": "yes_ask"}
    ).copy()
    lo, hi = spec["window"]
    q = q[q.hours_to_tip.between(lo, hi) & q.injury_status.fillna("").eq("")]
    if q.empty:
        return q
    if rule == "model_side":
        return one_per_player(orders(q, min_edge=0.03, fee_share=FEE_SHARE))
    o = orders(
        q.assign(p_real=q.p_model, p_model=0.0), min_edge=-1, fee_share=FEE_SHARE
    )
    o["p_model"] = o.p_real
    o["edge"] = (1 - o.p_model) - o.price - maker_fee(o.price, FEE_SHARE)
    threshold = spec["min_edge"]
    if isinstance(threshold, dict):
        need = o.stat.map(threshold).fillna(0.0)
    else:
        need = threshold
    if spec.get("book"):
        p_book = pd.to_numeric(o.get("p_book"), errors="coerce")
        if p_book is None or not isinstance(p_book, pd.Series):
            p_book = pd.Series(np.nan, index=o.index)
        covered = p_book.notna()
        book_edge = (1 - p_book) - o.price - maker_fee(o.price, FEE_SHARE)
        # On covered lines the book decides and its edge is recorded as the
        # order's edge; on the rest the model decides, as in v1.
        o["edge"] = np.where(covered, book_edge, o.edge)
        return o[(covered & (book_edge >= need)) | (~covered & (o.edge >= need))]
    return o[o.edge >= need]


def post_orders(priced, ledger, now, rule, start=START_BANKROLL):
    """Size and record the rule's new orders; markets already ordered are skipped."""
    o = candidate_orders(priced, rule)
    if o.empty:
        return empty_ledger()
    mine = ledger[ledger.rule == rule]
    if rule == "model_side":
        taken = set(zip(mine.game_id, mine.player_id, mine.stat))
        keep = [k not in taken for k in zip(o.game_id, o.player_id, o.stat)]
        o = o[np.array(keep, dtype=bool)]
    else:
        o = o[~o.ticker.isin(set(mine.ticker))]
    if o.empty:
        return empty_ledger()
    o = o.sort_values("edge", ascending=False).copy()
    o["day"] = _day(o.tip_utc)
    bank = bankroll(ledger, rule, start)
    rows = []
    for day, group in o.groupby("day"):
        reserved = float(pd.to_numeric(mine.loc[mine.day == day, "reserved"]).sum())
        room = max(DAY_CAP * bank - reserved, 0.0)
        sizing = V2_RULES.get(rule, {}).get("sizing", "flat")
        for r in group.itertuples():
            budget = min(stake_fraction(r.edge, sizing) * bank, room, MAX_ORDER_DOLLARS)
            n = min(math.floor(budget / r.price + 1e-9), MAX_CONTRACTS)
            if n < 1:
                continue
            room -= n * r.price
            tip = pd.Timestamp(r.tip_utc)
            rows.append(
                {
                    "rule": rule,
                    "placed_utc": pd.Timestamp(now).isoformat(timespec="minutes"),
                    "placed_ts": int(pd.Timestamp(now).timestamp()),
                    "tip_utc": tip.isoformat(),
                    "tip_ts": int(tip.timestamp()),
                    "day": day,
                    "game_id": int(r.game_id),
                    "player_id": int(r.player_id),
                    "player": r.player,
                    "team": r.team,
                    "stat": r.stat,
                    "line": r.line,
                    "ticker": r.ticker,
                    "side": r.side,
                    "price": round(float(r.price), 4),
                    "improved": bool(r.improved),
                    "yes_level": round(float(r.yes_level), 4),
                    "p_model": round(float(r.p_model), 4),
                    "edge": round(float(r.edge), 4),
                    "contracts": n,
                    "reserved": round(n * float(r.price), 2),
                    "status": "resting",
                    "p_book": round(float(r.p_book), 4)
                    if hasattr(r, "p_book") and pd.notna(r.p_book)
                    else np.nan,
                    "book_n": int(r.book_n)
                    if hasattr(r, "book_n") and pd.notna(r.book_n)
                    else 0,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def check_fills(ledger, fetch_trades, now_ts):
    """Mark resting orders whose game has started as filled or unfilled.

    ``fetch_trades(ticker, min_ts, max_ts)`` returns trade dicts with
    created_time, yes_price, count and taker_outcome_side.
    """
    ledger = ledger.copy()
    for column in ["status", "result"]:
        ledger[column] = ledger[column].astype(object)
    due = ledger.index[(ledger.status == "resting") & (ledger.tip_ts <= now_ts)]
    for i in due:
        row = ledger.loc[[i]].rename(columns={"placed_ts": "end_ts"})
        raw = fetch_trades(row.ticker.iat[0], row.end_ts.iat[0], row.tip_ts.iat[0])
        traded = 0.0
        if raw:
            t = pd.DataFrame(raw)
            t["ts"] = pd.to_datetime(t.created_time, utc=True, format="ISO8601").map(
                lambda x: int(x.timestamp())
            )
            row["improved"] = row.improved.astype(str).str.lower().eq("true")
            traded = float(fills(row, t).get(row.ticker.iat[0], 0.0))
        filled = min(int(ledger.at[i, "contracts"]), math.floor(traded))
        ledger.at[i, "filled_contracts"] = filled
        if filled > 0:
            ledger.at[i, "status"] = "filled"
        else:
            ledger.at[i, "status"] = "unfilled"
            ledger.at[i, "profit"] = 0.0
            ledger.at[i, "fee"] = 0.0
    return ledger


def settle(ledger, fetch_market):
    """Settle filled orders once Kalshi has settled their market."""
    ledger = ledger.copy()
    for column in ["status", "result"]:
        ledger[column] = ledger[column].astype(object)
    for i in ledger.index[ledger.status == "filled"]:
        market = fetch_market(ledger.at[i, "ticker"])
        if not market or market.get("result") not in {"yes", "no", "scalar"}:
            continue
        value = {"yes": 1.0, "no": 0.0}.get(
            market["result"], float(market.get("settlement_value_dollars") or 0.0)
        )
        side_value = value if ledger.at[i, "side"] == "yes" else 1 - value
        n = int(ledger.at[i, "filled_contracts"])
        price = float(ledger.at[i, "price"])
        fee = (
            math.ceil(round(100 * FEE_SHARE * 0.07 * n * price * (1 - price), 6)) / 100
        )
        ledger.at[i, "result"] = market["result"]
        ledger.at[i, "settlement_value"] = value
        ledger.at[i, "fee"] = fee
        ledger.at[i, "profit"] = round(n * (side_value - price) - fee, 2)
        ledger.at[i, "status"] = "settled"
    return ledger


def scorecard(ledger, rule, start=START_BANKROLL):
    mine = ledger[ledger.rule == rule]
    done = mine[mine.status.isin(["settled", "unfilled"])]
    settled = mine[mine.status == "settled"].copy()
    card = {
        "orders": len(mine),
        "resting_or_awaiting_result": int(
            mine.status.isin(["resting", "filled"]).sum()
        ),
        "fill_rate": float((done.status == "settled").mean()) if len(done) else None,
        "fills_settled": len(settled),
        "bankroll": round(bankroll(ledger, rule, start), 2),
    }
    if len(settled) >= 2:
        per_contract = pd.to_numeric(settled.profit) / pd.to_numeric(
            settled.filled_contracts
        )
        mean, t = clustered_mean_t(per_contract, settled.game_id)
        card["cents_per_contract"] = round(100 * float(mean), 2)
        card["t"] = round(float(t), 2) if np.isfinite(t) else None
        card["profit_dollars"] = round(float(pd.to_numeric(settled.profit).sum()), 2)
    return card
