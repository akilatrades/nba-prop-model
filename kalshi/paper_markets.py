"""Model-free "no" seller paper tests in other Kalshi markets (RESEARCH_LOG.md, 2026-10-16).

The market scan found the NBA flow asymmetry in other series with no model
at all. Each series here gets its own $500 paper bankroll and the scan's
rule: at each run, for every open market whose expected expiration is 4 to
10 hours away (pre-game for a three-hour event) with a two-sided quote, a
"no" order rests one cent inside the ask (at the bid when the spread is one
cent) until the next run or three hours before expiration, whichever is
first. Fills come from Kalshi's public trade prints with the price-priority
rules of ``kalshi.maker``; filled orders settle from Kalshi's result.
Sizing as in the NBA tests: 1% of the bankroll per order, 30% per day, at
most $25 and 100 contracts per order, maker fee 25% of the taker fee.
"""

import math

import numpy as np
import pandas as pd

from kalshi.maker import fills, maker_fee
from kalshi.paper_props import FEE_SHARE, START_BANKROLL
from kalshi.stats import clustered_mean_t
from kalshi.strategy import DAY_CAP, MAX_CONTRACTS, MAX_ORDER_DOLLARS, STAKE

SERIES = [
    "KXITFMATCH",
    "KXNASCARRACE",
    "KXLIGAMXSPREAD",
    "KXLALIGASPREAD",
    "KXNFLANYTD",
]
WINDOW_HOURS = (4, 10)
EVENT_HOURS = 3  # orders are cancelled this long before expiration
COLUMNS = [
    "series",
    "placed_utc",
    "placed_ts",
    "cancel_ts",
    "expiration_utc",
    "day",
    "event_ticker",
    "ticker",
    "title",
    "side",
    "price",
    "improved",
    "yes_level",
    "contracts",
    "reserved",
    "status",
    "filled_contracts",
    "result",
    "fee",
    "profit",
    "p_model",
    "vetoed",
]


def empty_ledger():
    return pd.DataFrame(columns=COLUMNS)


def _day(times):
    return (
        pd.to_datetime(times, utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.astype(str)
    )


def bankroll(ledger, series, start=START_BANKROLL):
    done = ledger[
        (ledger.series == series) & ledger.status.isin(["settled", "unfilled"])
    ]
    return start + float(pd.to_numeric(done.profit).fillna(0).sum())


def quotes(markets, now):
    """Open markets in the window with a two-sided quote; prices as floats."""
    m = pd.DataFrame(markets)
    if m.empty:
        return m
    exp = pd.to_datetime(
        m.get("expected_expiration_time"), utc=True, format="ISO8601", errors="coerce"
    )
    m["expiration_utc"] = exp
    hours = (exp - pd.Timestamp(now)).dt.total_seconds() / 3600
    m["yes_bid"] = pd.to_numeric(m.yes_bid_dollars, errors="coerce")
    m["yes_ask"] = pd.to_numeric(m.yes_ask_dollars, errors="coerce")
    ok = (
        hours.between(*WINDOW_HOURS)
        & (m.yes_bid > 0)
        & (m.yes_ask < 1)
        & (m.yes_ask > m.yes_bid)
    )
    return m[ok].copy()


def no_orders(q):
    spread = (q.yes_ask - q.yes_bid).round(4)
    improve = spread >= 0.02 - 1e-9
    no_bid = 1 - q.yes_ask
    price = np.where(improve, no_bid + 0.01, no_bid).round(4)
    out = q[["ticker", "event_ticker", "title", "expiration_utc"]].copy()
    out["side"] = "no"
    out["price"] = price
    out["improved"] = improve.to_numpy()
    out["yes_level"] = (1 - price).round(4)
    return out[(out.price > 0) & (out.price < 1)]


def post_orders(
    series, markets, ledger, now, next_run_hours=3, start=START_BANKROLL, p_model=None
):
    """New orders for one series.

    ``p_model``, when given, maps ticker to a model probability of "yes";
    markets without one are skipped, and an order is vetoed (recorded with
    status "vetoed", no capital) when the model's "no" probability minus the
    order price and the maker fee is below zero.
    """
    q = quotes(markets, now)
    if q.empty:
        return empty_ledger()
    o = no_orders(q)
    o["p_model"] = np.nan
    o["vetoed"] = False
    if p_model is not None:
        o["p_model"] = o.ticker.map(p_model)
        o = o[o.p_model.notna()].copy()
        edge = (1 - o.p_model) - o.price - maker_fee(o.price, FEE_SHARE)
        o["vetoed"] = edge < 0
    mine = ledger[ledger.series == series]
    resting = mine[mine.status.isin(["resting", "vetoed"])]
    o = o[~o.ticker.isin(set(resting.ticker))]
    if o.empty:
        return empty_ledger()
    now = pd.Timestamp(now)
    o["day"] = _day(o.expiration_utc)
    bank = bankroll(ledger, series, start)
    rows = []
    for day, group in o.groupby("day"):
        open_day = mine[(mine.day == day) & mine.status.isin(["resting", "filled"])]
        reserved = float(pd.to_numeric(open_day.reserved).sum())
        room = max(DAY_CAP * bank - reserved, 0.0)
        for r in group.itertuples():
            budget = min(STAKE * bank, room, MAX_ORDER_DOLLARS)
            n = min(math.floor(budget / r.price + 1e-9), MAX_CONTRACTS)
            if n < 1 and not r.vetoed:
                continue
            if not r.vetoed:
                room -= n * r.price
            cancel = min(
                now + pd.Timedelta(hours=next_run_hours),
                r.expiration_utc - pd.Timedelta(hours=EVENT_HOURS),
            )
            rows.append(
                {
                    "series": series,
                    "placed_utc": now.isoformat(timespec="minutes"),
                    "placed_ts": int(now.timestamp()),
                    "cancel_ts": int(cancel.timestamp()),
                    "expiration_utc": r.expiration_utc.isoformat(),
                    "day": day,
                    "event_ticker": r.event_ticker,
                    "ticker": r.ticker,
                    "title": r.title,
                    "side": "no",
                    "price": float(r.price),
                    "improved": bool(r.improved),
                    "yes_level": float(r.yes_level),
                    "contracts": 0 if r.vetoed else n,
                    "reserved": 0.0 if r.vetoed else round(n * float(r.price), 2),
                    "status": "vetoed" if r.vetoed else "resting",
                    "p_model": None
                    if pd.isna(r.p_model)
                    else round(float(r.p_model), 4),
                    "vetoed": bool(r.vetoed),
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def check_fills(ledger, fetch_trades, now_ts):
    ledger = ledger.copy()
    for column in ["status", "result"]:
        ledger[column] = ledger[column].astype(object)
    due = ledger.index[
        ledger.status.isin(["resting", "vetoed"]) & (ledger.cancel_ts <= now_ts)
    ]
    for i in due:
        row = ledger.loc[[i]].rename(
            columns={"placed_ts": "end_ts", "cancel_ts": "tip_ts"}
        )
        raw = fetch_trades(
            row.ticker.iat[0], int(row.end_ts.iat[0]), int(row.tip_ts.iat[0])
        )
        traded = 0.0
        if raw:
            t = pd.DataFrame(raw)
            t["ticker"] = row.ticker.iat[0]
            t["ts"] = pd.to_datetime(t.created_time, utc=True, format="ISO8601").map(
                lambda x: int(x.timestamp())
            )
            row["improved"] = row.improved.astype(str).str.lower().eq("true")
            traded = float(fills(row, t).get(row.ticker.iat[0], 0.0))
        if ledger.at[i, "status"] == "vetoed":
            # Report only: what the vetoed order would have filled.
            ledger.at[i, "filled_contracts"] = math.floor(traded)
            ledger.at[i, "status"] = (
                "vetoed_filled" if traded >= 1 else "vetoed_unfilled"
            )
            continue
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
    ledger = ledger.copy()
    for column in ["status", "result"]:
        ledger[column] = ledger[column].astype(object)
    for i in ledger.index[ledger.status.isin(["filled", "vetoed_filled"])]:
        market = fetch_market(ledger.at[i, "ticker"])
        if not market or market.get("result") not in {"yes", "no"}:
            continue
        won = 1.0 if market["result"] == "no" else 0.0
        if ledger.at[i, "status"] == "vetoed_filled":
            ledger.at[i, "result"] = market["result"]
            ledger.at[i, "profit"] = 0.0
            ledger.at[i, "fee"] = 0.0
            ledger.at[i, "status"] = "vetoed_settled"
            continue
        n = int(ledger.at[i, "filled_contracts"])
        price = float(ledger.at[i, "price"])
        fee = math.ceil(round(100 * n * maker_fee(price, FEE_SHARE), 6)) / 100
        ledger.at[i, "result"] = market["result"]
        ledger.at[i, "fee"] = fee
        ledger.at[i, "profit"] = round(n * (won - price) - fee, 2)
        ledger.at[i, "status"] = "settled"
    return ledger


def scorecard(ledger, series, start=START_BANKROLL):
    mine = ledger[ledger.series == series]
    done = mine[mine.status.isin(["settled", "unfilled"])]
    settled = mine[mine.status == "settled"].copy()
    vetoed = mine[mine.status.str.startswith("vetoed")]
    card = {
        "orders": len(mine) - len(vetoed),
        "vetoed": len(vetoed),
        "open": int(mine.status.isin(["resting", "filled"]).sum()),
        "fill_rate": float((done.status == "settled").mean()) if len(done) else None,
        "fills_settled": len(settled),
        "bankroll": round(bankroll(ledger, series, start), 2),
    }
    if len(settled) >= 2:
        per = pd.to_numeric(settled.profit) / pd.to_numeric(settled.filled_contracts)
        mean, t = clustered_mean_t(per, settled.event_ticker)
        card["cents_per_contract"] = round(100 * float(mean), 2)
        card["t"] = round(float(t), 2) if np.isfinite(t) else None
        card["profit_dollars"] = round(float(pd.to_numeric(settled.profit).sum()), 2)
    vs = vetoed[vetoed.status == "vetoed_settled"]
    if len(vs):
        won = (vs.result == "no").astype(float)
        per = (
            won
            - pd.to_numeric(vs.price)
            - maker_fee(pd.to_numeric(vs.price), FEE_SHARE)
        )
        card["vetoed_would_have_made_cents"] = round(100 * float(per.mean()), 2)
        card["vetoed_settled"] = len(vs)
    return card
