"""Live (or dry-run) trading of the NBA "no" seller rule.

Reads the latest priced markets from the projection run
(outputs/projections/latest.csv), builds the "no" seller's orders with the
same rule as the paper ledger (kalshi/paper_props.py, rule "no_seller"),
sizes them from the real account balance, passes each through the risk
limits, and places them on Kalshi with an expiry at tip-off. Then records
the account's fills and settlements in outputs/live/.

Modes, from the environment:

    TRADE_MODE=dry   (default) build and log orders; place nothing
    TRADE_MODE=live  place orders; KALSHI_ENV picks demo or prod

Run from the repository root: python -m live.nba_trade
"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kalshi.api import snapshot
from kalshi.paper_props import candidate_orders
from kalshi.risk import Exposure, Rejected, RiskLimits, check, commit
from kalshi.strategy import DAY_CAP, STAKE
from kalshi.trading import Order, TradingClient, TradingError
from markets.nba.kalshi import SERIES

LATEST = Path("outputs/projections/latest.csv")
OUT = Path("outputs/live")
ORDERS = OUT / "orders.csv"
FILLS = OUT / "fills.csv"
SETTLEMENTS = OUT / "settlements.csv"
RULE = "no_seller"
ORDER_COLUMNS = [
    "run_utc",
    "mode",
    "env",
    "day",
    "game_id",
    "player",
    "stat",
    "line",
    "ticker",
    "side",
    "price",
    "contracts",
    "cost",
    "p_model",
    "edge",
    "expiration_utc",
    "order_id",
    "client_order_id",
    "status",
    "rejected_reason",
]


def eastern_day(ts):
    return pd.Timestamp(ts).tz_convert("America/New_York").date().isoformat()


def exposure_from(ledger, today):
    """Dollars committed by day and game from orders not known to be done."""
    live = (
        ledger[ledger.status.isin(["placed", "resting", "filled"])]
        if len(ledger)
        else ledger
    )
    by_day = live.groupby("day").cost.sum().to_dict() if len(live) else {}
    by_game = live.groupby("game_id").cost.sum().to_dict() if len(live) else {}
    realized = 0.0
    if SETTLEMENTS.exists():
        st = pd.read_csv(SETTLEMENTS)
        if "day" in st and "profit" in st:
            realized = float(
                pd.to_numeric(st.loc[st.day == today, "profit"]).fillna(0).sum()
            )
    counts = live.groupby("game_id").size().to_dict() if len(live) else {}
    return Exposure(
        by_day, by_game, float(live.cost.sum()) if len(live) else 0.0, realized, counts
    )


def refresh_quotes(priced, now):
    """Replace the projection run's quotes with the exchange's current ones.

    The model probabilities come from the last projection run (up to three
    hours old); the bid and ask must be current, or an order could be posted
    through a price that has moved. Markets no longer open are dropped.
    """
    live = snapshot(SERIES, now=now)
    if live.empty:
        return priced.iloc[0:0]
    for column in ["yes_bid_dollars", "yes_ask_dollars"]:
        live[column] = pd.to_numeric(live[column], errors="coerce")
    quotes = live[["ticker", "yes_bid_dollars", "yes_ask_dollars"]]
    merged = priced.drop(
        columns=["yes_bid_dollars", "yes_ask_dollars"], errors="ignore"
    )
    merged = merged.merge(quotes, on="ticker", how="inner")
    print(f"Quotes refreshed: {len(merged)} of {len(priced)} priced markets still open")
    return merged


def settlement_frame(raw):
    """Kalshi settlements as one row per market with profit in dollars."""
    st = pd.DataFrame(raw)
    if st.empty:
        return st
    cents = lambda c: pd.to_numeric(st.get(c, 0), errors="coerce").fillna(0) / 100
    st["revenue_dollars"] = cents("revenue")
    st["cost_dollars"] = cents("yes_total_cost") + cents("no_total_cost")
    st["fee_dollars"] = cents("fee_cost") if "fee_cost" in st else 0.0
    st["profit"] = st.revenue_dollars - st.cost_dollars - st.fee_dollars
    st["day"] = pd.to_datetime(st.settled_time, utc=True, format="ISO8601").map(
        eastern_day
    )
    return st


def reconcile(ledger, client):
    """Compare the ledger's resting orders with the exchange's; flag any difference."""
    exchange = {o["order_id"]: o for o in client.orders("resting")}
    mine = ledger[ledger.status.isin(["placed", "resting"])]
    ours = set(mine.order_id.dropna().astype(str))
    unknown = sorted(set(exchange) - ours)
    missing = sorted(ours - set(exchange))
    report = {
        "exchange_resting": len(exchange),
        "ledger_resting": len(ours),
        "on_exchange_not_in_ledger": unknown,
        "in_ledger_not_on_exchange": len(missing),
    }
    if unknown:
        print(
            f"::warning::{len(unknown)} resting orders on the exchange are not in the ledger: {unknown[:5]}"
        )
    return report, exchange


def main():
    now = datetime.now(UTC)
    mode = os.environ.get("TRADE_MODE", "dry")
    summary_extra = {}
    client = TradingClient(dry_run=(mode != "live"))
    OUT.mkdir(parents=True, exist_ok=True)
    limits = RiskLimits()
    ledger = (
        pd.read_csv(ORDERS) if ORDERS.exists() else pd.DataFrame(columns=ORDER_COLUMNS)
    )

    if limits.kill_file.exists():
        print("KILL file present: cancelling everything, placing nothing")
        if client.can_sign:
            print(json.dumps(client.cancel_all()))
        return

    if not LATEST.exists():
        print("No priced markets yet")
        return
    priced = pd.read_csv(LATEST, low_memory=False)
    priced["tip_utc"] = pd.to_datetime(priced.tip_utc, utc=True)
    priced["hours_to_tip"] = (
        priced.tip_utc - pd.Timestamp(now)
    ).dt.total_seconds() / 3600
    priced = refresh_quotes(priced, now)
    candidates = candidate_orders(priced, RULE)
    if candidates.empty:
        print("No candidate orders in the window")
        return
    candidates = candidates.sort_values("edge", ascending=False)
    # Skip markets we already have an order in (any status but rejected).
    have = (
        set(ledger.loc[~ledger.status.isin(["rejected"]), "ticker"])
        if len(ledger)
        else set()
    )
    candidates = candidates[~candidates.ticker.isin(have)]

    # Sizing from the real balance, same fractions as the paper rule.
    try:
        bank = (
            client.balance()["balance"]
            if client.can_sign
            else float(os.environ.get("DRY_BANKROLL", "250"))
        )
    except TradingError as error:
        print(f"Could not read balance: {error}")
        return
    today = eastern_day(now)
    exposure = exposure_from(ledger, today)
    rows, placed = [], 0
    for r in candidates.itertuples():
        day = eastern_day(r.tip_utc)
        night_room = max(DAY_CAP * bank - exposure.by_day.get(day, 0.0), 0.0)
        budget = min(STAKE * bank, night_room, limits.max_order)
        n = int(budget // r.price)
        row = {
            "run_utc": now.isoformat(timespec="minutes"),
            "mode": mode,
            "env": client.env,
            "day": day,
            "game_id": int(r.game_id),
            "player": r.player,
            "stat": r.stat,
            "line": r.line,
            "ticker": r.ticker,
            "side": r.side,
            "price": round(float(r.price), 2),
            "contracts": n,
            "cost": round(n * float(r.price), 2),
            "p_model": round(float(r.p_model), 4),
            "edge": round(float(r.edge), 4),
            "expiration_utc": pd.Timestamp(r.tip_utc).isoformat(),
        }
        if n < 1:
            continue
        try:
            check(row["cost"], day, row["game_id"], limits, exposure, placed)
            order = Order(
                ticker=r.ticker,
                side=r.side,
                count=n,
                price_cents=round(float(r.price) * 100),
                expiration_ts=int(pd.Timestamp(r.tip_utc).timestamp()),
            )
            result = client.place(order)
            row.update(
                order_id=result.get("order_id"),
                client_order_id=order.client_order_id,
                status="placed" if mode == "live" else "dry",
                rejected_reason="",
            )
            commit(exposure, row["cost"], day, row["game_id"])
            placed += 1
        except (Rejected, TradingError) as error:
            row.update(
                order_id="",
                client_order_id="",
                status="rejected",
                rejected_reason=str(error),
            )
        rows.append(row)
    new = pd.DataFrame(rows, columns=ORDER_COLUMNS)
    ledger = pd.concat([ledger, new], ignore_index=True)[ORDER_COLUMNS]
    ledger.to_csv(ORDERS, index=False)

    # Record fills and the current state of resting orders.
    if client.can_sign:
        try:
            fills = pd.DataFrame(client.fills(min_ts=int(now.timestamp()) - 3 * 86400))
            if len(fills):
                fills["day"] = pd.to_datetime(
                    fills.created_time, utc=True, format="ISO8601"
                ).map(eastern_day)
                old = pd.read_csv(FILLS) if FILLS.exists() else pd.DataFrame()
                fills = pd.concat([old, fills], ignore_index=True)
                if "trade_id" in fills:
                    fills = fills.drop_duplicates("trade_id")
                fills.to_csv(FILLS, index=False)
            settled = settlement_frame(
                client.settlements(min_ts=int(now.timestamp()) - 30 * 86400)
            )
            if len(settled):
                settled.to_csv(SETTLEMENTS, index=False)
            report, exchange = reconcile(ledger, client)
            summary_extra = {
                "reconcile": {
                    k: (v if not isinstance(v, list) else len(v))
                    for k, v in report.items()
                }
            }
            resting = set(exchange)
            placed_mask = ledger.status.isin(["placed", "resting"])
            ledger.loc[placed_mask & ~ledger.order_id.isin(resting), "status"] = "done"
            ledger.loc[placed_mask & ledger.order_id.isin(resting), "status"] = (
                "resting"
            )
            ledger.to_csv(ORDERS, index=False)
        except TradingError as error:
            print(f"Could not read fills/orders: {error}")

    summary = {
        "run_utc": now.isoformat(timespec="minutes"),
        "mode": mode,
        "env": client.env,
        "bankroll": round(bank, 2),
        "candidates": len(candidates),
        "placed": placed,
        "rejected": int((new.status == "rejected").sum()),
        "rejections": new.loc[new.status == "rejected", "rejected_reason"]
        .value_counts()
        .to_dict(),
        "open_exposure": round(exposure.open_total, 2),
        **summary_extra,
    }
    (OUT / "last_run.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_scorecard(ledger, now)
    print(json.dumps(summary, indent=2))


def write_scorecard(ledger, now):
    """outputs/live/scorecard.md: orders by status, fills and realized P&L."""
    lines = [
        "# Live trader",
        "",
        f'Updated {now:%Y-%m-%d %H:%M} UTC. Risk limits: kalshi/risk.py. Rule: the NBA "no" seller.',
        "",
        "| Status | Orders | Dollars |",
        "|---|---:|---:|",
    ]
    if len(ledger):
        g = ledger.groupby("status").agg(n=("ticker", "size"), cost=("cost", "sum"))
        for status, r in g.iterrows():
            lines.append(f"| {status} | {int(r.n)} | ${r.cost:,.2f} |")
        reasons = ledger.loc[
            ledger.status == "rejected", "rejected_reason"
        ].value_counts()
        if len(reasons):
            lines += ["", "Rejections:", ""] + [
                f"- {k}: {v}" for k, v in reasons.items()
            ]
    if SETTLEMENTS.exists():
        st = pd.read_csv(SETTLEMENTS)
        if len(st):
            by_day = st.groupby("day").profit.sum()
            lines += ["", "| Day | Realized P&L |", "|---|---:|"]
            lines += [f"| {d} | ${p:,.2f} |" for d, p in by_day.tail(14).items()]
            lines += [
                "",
                f"Total realized: ${st.profit.sum():,.2f} over {len(st)} settled markets.",
            ]
    (OUT / "scorecard.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        main()
    except TradingError as error:
        print(f"::error::{error}")
        sys.exit(1)
