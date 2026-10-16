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

from kalshi.paper_props import candidate_orders
from kalshi.risk import Exposure, Rejected, RiskLimits, check, commit
from kalshi.strategy import DAY_CAP, STAKE
from kalshi.trading import Order, TradingClient, TradingError

LATEST = Path("outputs/projections/latest.csv")
OUT = Path("outputs/live")
ORDERS = OUT / "orders.csv"
FILLS = OUT / "fills.csv"
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
    if FILLS.exists():
        f = pd.read_csv(FILLS)
        f = f[f.day == today] if "day" in f else f.iloc[0:0]
        realized = float(
            pd.to_numeric(f.get("profit", pd.Series(dtype=float))).fillna(0).sum()
        )
    return Exposure(
        by_day, by_game, float(live.cost.sum()) if len(live) else 0.0, realized
    )


def main():
    now = datetime.now(UTC)
    mode = os.environ.get("TRADE_MODE", "dry")
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
            resting = {o["order_id"] for o in client.orders("resting")}
            placed_mask = ledger.status == "placed"
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
    }
    (OUT / "last_run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except TradingError as error:
        print(f"::error::{error}")
        sys.exit(1)
