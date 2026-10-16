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
    TRADE_RULE       which "no" seller variant to trade (default no_seller;
                     no_seller_bookline once its historical test passes)
    TRADE_MARKET     nba (default) or nfl (live/nfl_trade.py builds the
                     priced frame; ledgers in outputs/live/nfl/)

Each run also (RESEARCH_LOG.md, 2026-10-16): cancels resting orders on
players whose injury-report entry changed since the order was placed or who
are now Out or Doubtful; and, in live mode, rests a sell order on any "no"
position whose market has moved at least EXIT_GAIN in our favour (the
early exit), logged in outputs/live/exits.csv.

Run from the repository root: python -m live.nba_trade
"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kalshi.api import fetch_market, snapshot
from kalshi.books import attach_latest
from kalshi.paper_props import ALL_RULES, candidate_orders
from kalshi.risk import Exposure, Rejected, RiskLimits, check, commit
from kalshi.strategy import DAY_CAP, STAKE
from kalshi.trading import Order, TradingClient, TradingError
from markets.nba.kalshi import SERIES
from markets.nba.live import injuries

LATEST = Path("outputs/projections/latest.csv")
OUT = Path("outputs/live")
ORDERS = OUT / "orders.csv"
FILLS = OUT / "fills.csv"
SETTLEMENTS = OUT / "settlements.csv"
EXITS = OUT / "exits.csv"
MARKET_DIRS = {"nba": Path("outputs/live"), "nfl": Path("outputs/live/nfl")}
MARKET_PREFIXES = {"nba": ("KXNBA",), "nfl": ("KXNFL",)}


def configure(market):
    """Point the ledger paths at this market's folder (NBA keeps outputs/live)."""
    global OUT, ORDERS, FILLS, SETTLEMENTS, EXITS
    if market not in MARKET_DIRS:
        raise TradingError(
            f"TRADE_MARKET {market!r} is not one of {sorted(MARKET_DIRS)}"
        )
    OUT = MARKET_DIRS[market]
    ORDERS, FILLS = OUT / "orders.csv", OUT / "fills.csv"
    SETTLEMENTS, EXITS = OUT / "settlements.csv", OUT / "exits.csv"


def only_market(frame, market):
    """Rows of an account-wide frame (fills, settlements) for this market's tickers."""
    if frame is None or not len(frame) or "ticker" not in frame:
        return frame
    return frame[frame.ticker.astype(str).str.startswith(MARKET_PREFIXES[market])]


def other_open_exposure(market):
    """Dollars resting or filled in the other markets' ledgers (one account)."""
    total = 0.0
    for name, folder in MARKET_DIRS.items():
        if name == market:
            continue
        path = folder / "orders.csv"
        if path.exists():
            o = pd.read_csv(path)
            if len(o):
                live = o[o.status.isin(["placed", "resting", "filled"])]
                total += float(
                    pd.to_numeric(live.cost, errors="coerce").fillna(0).sum()
                )
    return total


DEFAULT_RULE = "no_seller"
EXIT_GAIN = 0.06  # "yes" ask at least this far below the level we sold at
MIN_HOURS_TO_TIP = 0.25  # no new order inside 15 minutes of tip-off
ORDER_COLUMNS = [
    "run_utc",
    "mode",
    "env",
    "rule",
    "day",
    "game_id",
    "team",
    "player_id",
    "player",
    "stat",
    "line",
    "ticker",
    "side",
    "price",
    "contracts",
    "cost",
    "p_model",
    "p_book",
    "edge",
    "expiration_utc",
    "order_id",
    "client_order_id",
    "status",
    "rejected_reason",
]
EXIT_COLUMNS = [
    "run_utc",
    "mode",
    "env",
    "day",
    "ticker",
    "contracts",
    "entry_no_price",
    "exit_no_price",
    "yes_ask_at_exit",
    "expected_gain",
    "expiration_utc",
    "order_id",
    "client_order_id",
    "status",
    "result",
    "hold_profit_per_contract",
]


def eastern_day(ts):
    return pd.Timestamp(ts).tz_convert("America/New_York").date().isoformat()


def game_key(value):
    """NBA games are ESPN integer ids; NFL games are the event's date-and-teams code."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def exposure_from(ledger, today, extra_open=0.0):
    """Dollars committed by day and game from orders not known to be done.

    ``extra_open`` adds what other markets' ledgers have open in the same
    account, so the open-exposure cap holds across markets.
    """
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
    by_team = (
        live.groupby("team").cost.sum().to_dict()
        if len(live) and "team" in live
        else {}
    )
    return Exposure(
        by_day,
        by_game,
        (float(live.cost.sum()) if len(live) else 0.0) + float(extra_open),
        realized,
        counts,
        by_team,
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

    def cents(column):
        return pd.to_numeric(st.get(column, 0), errors="coerce").fillna(0) / 100

    st["revenue_dollars"] = cents("revenue")
    st["cost_dollars"] = cents("yes_total_cost") + cents("no_total_cost")
    st["fee_dollars"] = cents("fee_cost") if "fee_cost" in st else 0.0
    st["profit"] = st.revenue_dollars - st.cost_dollars - st.fee_dollars
    st["day"] = pd.to_datetime(st.settled_time, utc=True, format="ISO8601").map(
        eastern_day
    )
    return st


def drop_fresh_injury_news(candidates, now, hours=6):
    """Skip players whose injury-report entry changed in the last ``hours``.

    The model priced the market from the last projection run; a status that
    changed since then is exactly the news a sharp would be trading on.
    """
    try:
        report = injuries()
    except (OSError, ValueError) as error:
        print(f"Injury report unavailable ({error}); no injury filter this run")
        return candidates
    if report.empty or "updated" not in report:
        return candidates
    updated = pd.to_datetime(report.updated, utc=True, errors="coerce")
    recent = report[updated >= pd.Timestamp(now) - pd.Timedelta(hours=hours)]
    flagged = set(recent.player_id.dropna().astype("int64"))
    listed = set(report.player_id.dropna().astype("int64"))
    keep = ~candidates.player_id.astype("int64").isin(
        flagged
        | (listed & set(report.loc[report.out, "player_id"].dropna().astype("int64")))
    )
    dropped = int((~keep).sum())
    if dropped:
        print(f"Injury filter: {dropped} candidates dropped (fresh or Out status)")
    return candidates[keep]


def cancel_on_injury(ledger, client, now, report=None):
    """Cancel resting orders on players with new injury news.

    An order placed before a player's report entry changed, or on a player now
    Out or Doubtful, is cancelled: the market knows the news before the model
    does, and a resting "no" on a scratched starter's teammate is the kind of
    order a sharp sweeps. Returns the ledger with those rows marked
    ``cancelled`` and the reason, plus the number cancelled.
    """
    if not len(ledger):
        return ledger, 0
    if report is None:
        try:
            report = injuries()
        except (OSError, ValueError) as error:
            print(f"Injury report unavailable ({error}); no cancels this run")
            return ledger, 0
    if report.empty:
        return ledger, 0
    ledger = ledger.copy()
    rep = report.dropna(subset=["player_id"]).copy()
    rep["player_id"] = rep.player_id.astype("int64")
    rep["updated"] = pd.to_datetime(rep.get("updated"), utc=True, errors="coerce")
    rep["status_l"] = (
        rep.get("status", pd.Series("", index=rep.index)).fillna("").str.lower()
    )
    rep = rep.sort_values("updated").drop_duplicates("player_id", keep="last")
    rep = rep.set_index("player_id")
    resting = ledger.status.isin(["placed", "resting", "dry"])
    cancelled = 0
    for i in ledger.index[resting]:
        pid = ledger.at[i, "player_id"]
        if pd.isna(pid) or int(pid) not in rep.index:
            continue
        entry = rep.loc[int(pid)]
        placed = pd.to_datetime(ledger.at[i, "run_utc"], utc=True, errors="coerce")
        out = bool(entry.get("out", False)) or entry.status_l in {"out", "doubtful"}
        changed = (
            pd.notna(entry.updated) and pd.notna(placed) and entry.updated > placed
        )
        if not (out or changed):
            continue
        reason = "injury: " + (entry.status_l or "entry changed")
        try:
            client.cancel(str(ledger.at[i, "order_id"]))
        except TradingError as error:
            print(f"Cancel failed for {ledger.at[i, 'ticker']}: {error}")
            continue
        ledger.at[i, "status"] = "cancelled"
        ledger.at[i, "rejected_reason"] = reason
        cancelled += 1
    if cancelled:
        print(f"Injury cancels: {cancelled} resting orders pulled")
    return ledger, cancelled


def entry_prices(fills):
    """Average "no" price paid per ticker from the account's fills."""
    if fills is None or not len(fills):
        return {}
    f = fills.copy()
    if not {"side", "no_price", "count", "ticker"} <= set(f.columns):
        return {}
    action = f["action"] if "action" in f else pd.Series("buy", index=f.index)
    f = f[(f.side == "no") & (action.fillna("buy") == "buy")]
    if f.empty:
        return {}
    f["no_price"] = pd.to_numeric(f.no_price, errors="coerce") / 100
    f["count"] = pd.to_numeric(f["count"], errors="coerce")
    g = f.groupby("ticker").apply(
        lambda d: float((d.no_price * d["count"]).sum() / d["count"].sum())
    )
    return g.to_dict()


def exit_orders(positions, quotes, entries, resting_sells, now, min_gain=EXIT_GAIN):
    """Sell orders for "no" positions whose market has moved our way.

    ``positions``: Kalshi market positions (market_ticker, position; negative
    is "no" contracts). ``quotes``: frame with ticker, yes_bid, yes_ask,
    tip_utc. ``entries``: ticker -> average "no" price paid.
    ``resting_sells``: ticker -> contracts already resting to sell.

    The exit rests 1 cent inside the "no" bid (post-only, maker), expiring at
    tip-off, when the "yes" ask has fallen at least ``min_gain`` below the
    level we sold at (yes level = 1 - entry no price).
    """
    q = quotes.set_index("ticker") if len(quotes) else quotes
    out = []
    for p in positions:
        ticker = p.get("market_ticker") or p.get("ticker")
        held = -int(p.get("position", 0) or 0)  # "no" contracts
        if held <= 0 or ticker not in getattr(q, "index", []):
            continue
        entry = entries.get(ticker)
        if entry is None:
            continue
        yes_bid, yes_ask = (
            float(q.at[ticker, "yes_bid"]),
            float(q.at[ticker, "yes_ask"]),
        )
        if not (0 < yes_bid < yes_ask < 1):
            continue
        level = 1 - entry
        if yes_ask > level - min_gain + 1e-9:
            continue
        no_bid = 1 - yes_ask
        no_ask = 1 - yes_bid
        price = no_bid + 0.01 if no_ask - no_bid >= 0.02 - 1e-9 else no_bid
        free = held - int(resting_sells.get(ticker, 0))
        if free <= 0:
            continue
        tip = pd.Timestamp(q.at[ticker, "tip_utc"])
        out.append(
            {
                "ticker": ticker,
                "contracts": free,
                "entry_no_price": round(entry, 4),
                "exit_no_price": round(price, 2),
                "yes_ask_at_exit": round(yes_ask, 2),
                "expected_gain": round((price - entry) * free, 2),
                "expiration_utc": tip.isoformat(),
                "expiration_ts": int(tip.timestamp()) if pd.notna(tip) else None,
            }
        )
    return out


def place_exits(client, candidates, now, mode):
    rows = []
    for c in candidates:
        row = {
            "run_utc": now.isoformat(timespec="minutes"),
            "mode": mode,
            "env": client.env,
            "day": eastern_day(now),
            **{k: v for k, v in c.items() if k != "expiration_ts"},
        }
        try:
            order = Order(
                ticker=c["ticker"],
                side="no",
                count=int(c["contracts"]),
                price_cents=round(c["exit_no_price"] * 100),
                expiration_ts=c["expiration_ts"],
                action="sell",
            )
            result = client.place(order)
            row.update(
                order_id=result.get("order_id"),
                client_order_id=order.client_order_id,
                status="placed" if mode == "live" else "dry",
            )
        except TradingError as error:
            row.update(order_id="", client_order_id="", status=f"rejected: {error}")
        rows.append(row)
    return pd.DataFrame(rows, columns=EXIT_COLUMNS)


def score_exits(exits, fetch=fetch_market):
    """Fill in each exit's market result and what holding would have made."""
    if not len(exits):
        return exits
    exits = exits.copy()
    exits["result"] = exits.result.astype(object)
    exits["hold_profit_per_contract"] = pd.to_numeric(
        exits.hold_profit_per_contract, errors="coerce"
    )
    for i in exits.index[exits.result.isna() | (exits.result.astype(str) == "")]:
        try:
            market = fetch(exits.at[i, "ticker"])
        except (OSError, ValueError):
            continue
        result = (market or {}).get("result")
        if result not in {"yes", "no"}:
            continue
        exits.at[i, "result"] = result
        entry = float(exits.at[i, "entry_no_price"])
        exits.at[i, "hold_profit_per_contract"] = round(
            (1.0 if result == "no" else 0.0) - entry, 4
        )
    return exits


def reconcile(ledger, client, market="nba"):
    """Compare the ledger's resting orders with the exchange's; flag any difference.

    Only this market's tickers are compared: the account also carries the
    other markets' orders, which live in their own ledgers.
    """
    exchange = {
        o["order_id"]: o
        for o in client.orders("resting")
        if str(o.get("ticker", "")).startswith(MARKET_PREFIXES[market])
        or "ticker" not in o
    }
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
    rule = os.environ.get("TRADE_RULE", DEFAULT_RULE)
    market = os.environ.get("TRADE_MARKET", "nba")
    if rule not in ALL_RULES:
        raise TradingError(f"TRADE_RULE {rule!r} is not one of {ALL_RULES}")
    if market != "nba":
        configure(market)
    summary_extra = {}
    client = TradingClient(dry_run=(mode != "live"))
    OUT.mkdir(parents=True, exist_ok=True)
    limits = RiskLimits()
    ledger = (
        pd.read_csv(ORDERS) if ORDERS.exists() else pd.DataFrame(columns=ORDER_COLUMNS)
    )
    for column in ORDER_COLUMNS:
        if column not in ledger:
            ledger[column] = pd.NA

    if limits.kill_file.exists():
        print("KILL file present: cancelling everything, placing nothing")
        if client.can_sign:
            print(json.dumps(client.cancel_all()))
        return

    cancelled = 0
    if market == "nfl":
        from live.nfl_trade import candidates as nfl_candidates

        priced = nfl_candidates(now)
        if priced.empty:
            print("No priced markets yet")
            return
    else:
        if not LATEST.exists():
            print("No priced markets yet")
            return
        priced = pd.read_csv(LATEST, low_memory=False)
        priced["tip_utc"] = pd.to_datetime(priced.tip_utc, utc=True)
        priced["hours_to_tip"] = (
            priced.tip_utc - pd.Timestamp(now)
        ).dt.total_seconds() / 3600
        priced = refresh_quotes(priced, now)
        priced = attach_latest(priced)
        ledger, cancelled = cancel_on_injury(ledger, client, now)
        if cancelled:
            ledger.to_csv(ORDERS, index=False)
    candidates = candidate_orders(priced, rule)
    candidates = candidates[candidates.hours_to_tip >= MIN_HOURS_TO_TIP]
    if candidates.empty:
        print("No candidate orders in the window")
        return
    if market == "nba":
        candidates = drop_fresh_injury_news(candidates, now)
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
    exposure = exposure_from(ledger, today, other_open_exposure(market))
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
            "rule": rule,
            "day": day,
            "game_id": game_key(r.game_id),
            "team": r.team,
            "player_id": int(r.player_id),
            "player": r.player,
            "stat": r.stat,
            "line": r.line,
            "ticker": r.ticker,
            "side": r.side,
            "price": round(float(r.price), 2),
            "contracts": n,
            "cost": round(n * float(r.price), 2),
            "p_model": round(float(r.p_model), 4),
            "p_book": round(float(r.p_book), 4)
            if hasattr(r, "p_book") and pd.notna(r.p_book)
            else pd.NA,
            "edge": round(float(r.edge), 4),
            "expiration_utc": pd.Timestamp(r.tip_utc).isoformat(),
        }
        if n < 1:
            continue
        try:
            check(
                row["cost"], day, row["game_id"], limits, exposure, placed, team=r.team
            )
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
            commit(exposure, row["cost"], day, row["game_id"], team=r.team)
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
            fills = only_market(
                pd.DataFrame(client.fills(min_ts=int(now.timestamp()) - 3 * 86400)),
                market,
            )
            if len(fills):
                fills["day"] = pd.to_datetime(
                    fills.created_time, utc=True, format="ISO8601"
                ).map(eastern_day)
                old = pd.read_csv(FILLS) if FILLS.exists() else pd.DataFrame()
                fills = pd.concat([old, fills], ignore_index=True)
                if "trade_id" in fills:
                    fills = fills.drop_duplicates("trade_id")
                fills.to_csv(FILLS, index=False)
            settled = only_market(
                settlement_frame(
                    client.settlements(min_ts=int(now.timestamp()) - 30 * 86400)
                ),
                market,
            )
            if len(settled):
                settled.to_csv(SETTLEMENTS, index=False)
            report, exchange = reconcile(ledger, client, market)
            summary_extra = {
                "reconcile": {
                    k: (v if not isinstance(v, list) else len(v))
                    for k, v in report.items()
                }
            }
            # Early exits on "no" positions that have moved our way.
            exits = (
                pd.read_csv(EXITS)
                if EXITS.exists()
                else pd.DataFrame(columns=EXIT_COLUMNS)
            )
            resting_sells = {}
            for o in exchange.values():
                if o.get("action") == "sell" and o.get("side") == "no":
                    t = o.get("ticker")
                    resting_sells[t] = resting_sells.get(t, 0) + int(
                        o.get("remaining_count", o.get("count", 0)) or 0
                    )
            quotes = priced.rename(
                columns={"yes_bid_dollars": "yes_bid", "yes_ask_dollars": "yes_ask"}
            )[["ticker", "yes_bid", "yes_ask", "tip_utc"]]
            wanted = exit_orders(
                client.positions(),
                quotes,
                entry_prices(pd.read_csv(FILLS) if FILLS.exists() else None),
                resting_sells,
                now,
            )
            new_exits = place_exits(client, wanted, now, mode)
            exits = pd.concat([exits, new_exits], ignore_index=True)[EXIT_COLUMNS]
            exits = score_exits(exits)
            exits.to_csv(EXITS, index=False)
            summary_extra["exits_placed"] = int(len(new_exits))
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
        "rule": rule,
        "market": market,
        "bankroll": round(bank, 2),
        "injury_cancels": cancelled,
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
        f"Updated {now:%Y-%m-%d %H:%M} UTC. Risk limits: kalshi/risk.py. "
        f"Rule: {os.environ.get('TRADE_RULE', DEFAULT_RULE)}.",
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
