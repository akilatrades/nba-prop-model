"""Execution quality of the live trader's real orders (RESEARCH_LOG.md,
2026-10-17, "Execution accuracy").

Two things, both cheap:

- ``log_queue_positions``: every QUEUE_EVERY_MINUTES, ask the exchange how
  many contracts rest ahead of each of our orders (one request per resting
  order) and append to <market folder>/queue.csv. The first reading after
  placement is the order's starting queue position.
- ``execution_report``: one row per real order joining the ledger, the
  queue log and the exchange fills: minutes from signal to placement, the
  starting queue position, minutes to first fill, contracts filled against
  ordered, and realised profit where settled. Written to execution.csv and
  execution.md in the market folder; the dashboard reads the csv.

The paper simulator fills an order when a print trades through its price;
the real exchange fills it when the queue ahead has been eaten too. The
gap between the two is what this measures.
"""

import pandas as pd

QUEUE_EVERY_MINUTES = 15
QUEUE_COLUMNS = ["run_utc", "order_id", "ticker", "ahead"]


def log_queue_positions(client, ledger, now, path, every_minutes=QUEUE_EVERY_MINUTES):
    """Append the queue position of each resting order; returns rows added."""
    resting = ledger[ledger.status.isin(["placed", "resting"])]
    resting = resting[resting.order_id.notna() & (resting.order_id.astype(str) != "")]
    if resting.empty:
        return 0
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=QUEUE_COLUMNS)
    now = pd.Timestamp(now)
    if len(old):
        last = pd.to_datetime(old.run_utc, utc=True, errors="coerce").max()
        if pd.notna(last) and now - last < pd.Timedelta(minutes=every_minutes):
            return 0
    rows = []
    for r in resting.itertuples():
        try:
            ahead = client.queue_position(str(r.order_id))
        except Exception as error:  # noqa: BLE001 - one bad read must not stop the run
            print(f"queue position failed for {r.ticker}: {error}")
            continue
        rows.append(
            {
                "run_utc": now.isoformat(timespec="minutes"),
                "order_id": str(r.order_id),
                "ticker": r.ticker,
                "ahead": ahead,
            }
        )
    if rows:
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True)[QUEUE_COLUMNS].to_csv(
            path, index=False
        )
    return len(rows)


def qualifying_volume(trades, level, start_ts, end_ts):
    """Public contracts that traded through our resting "no" (a taker buying
    "yes" at or above our yes level) between placement and ``end_ts``."""
    if not trades:
        return 0.0
    t = pd.DataFrame(trades)
    if t.empty:
        return 0.0
    t["ts"] = pd.to_datetime(t.created_time, utc=True, errors="coerce")
    t["yes_price"] = pd.to_numeric(t.yes_price, errors="coerce")
    t["count"] = pd.to_numeric(t["count"], errors="coerce").fillna(0)
    side = t.get("taker_outcome_side", pd.Series("", index=t.index)).fillna("")
    hit = (
        (side.str.lower() == "yes")
        & (t.yes_price >= level - 1e-9)
        & (t.ts >= pd.Timestamp(start_ts, unit="s", tz="UTC"))
        & (t.ts <= pd.Timestamp(end_ts, unit="s", tz="UTC"))
    )
    return float(t.loc[hit, "count"].sum())


def order_profit(fills, settlements):
    """Realised profit per order from its own buy fills and the market result:
    for a "no" bought at p, (1 if the market settled no else 0) - p per
    contract, minus the fill's fee. Returns {order_id: (contracts, profit)}."""
    if fills is None or not len(fills) or "order_id" not in fills:
        return {}
    f = fills.copy()
    f = f[f.order_id.notna() & (f.order_id.astype(str) != "")]
    if f.empty:
        return {}
    result = {}
    if settlements is not None and len(settlements) and "market_result" in settlements:
        result = (
            settlements.drop_duplicates("ticker")
            .set_index("ticker")
            .market_result.to_dict()
        )
    f["count"] = pd.to_numeric(f["count"], errors="coerce").fillna(0)
    f["fee_cost"] = pd.to_numeric(f.get("fee_cost", 0), errors="coerce").fillna(0)
    out = {}
    for r in f.itertuples():
        res = result.get(r.ticker)
        if res not in ("yes", "no"):
            continue
        action = getattr(r, "action", "buy") or "buy"
        if action != "buy":
            continue
        raw = r.no_price if r.side == "no" else r.yes_price
        price = pd.to_numeric(raw, errors="coerce") / 100
        payout = 1.0 if res == r.side else 0.0
        profit = r.count * (payout - price) - r.fee_cost
        c, p = out.get(str(r.order_id), (0.0, 0.0))
        out[str(r.order_id)] = (c + r.count, p + profit)
    return out


def execution_report(ledger, fills, queue, settlements=None, trades=None, now=None):
    """One row per real order (status placed/resting/done/filled/cancelled).

    ``trades``: optional ``fetch_trades(ticker, min_ts, max_ts)`` used to
    count the public volume that traded through each order (the capture
    denominator); orders older than three days are not re-fetched.
    """
    real = ledger[
        ledger.status.isin(["placed", "resting", "done", "filled", "cancelled"])
    ]
    real = real[real.order_id.notna() & (real.order_id.astype(str) != "")].copy()
    if real.empty:
        return pd.DataFrame()
    real["order_id"] = real.order_id.astype(str)

    def utc(column):
        if column in real:
            return pd.to_datetime(real[column], utc=True, errors="coerce")
        return pd.Series(pd.NaT, index=real.index, dtype="datetime64[ns, UTC]")

    real["placed_utc"] = utc("run_utc")
    real["accept_utc"] = utc("accept_utc")
    real["expiration_utc"] = utc("expiration_utc")
    cols = [
        "order_id",
        "ticker",
        "day",
        "player",
        "stat",
        "line",
        "price",
        "contracts",
        "edge",
        "placed_utc",
        "accept_utc",
        "expiration_utc",
    ]
    out = real[[c for c in cols if c in real]].copy()
    out["signal_to_accept_seconds"] = (
        out.accept_utc - out.placed_utc
    ).dt.total_seconds()
    # starting queue: the reading at placement, else the first periodic one
    out["queue_ahead_at_start"] = pd.to_numeric(
        real.get("queue_ahead_at_place"), errors="coerce"
    )
    if queue is not None and len(queue):
        q = queue.copy()
        q["order_id"] = q.order_id.astype(str)
        q["run_utc"] = pd.to_datetime(q.run_utc, utc=True, errors="coerce")
        first = (
            q.sort_values("run_utc")
            .drop_duplicates("order_id")
            .set_index("order_id")
            .ahead
        )
        out["queue_ahead_at_start"] = out.queue_ahead_at_start.fillna(
            out.order_id.map(first)
        )
    # our fills, by order id
    out["first_fill_utc"] = pd.Series(
        pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"
    )
    out["filled"] = 0.0
    if fills is not None and len(fills) and "order_id" in fills:
        f = fills.copy()
        f["order_id"] = f.order_id.astype(str)
        f["created_time"] = pd.to_datetime(f.created_time, utc=True, errors="coerce")
        f["count"] = pd.to_numeric(f["count"], errors="coerce").fillna(0)
        g = f.groupby("order_id").agg(
            first_fill_utc=("created_time", "min"), filled=("count", "sum")
        )
        out = out.drop(columns=["first_fill_utc", "filled"]).merge(
            g, left_on="order_id", right_index=True, how="left"
        )
        out["filled"] = out.filled.fillna(0.0)
    out["fill_share_of_order"] = (out.filled / out.contracts.replace(0, pd.NA)).astype(
        float
    )
    out["minutes_to_first_fill"] = (
        out.first_fill_utc - out.placed_utc
    ).dt.total_seconds() / 60
    # capture: our fills against the public volume that traded through us
    out["qualifying_volume"] = pd.NA
    if trades is not None and now is not None:
        now = pd.Timestamp(now)
        for i in out.index:
            placed = out.at[i, "placed_utc"]
            if pd.isna(placed) or now - placed > pd.Timedelta(days=3):
                continue
            exp = out.at[i, "expiration_utc"]
            end = min(now, exp) if pd.notna(exp) else now
            level = 1 - float(out.at[i, "price"])
            try:
                prints = trades(
                    out.at[i, "ticker"], int(placed.timestamp()), int(end.timestamp())
                )
            except Exception as error:  # noqa: BLE001 - measurement only
                print(f"trades for {out.at[i, 'ticker']} failed: {error}")
                continue
            out.at[i, "qualifying_volume"] = qualifying_volume(
                prints, level, int(placed.timestamp()), int(end.timestamp())
            )
    qv = pd.to_numeric(out.qualifying_volume, errors="coerce")
    out["capture"] = (out.filled / qv.where(qv > 0)).clip(upper=1.0)
    # order-level realised profit from the order's own fills and the result
    prof = order_profit(fills, settlements)
    out["settled_contracts"] = out.order_id.map(lambda o: prof.get(o, (None, None))[0])
    out["settled_profit"] = out.order_id.map(lambda o: prof.get(o, (None, None))[1])
    return out


def summary(report):
    """Headline numbers for the dashboard and the markdown file."""
    if report is None or report.empty:
        return {"orders": 0}
    filled = report[report.filled > 0]
    d = {
        "orders": int(len(report)),
        "orders_with_queue_reading": int(report.queue_ahead_at_start.notna().sum()),
        "median_queue_ahead_at_start": float(report.queue_ahead_at_start.median())
        if report.queue_ahead_at_start.notna().any()
        else None,
        "orders_filled": int(len(filled)),
        "fill_rate": round(float((report.filled > 0).mean()), 3),
        "mean_fill_share_of_order": round(
            float(report.fill_share_of_order.fillna(0).mean()), 3
        ),
        "orders_with_capture_measured": int(report.capture.notna().sum()),
        "mean_capture_of_qualifying_volume": round(float(report.capture.mean()), 3)
        if report.capture.notna().any()
        else None,
        "median_signal_to_accept_seconds": round(
            float(report.signal_to_accept_seconds.median()), 1
        )
        if report.signal_to_accept_seconds.notna().any()
        else None,
        "median_minutes_to_first_fill": round(
            float(filled.minutes_to_first_fill.median()), 1
        )
        if len(filled)
        else None,
    }
    s = report.settled_profit.dropna()
    c = pd.to_numeric(report.settled_contracts, errors="coerce").dropna()
    d["settled_orders"] = int(len(s))
    d["settled_contracts"] = int(c.sum()) if len(c) else 0
    d["settled_profit"] = round(float(s.sum()), 2)
    d["cents_per_settled_contract"] = (
        round(100 * float(s.sum() / c.sum()), 2) if len(c) and c.sum() else None
    )
    return d


def write_markdown(report, path, now):
    d = summary(report)
    lines = [
        "# Execution quality (real orders)",
        "",
        f"Updated {pd.Timestamp(now):%Y-%m-%d %H:%M} UTC.",
        "",
        "| Measure | Value |",
        "|---|---:|",
    ]
    lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in d.items()]
    if report is not None and len(report):
        lines += [
            "",
            "| Placed (UTC) | Ticker | Price | Ordered | Queue ahead at start | Filled | Qualifying vol | Min to first fill |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in (
            report.sort_values("placed_utc", ascending=False).head(50).itertuples()
        ):
            lines.append(
                f"| {r.placed_utc:%m-%d %H:%M} | {r.ticker} | {r.price:.2f} | {r.contracts} | "
                f"{'' if pd.isna(r.queue_ahead_at_start) else f'{r.queue_ahead_at_start:.0f}'} | "
                f"{r.filled:.0f} | {'' if pd.isna(r.qualifying_volume) else f'{float(r.qualifying_volume):.0f}'} | "
                f"{'' if pd.isna(r.minutes_to_first_fill) else f'{r.minutes_to_first_fill:.0f}'} |"
            )
    path.write_text("\n".join(lines) + "\n")
    return d
