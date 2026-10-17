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


def execution_report(ledger, fills, queue, settlements=None):
    """One row per real order (status placed/resting/done/filled/cancelled)."""
    real = ledger[
        ledger.status.isin(["placed", "resting", "done", "filled", "cancelled"])
    ]
    real = real[real.order_id.notna() & (real.order_id.astype(str) != "")].copy()
    if real.empty:
        return pd.DataFrame()
    real["order_id"] = real.order_id.astype(str)
    real["placed_utc"] = pd.to_datetime(real.run_utc, utc=True, errors="coerce")
    out = real[
        [
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
        ]
    ].copy()
    # starting queue position: the first reading after placement
    if queue is not None and len(queue):
        q = queue.copy()
        q["order_id"] = q.order_id.astype(str)
        q["run_utc"] = pd.to_datetime(q.run_utc, utc=True, errors="coerce")
        first = q.sort_values("run_utc").drop_duplicates("order_id")
        out = out.merge(
            first[["order_id", "ahead", "run_utc"]].rename(
                columns={"ahead": "queue_ahead_at_start", "run_utc": "queue_read_utc"}
            ),
            on="order_id",
            how="left",
        )
    else:
        out["queue_ahead_at_start"] = pd.NA
    # fills: by ticker (Kalshi fills carry the ticker, and the order id when
    # available), first fill time and contracts
    if fills is not None and len(fills):
        f = fills.copy()
        f["created_time"] = pd.to_datetime(f.created_time, utc=True, errors="coerce")
        f["count"] = pd.to_numeric(f.get("count"), errors="coerce").fillna(0)
        key = "order_id" if "order_id" in f and f.order_id.notna().any() else "ticker"
        if key == "order_id":
            f["order_id"] = f.order_id.astype(str)
        g = f.groupby(key).agg(
            first_fill_utc=("created_time", "min"), filled=("count", "sum")
        )
        out = out.merge(g, left_on=key, right_index=True, how="left")
    else:
        out["first_fill_utc"] = pd.NaT
        out["filled"] = 0.0
    out["filled"] = out.filled.fillna(0.0)
    out["fill_share"] = (out.filled / out.contracts.replace(0, pd.NA)).astype(float)
    out["minutes_to_first_fill"] = (
        out.first_fill_utc - out.placed_utc
    ).dt.total_seconds() / 60
    # realised profit per contract where the market has settled
    if settlements is not None and len(settlements) and "profit" in settlements:
        st = settlements.drop_duplicates("ticker")[["ticker", "profit"]]
        out = out.merge(
            st.rename(columns={"profit": "settled_profit"}), on="ticker", how="left"
        )
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
        "mean_fill_share": round(float(report.fill_share.fillna(0).mean()), 3),
        "median_minutes_to_first_fill": round(
            float(filled.minutes_to_first_fill.median()), 1
        )
        if len(filled)
        else None,
    }
    if "settled_profit" in report:
        s = report.settled_profit.dropna()
        d["settled_orders"] = int(len(s))
        d["settled_profit"] = round(float(s.sum()), 2)
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
            "| Placed (UTC) | Ticker | Price | Ordered | Queue ahead at start | Filled | Min to first fill |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in (
            report.sort_values("placed_utc", ascending=False).head(50).itertuples()
        ):
            lines.append(
                f"| {r.placed_utc:%m-%d %H:%M} | {r.ticker} | {r.price:.2f} | {r.contracts} | "
                f"{'' if pd.isna(r.queue_ahead_at_start) else f'{r.queue_ahead_at_start:.0f}'} | "
                f"{r.filled:.0f} | {'' if pd.isna(r.minutes_to_first_fill) else f'{r.minutes_to_first_fill:.0f}'} |"
            )
    path.write_text("\n".join(lines) + "\n")
    return d
