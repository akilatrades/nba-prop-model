"""Adverse selection in the NBA "no" seller's fills (RESEARCH_LOG.md, 2026-10-16,
"Execution studies"): markouts after each fill and a pre-posting toxicity veto.

For every 2025-26 resting "no" order that filled (outputs/decompose/
orders_no.csv.gz), find the first trade print that would have hit it, then:

- markouts: the last traded "yes" price 5, 15 and 60 minutes after the fill
  (and at the final pre-tip print), minus the level we sold at. We sold
  "yes" at the level, so a falling price is a gain; markout is reported from
  our side (positive = price moved our way).
- pre-fill features, all observable before the order was posted: the yes
  price change over the hour before posting, trade count and "yes"-taker
  share in the 15 minutes before posting, and the market's volume so far.

Then the fills are split by those features and the settlement profit of
each group is compared, to see whether a filter known before posting
would have kept us out of the fills that lose.

    python -m research.markouts   # offline, writes outputs/execution/markouts.*
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi.stats import clustered_mean_t

FILLS = Path("outputs/decompose/orders_no.csv.gz")
TRADES = [
    Path(f"data/kalshi_history/trades_{s}.parquet")
    for s in ("KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT")
]
OUT = Path("outputs/execution")
HORIZONS = {"m5": 300, "m15": 900, "m60": 3600}


def load():
    o = pd.read_csv(FILLS, low_memory=False)
    o = o[o.filled.astype(str).str.lower().eq("true")].copy()
    o["improved"] = o.improved.astype(str).str.lower().eq("true")
    t = pd.concat([pd.read_parquet(p) for p in TRADES if p.exists()], ignore_index=True)
    stamps = pd.to_datetime(t.created_time, utc=True, format="ISO8601")
    t["ts"] = (
        stamps.dt.tz_convert(None).to_numpy().astype("datetime64[s]").astype("int64")
    )
    t = t.sort_values(["ticker", "ts"]).reset_index(drop=True)
    t["yes_price"] = t.yes_price.astype(float)
    t["count"] = t["count"].astype(float)
    return o, t


def fill_events(orders, trades):
    """For each filled order, the first print at or through our level after posting."""
    rows = []
    by_ticker = {k: g for k, g in trades.groupby("ticker")}
    for r in orders.itertuples():
        g = by_ticker.get(r.ticker)
        if g is None:
            continue
        after = g[
            (g.ts > r.end_ts) & (g.ts <= r.tip_ts) & (g.taker_outcome_side == "yes")
        ]
        level = float(r.yes_level)
        hit = after[
            (after.yes_price > level + 1e-6)
            | ((after.yes_price - level).abs() < 1e-6) & bool(r.improved)
        ]
        if hit.empty:
            continue
        first = hit.iloc[0]
        rows.append(
            {
                "ticker": r.ticker,
                "game_id": r.game_id,
                "stat": r.stat,
                "end_ts": int(r.end_ts),
                "tip_ts": int(r.tip_ts),
                "fill_ts": int(first.ts),
                "fill_size": float(first["count"]),
                "level": level,
                "price": float(r.price),
                "pnl": float(r.pnl_if_filled),
                "model_edge": float(r.model_edge),
                "over": float(r.over),
            }
        )
    return pd.DataFrame(rows)


def features_and_markouts(ev, trades):
    by_ticker = {k: g for k, g in trades.groupby("ticker")}
    out = []
    for r in ev.itertuples():
        g = by_ticker[r.ticker]
        before = g[g.ts <= r.end_ts]
        hour = before[before.ts > r.end_ts - 3600]
        q15 = before[before.ts > r.end_ts - 900]
        last_before = before.yes_price.iloc[-1] if len(before) else np.nan
        first_hour = hour.yes_price.iloc[0] if len(hour) else last_before
        row = {
            "momentum_1h": (last_before - first_hour)
            if pd.notna(last_before) and pd.notna(first_hour)
            else 0.0,
            "trades_15m": int(len(q15)),
            "yes_share_15m": float((q15.taker_outcome_side == "yes").mean())
            if len(q15)
            else np.nan,
            "volume_before": float(before["count"].sum()),
            "minutes_to_tip_at_fill": (r.tip_ts - r.fill_ts) / 60,
        }
        after = g[g.ts > r.fill_ts]
        for k, sec in HORIZONS.items():
            w = after[after.ts <= min(r.fill_ts + sec, r.tip_ts)]
            row[k] = (r.level - w.yes_price.iloc[-1]) if len(w) else np.nan
        pre_tip = after[after.ts <= r.tip_ts]
        row["to_tip"] = (
            (r.level - pre_tip.yes_price.iloc[-1]) if len(pre_tip) else np.nan
        )
        out.append(row)
    return pd.concat([ev.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def summary(frame, label):
    if frame.empty:
        return {"label": label, "fills": 0}
    mean, t = clustered_mean_t(frame.pnl, frame.game_id)
    d = {
        "label": label,
        "fills": int(len(frame)),
        "cents": round(100 * float(frame.pnl.mean()), 2),
        "t": round(float(t), 2) if np.isfinite(t) else None,
    }
    for k in list(HORIZONS) + ["to_tip"]:
        d[f"markout_{k}"] = (
            round(100 * float(frame[k].mean()), 2) if frame[k].notna().any() else None
        )
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "markouts.csv.gz"
    if cache.exists():
        df = pd.read_csv(cache)
        orders = pd.read_csv(FILLS, low_memory=False)
        orders = orders[orders.filled.astype(str).str.lower().eq("true")]
    else:
        orders, trades = load()
        ev = fill_events(orders, trades)
        df = features_and_markouts(ev, trades)
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_csv(cache, index=False)
    rows = [summary(df, "all fills")]
    # Pre-registered splits (RESEARCH_LOG.md): each feature at its median,
    # plus the two toxicity flags defined before the run.
    for col, lab in [
        ("momentum_1h", "yes price rose in the hour before posting"),
        ("trades_15m", "busy 15 min before posting"),
        ("yes_share_15m", "yes-taker share 15 min before posting"),
        ("fill_size", "size of the print that filled us"),
        ("minutes_to_tip_at_fill", "minutes to tip at fill"),
    ]:
        x = df[col]
        med = float(x.median()) if x.notna().any() else 0.0
        hi, lo = df[x > med], df[x <= med]
        rows.append(summary(hi, f"{lab}: above median ({med:.3g})"))
        rows.append(summary(lo, f"{lab}: at or below median"))
    toxic = (df.momentum_1h > 0.02) | ((df.yes_share_15m > 0.8) & (df.trades_15m >= 3))
    rows.append(
        summary(
            df[toxic], "toxicity flag: momentum > 2c or yes-share > 80% on 3+ prints"
        )
    )
    rows.append(summary(df[~toxic], "toxicity flag off"))
    late = df.minutes_to_tip_at_fill <= 15
    rows.append(summary(df[late], "filled in the last 15 minutes before tip"))
    rows.append(summary(df[~late], "filled earlier"))
    res = {"fills_matched": int(len(df)), "of_orders": int(len(orders)), "rows": rows}
    (OUT / "markouts.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = [
        "# Markouts and adverse selection, NBA no-seller fills 2025-26",
        "",
        f"{len(df):,} of {len(orders):,} paper fills matched to the print that filled them.",
        "",
        "| Fills | n | ¢/contract (settlement) | t | markout 5m | 15m | 60m | to tip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("fills"):
            lines.append(
                f"| {r['label']} | {r['fills']:,} | {r['cents']:+.2f} | {r['t']} | {r['markout_m5']} | {r['markout_m15']} | {r['markout_m60']} | {r['markout_to_tip']} |"
            )
    (OUT / "markouts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
