"""Market scan (RESEARCH_LOG.md, 2026-10-16): where else does the "no" seller work?

Stage ``list``: every Kalshi Sports series, with the volume on its first page
of settled markets; writes outputs/scan/series.json.
Stage ``scan --shard i --shards n``: for the series assigned to this shard,
sample up to SAMPLE settled markets from the last 400 days, fetch hourly
candles for the 7-to-4-hour pre-expiration window and the trade prints for
the 7-to-3-hour window, simulate resting "no" and "yes" orders, and write
outputs/scan/scan_<shard>.json plus the order-level rows.
Stage ``report``: merge the shards into outputs/scan/results.json.

Prices are only used as the pre-registered test describes. Prints counts.
"""

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi.api import candles, fetch_markets, get_json, trades
from kalshi.maker import maker_fee
from kalshi.stats import clustered_mean_t

OUT = Path("outputs/scan")
SAMPLE = 1200
MIN_PAGE_VOLUME = 20000
LOOKBACK_DAYS = 400
WINDOW = (7, 4)  # hours before expected expiration for order placement
TICK = 0.01
FEE_SHARE = 0.25


def annotate(level, message):
    print(message, flush=True)
    print(f"::{level}::" + message.replace("%", "%25").replace("\n", "%0A"), flush=True)


def to_utc(x):
    return pd.to_datetime(x, utc=True, format="ISO8601", errors="coerce")


# --- Stage: list ---------------------------------------------------------------


def all_series(category=None):
    out, cursor = [], ""
    while True:
        params = {"limit": 1000}
        if category:
            params["category"] = category
        if cursor:
            params["cursor"] = cursor
        page = get_json("/series", params) or {}
        out.extend(page.get("series", []))
        cursor = page.get("cursor") or ""
        if not cursor or not page.get("series"):
            return out


def list_stage(skip_done=False):
    series = all_series()
    done = set()
    if skip_done and (OUT / "results.json").exists():
        done = {
            r["series"]
            for r in json.loads((OUT / "results.json").read_text())
            if "no_resting" in r
        }
    rows = []
    for s in series:
        ticker = s.get("ticker", "")
        if "MVE" in ticker or "COMBO" in ticker or "PREPACK" in ticker:
            continue
        page = (
            get_json("/historical/markets", {"series_ticker": ticker, "limit": 1000})
            or {}
        )
        markets = page.get("markets", [])
        if not markets:
            page = (
                get_json(
                    "/markets",
                    {"series_ticker": ticker, "status": "settled", "limit": 1000},
                )
                or {}
            )
            markets = page.get("markets", [])
        volume = sum(float(m.get("volume_fp") or 0) for m in markets)
        rows.append(
            {
                "ticker": ticker,
                "title": s.get("title"),
                "category": s.get("category"),
                "frequency": s.get("frequency"),
                "first_page_markets": len(markets),
                "has_more": bool(page.get("cursor")),
                "first_page_volume": volume,
            }
        )
    frame = pd.DataFrame(rows).sort_values("first_page_volume", ascending=False)
    keep = frame[
        (frame.first_page_volume >= MIN_PAGE_VOLUME) & ~frame.ticker.isin(done)
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "series.json").write_text(
        json.dumps(
            {"all": frame.to_dict("records"), "selected": keep.ticker.tolist()},
            indent=1,
            default=str,
        )
        + "\n"
    )
    annotate(
        "notice",
        f"{len(frame)} series in all categories, {len(keep)} with >= ${MIN_PAGE_VOLUME:,} "
        f"on the first page and not scanned before; by category: "
        f"{keep.category.value_counts().to_dict()}",
    )


# --- Stage: scan -----------------------------------------------------------------


def sample_markets(series):
    markets = fetch_markets(series, status=None, historical=True)
    markets += fetch_markets(series, status="settled", historical=False)
    frame = pd.DataFrame(markets).drop_duplicates("ticker", keep="last")
    if frame.empty:
        return frame
    frame["expiry"] = to_utc(frame.get("expected_expiration_time")).fillna(
        to_utc(frame.close_time)
    )
    frame["volume"] = pd.to_numeric(frame.volume_fp, errors="coerce").fillna(0)
    recent = frame.expiry >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(
        days=LOOKBACK_DAYS
    )
    frame = frame[recent & (frame.volume > 0) & frame.result.isin(["yes", "no"])]
    if len(frame) > SAMPLE:
        frame = frame.sample(SAMPLE, random_state=0)
    frame.attrs["sample"] = SAMPLE
    return frame.reset_index(drop=True)


def candles_any(ticker, start, end, period=60):
    """Historical endpoint first, then the live one (recently settled markets)."""
    return candles(ticker, start, end, period, True) or candles(
        ticker, start, end, period, False
    )


def trades_any(ticker, start, end):
    return trades(ticker, start, end, True) or trades(ticker, start, end, False)


def fetch_all(function, jobs, workers=8):
    results, failed = {}, 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(function, *args): key for key, args in jobs.items()}
        for future in as_completed(futures):
            try:
                results[futures[future]] = future.result()
            except (OSError, ValueError, KeyError):
                failed += 1
    return results, failed


def resting_orders(c, side):
    """One resting order per candle; price, yes level, improved flag."""
    q = c.copy()
    spread = (q.yes_ask - q.yes_bid).round(4)
    improve = spread >= 2 * TICK - 1e-9
    if side == "yes":
        price = np.where(improve, q.yes_bid + TICK, q.yes_bid)
    else:
        no_bid = 1 - q.yes_ask
        price = np.where(improve, no_bid + TICK, no_bid)
    q["side"] = side
    q["price"] = np.round(price, 4)
    q["improved"] = improve
    q["yes_level"] = np.round(np.where(side == "yes", q.price, 1 - q.price), 4)
    return q[(q.price > 0) & (q.price < 1)]


def fill_orders(o, t):
    """Contracts traded through each order's price in the hour after it rests."""
    if o.empty or t.empty:
        return pd.Series(dtype=float)
    o = o.reset_index(drop=True)
    o["order_id"] = np.arange(len(o))
    m = t.merge(
        o[["order_id", "ticker", "end_ts", "side", "yes_level", "improved"]],
        on="ticker",
    )
    m = m[(m.ts > m.end_ts) & (m.ts <= m.end_ts + 3600)]
    same = (m.yes_price - m.yes_level).abs() < 1e-6
    hit_yes = (m.side == "yes") & (m.taker_outcome_side == "no")
    hit_yes &= (m.yes_price < m.yes_level - 1e-6) | (same & m.improved)
    hit_no = (m.side == "no") & (m.taker_outcome_side == "yes")
    hit_no &= (m.yes_price > m.yes_level + 1e-6) | (same & m.improved)
    return m[hit_yes | hit_no].groupby("order_id")["count"].sum()


def summarize(f, label):
    if len(f) < 2:
        return {"fills": len(f)}
    mean, t = clustered_mean_t(f.pnl, f.event_ticker)
    mid = f.date.sort_values().iloc[len(f) // 2]
    first, second = f[f.date < mid], f[f.date >= mid]
    return {
        "fills": len(f),
        "events": int(f.event_ticker.nunique()),
        "contracts_traded_through": float(f.traded.sum()),
        "cents_per_contract": 100 * float(mean),
        "t": float(t),
        "first_half_cents": 100 * float(first.pnl.mean()) if len(first) else None,
        "second_half_cents": 100 * float(second.pnl.mean()) if len(second) else None,
        "hit_rate": float(f.won.mean()),
        "avg_price": float(f.price.mean()),
    }


def scan_series(series):
    t0 = time.monotonic()
    m = sample_markets(series)
    if m.empty:
        return {"series": series, "markets": 0}
    exp = m.expiry.map(lambda x: int(x.timestamp()))
    candle_jobs = {
        r.ticker: (r.ticker, e - WINDOW[0] * 3600, e - WINDOW[1] * 3600 + 1)
        for r, e in zip(m.itertuples(), exp)
    }
    got, failed_c = fetch_all(candles_any, candle_jobs)
    c = pd.DataFrame([row for rows in got.values() for row in rows])
    if c.empty:
        return {"series": series, "markets": len(m), "candles": 0}
    c = c[(c.yes_bid > 0) & (c.yes_ask < 1) & (c.yes_ask > c.yes_bid)]
    trade_jobs = {
        r.ticker: (r.ticker, e - WINDOW[0] * 3600, e - (WINDOW[1] - 1) * 3600 + 1)
        for r, e in zip(m.itertuples(), exp)
    }
    got_t, failed_t = fetch_all(trades_any, trade_jobs)
    t = pd.DataFrame([row for rows in got_t.values() for row in rows])
    if len(t):
        t["ts"] = to_utc(t.created_time).map(lambda x: int(x.timestamp()))
    info = m.set_index("ticker")
    result = {
        "series": series,
        "markets": len(m),
        "markets_with_quotes": int(c.ticker.nunique()),
        "sample_volume": float(m.volume.sum()),
        "median_spread_cents": float(100 * (c.yes_ask - c.yes_bid).median()),
        "trade_prints": len(t),
        "failed_requests": failed_c + failed_t,
        "expiry_minus_close_hours_median": float(
            ((m.expiry - to_utc(m.close_time)).dt.total_seconds() / 3600).median()
        ),
        "seconds": round(time.monotonic() - t0),
    }
    rows = []
    for side in ["no", "yes"]:
        o = resting_orders(c, side).reset_index(drop=True)
        o["traded"] = fill_orders(o, t).reindex(range(len(o))).fillna(0).to_numpy()
        o["result"] = o.ticker.map(info.result)
        o["event_ticker"] = o.ticker.map(info.event_ticker)
        o["date"] = o.ticker.map(info.expiry).dt.date
        o["won"] = np.where(o.side == o.result, 1.0, 0.0)
        o["pnl"] = o.won - o.price - maker_fee(o.price, FEE_SHARE)
        f = o[o.traded >= 1]
        result[f"{side}_orders"] = len(o)
        result[f"{side}_fill_rate"] = float((o.traded >= 1).mean()) if len(o) else None
        result[f"{side}_resting"] = summarize(f, side)
        rows.append(o.assign(series=series))
    pd.concat(rows).to_csv(OUT / f"orders_{series}.csv.gz", index=False)
    no = result["no_resting"]
    yes = result["yes_resting"]
    result["passes"] = bool(
        no.get("fills", 0) >= 2
        and no.get("cents_per_contract", 0) > 0
        and no.get("t", 0) >= 2.0
        and (no.get("first_half_cents") or 0) > 0
        and (no.get("second_half_cents") or 0) > 0
        and not (yes.get("t", 0) >= 2.0)
    )
    return result


def scan_stage(shard, shards, minutes):
    deadline = time.monotonic() + 60 * minutes
    selected = json.loads((OUT / "series.json").read_text())["selected"]
    mine = [s for i, s in enumerate(selected) if i % shards == shard]
    results = []
    for series in mine:
        if time.monotonic() > deadline:
            results.append({"series": series, "skipped": "out of time"})
            continue
        try:
            results.append(scan_series(series))
        except Exception as error:  # noqa: BLE001 - keep scanning the other series
            results.append({"series": series, "error": repr(error)[:300]})
        print(json.dumps(results[-1], default=str)[:600], flush=True)
    (OUT / f"scan_{shard}.json").write_text(
        json.dumps(results, indent=1, default=str) + "\n"
    )
    brief = [
        (
            r["series"],
            r.get("markets"),
            round(r["no_resting"].get("cents_per_contract", 0), 2)
            if "no_resting" in r
            else None,
            round(r["no_resting"].get("t", 0), 1) if "no_resting" in r else None,
            r.get("passes"),
        )
        for r in results
    ]
    annotate("notice", f"shard {shard}: {brief}")


def report_stage():
    """Merge the shards with earlier results (a series' newest scan wins)."""
    previous = {}
    if (OUT / "results.json").exists():
        for r in json.loads((OUT / "results.json").read_text()):
            if "no_resting" in r:
                previous[r["series"]] = r
    for path in sorted(OUT.glob("scan_*.json")):
        for r in json.loads(path.read_text()):
            if "no_resting" in r or r["series"] not in previous:
                previous[r["series"]] = r
    rows = list(previous.values())
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")
    write_table(rows)
    print(
        f"{len(rows)} series in results, {sum('no_resting' in r for r in rows)} scanned"
    )


def write_table(rows):
    """outputs/scan/table.csv: one row per series, flat, for the README and dashboard."""
    meta = {}
    if (OUT / "series.json").exists():
        listed = json.loads((OUT / "series.json").read_text())
        for m in listed.get("all", []) if isinstance(listed, dict) else listed:
            meta[m.get("ticker") or m.get("series")] = m
    out = []
    for r in rows:
        m = meta.get(r["series"], {})
        no, yes = r.get("no_resting") or {}, r.get("yes_resting") or {}
        out.append(
            {
                "series": r["series"],
                "category": r.get("category") or m.get("category"),
                "title": r.get("title") or m.get("title"),
                "markets": r.get("markets"),
                "vol_$": r.get("sample_volume"),
                "spread": r.get("median_spread_cents"),
                "no_fills": no.get("fills"),
                "no_c": no.get("cents_per_contract"),
                "no_t": no.get("t"),
                "no_h1": no.get("first_half_cents"),
                "no_h2": no.get("second_half_cents"),
                "no_fill%": r.get("no_fill_rate"),
                "yes_fills": yes.get("fills"),
                "yes_c": yes.get("cents_per_contract"),
                "yes_t": yes.get("t"),
                "passes": r.get("passes"),
                "skipped": r.get("skipped"),
            }
        )
    pd.DataFrame(out).sort_values("no_t", ascending=False).to_csv(
        OUT / "table.csv", index=False
    )


def main():
    global SAMPLE
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["list", "scan", "report"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--minutes", type=float, default=300)
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--skip-done", action="store_true")
    args = parser.parse_args()
    SAMPLE = args.sample
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "list":
        list_stage(args.skip_done)
    elif args.stage == "scan":
        scan_stage(args.shard, args.shards, args.minutes)
    else:
        report_stage()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - report any failure as an annotation
        annotate("error", traceback.format_exc()[-3000:])
        sys.exit(1)
