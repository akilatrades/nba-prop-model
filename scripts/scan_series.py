"""Scan Kalshi sports series for the structural edge (RESEARCH_LOG.md, scan).

``list``: every Kalshi series in the Sports category, with how many settled
markets it had since --start and their volume (first page of the historical
endpoint; prints counts). Writes data/scan/series.parquet.

``scan --series X``: for one series, every settled market since --start, a
random sample (seed 0, --sample) of those that traded, with hourly candles
and trade prints over each market's life. Writes data/scan/<series>_*.parquet.
Prices are not examined here.
"""

import argparse
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_model.kalshi import candles, fetch_markets, get_json, trades

OUT = Path("data/scan")
FIELDS = [
    "ticker",
    "event_ticker",
    "title",
    "yes_sub_title",
    "floor_strike",
    "strike_type",
    "open_time",
    "close_time",
    "expected_expiration_time",
    "result",
    "volume_fp",
    "last_price_dollars",
]


def annotate(level, message):
    print(message, flush=True)
    print(f"::{level}::" + message.replace("%", "%25").replace("\n", "%0A"), flush=True)


def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def to_utc(column):
    return pd.to_datetime(column, utc=True, format="ISO8601", errors="coerce")


def list_series(start):
    series, cursor = [], ""
    while True:
        params = {"limit": 1000, "category": "Sports"}
        if cursor:
            params["cursor"] = cursor
        page = get_json("/series", params) or {}
        series.extend(page.get("series", []))
        cursor = page.get("cursor") or ""
        if not cursor or not page.get("series"):
            break
    print(f"{len(series)} sports series")
    min_ts = int(pd.Timestamp(start, tz="UTC").timestamp())

    def count(s):
        page = (
            get_json(
                "/historical/markets",
                {"series_ticker": s["ticker"], "limit": 1000, "min_close_ts": min_ts},
            )
            or {}
        )
        m = page.get("markets", [])
        vol = sum(float(x.get("volume_fp") or 0) for x in m)
        return {
            "series": s["ticker"],
            "title": s.get("title"),
            "frequency": s.get("frequency"),
            "tags": ",".join(s.get("tags") or []),
            "markets_first_page": len(m),
            "has_more": bool(page.get("cursor")),
            "traded_first_page": sum(
                1 for x in m if float(x.get("volume_fp") or 0) > 0
            ),
            "volume_first_page": vol,
            "sample_title": m[0].get("title") if m else None,
        }

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(count, series))
    frame = pd.DataFrame(rows).sort_values("volume_first_page", ascending=False)
    save(frame, OUT / "series.parquet")
    active = frame[frame.markets_first_page > 0]
    annotate(
        "notice",
        f"{len(frame)} sports series, {len(active)} with settled markets since {start}; "
        f"top by volume: {active.head(40)[['series', 'markets_first_page', 'volume_first_page']].to_dict('records')}",
    )


def run_jobs(function, jobs, workers, deadline):
    results, failed = {}, 0
    with ThreadPoolExecutor(workers) as pool:
        it = iter(jobs.items())
        pending = {}

        def submit():
            item = next(it, None)
            if item and time.monotonic() < deadline:
                pending[pool.submit(function, *item[1])] = item[0]

        for _ in range(workers * 3):
            submit()
        while pending:
            future = next(as_completed(list(pending)))
            key = pending.pop(future)
            try:
                results[key] = future.result()
            except (OSError, ValueError, KeyError):
                failed += 1
            submit()
    return results, failed


def scan(series, start, sample, minutes, workers):
    deadline = time.monotonic() + 60 * minutes
    got = fetch_markets(series, status=None, historical=True)
    rows = [{k: m.get(k) for k in FIELDS} for m in got]
    m = pd.DataFrame(rows, columns=FIELDS)
    m["close"] = to_utc(m.close_time)
    m = m[m.close >= pd.Timestamp(start, tz="UTC")].drop_duplicates("ticker")
    save(m.drop(columns=["close"]), OUT / f"{series}_markets.parquet")
    volume = pd.to_numeric(m.volume_fp, errors="coerce").fillna(0)
    traded = m[m.result.isin(["yes", "no"]) & (volume > 0)]
    rng = np.random.default_rng(0)
    pick = traded.iloc[rng.permutation(len(traded))[:sample]]
    start_ts = (to_utc(pick.open_time) - pd.Timedelta(hours=1)).map(
        lambda x: int(x.timestamp())
    )
    end_ts = (pick.close + pd.Timedelta(hours=1)).map(lambda x: int(x.timestamp()))
    jobs_c = {t: (t, a, b, 60, True) for t, a, b in zip(pick.ticker, start_ts, end_ts)}
    jobs_t = {t: (t, a, b, True) for t, a, b in zip(pick.ticker, start_ts, end_ts)}
    c, failed_c = run_jobs(candles, jobs_c, workers, deadline)
    t, failed_t = run_jobs(trades, jobs_t, workers, deadline)
    cf = pd.DataFrame([r for v in c.values() for r in v])
    tf = pd.DataFrame([r for v in t.values() for r in v])
    if len(tf):
        tf["ts"] = to_utc(tf.created_time).map(lambda x: int(x.timestamp()))
        tf = tf.drop(columns=["created_time"])
    save(cf, OUT / f"{series}_candles.parquet")
    save(tf, OUT / f"{series}_trades.parquet")
    annotate(
        "notice",
        f"{series}: {len(m)} settled markets since {start}, {len(traded)} traded, "
        f"sampled {len(pick)}; candles for {len(c)} (failed {failed_c}, {len(cf)} rows), "
        f"trades for {len(t)} (failed {failed_t}, {len(tf)} prints)",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["list", "scan"])
    parser.add_argument("--series")
    parser.add_argument("--start", default="2025-09-01")
    parser.add_argument("--sample", type=int, default=600)
    parser.add_argument("--minutes", type=float, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part == "list":
        list_series(args.start)
    else:
        scan(args.series, args.start, args.sample, args.minutes, args.workers)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - report any failure as an annotation
        annotate("error", traceback.format_exc()[-3000:])
        sys.exit(1)
