"""Download Kalshi trade prints before tip-off for every priced market.

For each market in outputs/market_test/priced_markets.csv.gz (one series at
a time), fetches every trade from six hours before the scheduled tip-off to
tip-off. Used by the limit-order test (RESEARCH_LOG.md, idea 1). Resumable;
prints counts only.

Writes data/kalshi_history/trades_<series>.parquet.

Usage: python scripts/backfill_trades.py --series KXNBAPTS --minutes 300
"""

import argparse
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi.api import trades
from markets.nba.kalshi import SERIES

OUT = Path("data/kalshi_history")
PRICED = Path("outputs/market_test/priced_markets.csv.gz")
HOURS_BEFORE = 6


def annotate(level, message):
    print(message)
    print(f"::{level}::" + message.replace("%", "%25").replace("\n", "%0A"))


def save(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", required=True, choices=sorted(SERIES))
    parser.add_argument("--minutes", type=float, default=300)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    deadline = time.monotonic() + 60 * args.minutes
    OUT.mkdir(parents=True, exist_ok=True)
    priced = pd.read_csv(PRICED, usecols=["ticker", "tip_utc"], low_memory=False)
    priced = priced[priced.ticker.str.startswith(args.series + "-")]
    tip = pd.to_datetime(priced.tip_utc, utc=True, format="ISO8601")
    jobs = {
        t: (int((x - pd.Timedelta(hours=HOURS_BEFORE)).timestamp()), int(x.timestamp()))
        for t, x in zip(priced.ticker, tip)
    }
    path = OUT / f"trades_{args.series}.parquet"
    done_path = OUT / f"trades_done_{args.series}.txt"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    todo = [t for t in jobs if t not in done]
    print(f"{args.series}: {len(jobs)} markets, {len(todo)} still to fetch")
    rows, fetched, failed = [], 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        it = iter(todo)
        pending = {}
        for _ in range(args.workers * 4):
            t = next(it, None)
            if t:
                pending[pool.submit(trades, t, *jobs[t])] = t
        while pending:
            future = next(as_completed(list(pending)))
            ticker = pending.pop(future)
            try:
                rows.extend(future.result())
                done.add(ticker)
                fetched += 1
            except (OSError, ValueError, KeyError) as error:
                failed += 1
                if failed <= 5:
                    print(f"failed {ticker}: {error}")
            if time.monotonic() < deadline:
                t = next(it, None)
                if t:
                    pending[pool.submit(trades, t, *jobs[t])] = t
    new = pd.DataFrame(rows)
    combined = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(combined):
        save(combined, path)
    done_path.write_text("\n".join(sorted(done)) + "\n")
    remaining = len(jobs) - len(done)
    annotate(
        "notice",
        f"{args.series}: fetched {fetched} markets ({len(new)} trades), failed "
        f"{failed}, remaining {remaining}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - report any failure as an annotation
        annotate("error", traceback.format_exc()[-3000:])
        sys.exit(1)
