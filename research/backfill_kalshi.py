"""Download settled Kalshi NBA player-prop markets and their pre-game prices.

Stage 1 lists every settled market in one prop series from Kalshi's
historical endpoint and keeps those whose game falls between --start and
--end. Stage 2 fetches hourly candles (yes bid, yes ask, last price, volume,
open interest) for the hours around each game, for markets that settled
yes or no and traded at least once. Markets settled at a fixed value
("scalar", which Kalshi uses when the player does not play) are listed but
get no candles.

Kalshi's occurrence time for these markets is a scheduled time near the end
of the game (close time when it is missing), so the window runs from
WINDOW_BEFORE hours before it to one hour after; it always contains tip-off.

The run is resumable: tickers already in the candle file are skipped, and it
stops cleanly after --minutes, saving what it has. It prints counts only.
Prices are not examined here; see RESEARCH_LOG.md for when they are used.

Writes data/kalshi_history/markets_<series>.parquet and candles_<series>.parquet.

Usage: python scripts/backfill_kalshi.py --series KXNBAPTS --minutes 320
"""

import argparse
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi.api import candles, fetch_markets, settled_frame
from markets.nba.kalshi import ALL_SERIES, parse

OUT = Path("data/kalshi_history")
WINDOW_BEFORE = 10  # hours before the occurrence time
WINDOW_AFTER = 1


def list_markets(series, start, end):
    markets = fetch_markets(series, status=None, historical=True)
    frame = parse(settled_frame(markets, series))
    in_window = frame.game_date.between(
        pd.Timestamp(start).date(), pd.Timestamp(end).date()
    )
    return frame[in_window].drop_duplicates("ticker").reset_index(drop=True)


def needs_candles(markets):
    volume = pd.to_numeric(markets.volume_fp, errors="coerce").fillna(0)
    return markets[markets.result.isin(["yes", "no"]) & (volume > 0)]


def to_utc(column):
    return pd.to_datetime(column, utc=True, format="ISO8601", errors="coerce")


def seconds(timestamp):
    return int(timestamp.timestamp())


def save(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", required=True, choices=sorted(ALL_SERIES))
    parser.add_argument("--start", default="2025-10-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--minutes", type=float, default=320)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    deadline = time.monotonic() + 60 * args.minutes
    OUT.mkdir(parents=True, exist_ok=True)

    market_path = OUT / f"markets_{args.series}.parquet"
    candle_path = OUT / f"candles_{args.series}.parquet"
    if market_path.exists():
        markets = pd.read_parquet(market_path)
    else:
        markets = list_markets(args.series, args.start, args.end)
        save(markets, market_path)
    print(
        f"{args.series}: {len(markets)} settled markets between {args.start} and {args.end}"
    )
    print(markets.result.value_counts(dropna=False).to_string())

    done = pd.read_parquet(candle_path) if candle_path.exists() else pd.DataFrame()
    finished = set(done.ticker) if len(done) else set()
    empty_path = OUT / f"no_candles_{args.series}.txt"
    no_candles = set(empty_path.read_text().split()) if empty_path.exists() else set()
    todo = needs_candles(markets)
    todo = todo[~todo.ticker.isin(finished | no_candles)]
    print(f"{len(todo)} markets still need candles ({len(finished)} done)")
    if todo.empty:
        annotate("notice", f"{args.series}: complete, {len(finished)} markets")
        return

    # Occurrence time when listed, otherwise close time (both are after tip-off).
    occurrence = to_utc(todo.occurrence_datetime).fillna(to_utc(todo.close_time))
    todo, occurrence = todo[occurrence.notna()], occurrence[occurrence.notna()]
    jobs = list(
        zip(
            todo.ticker,
            (occurrence - pd.Timedelta(hours=WINDOW_BEFORE)).map(seconds),
            (occurrence + pd.Timedelta(hours=WINDOW_AFTER)).map(seconds),
        )
    )
    rows, fetched, failed = [], 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        pending = {}
        it = iter(jobs)
        for _ in range(args.workers * 4):
            job = next(it, None)
            if job:
                pending[pool.submit(candles, *job)] = job[0]
        while pending:
            for future in as_completed(list(pending)):
                ticker = pending.pop(future)
                try:
                    got = future.result()
                    if got:
                        rows.extend(got)
                    else:
                        no_candles.add(ticker)
                    fetched += 1
                except (OSError, ValueError, KeyError) as error:
                    # keep going; the next run retries it
                    failed += 1
                    if failed <= 5:
                        print(f"failed {ticker}: {error}")
                if time.monotonic() < deadline:
                    job = next(it, None)
                    if job:
                        pending[pool.submit(candles, *job)] = job[0]
                if fetched and fetched % 2000 == 0 and ticker not in no_candles:
                    print(f"  {fetched} fetched", flush=True)
                break

    new = pd.DataFrame(rows)
    combined = pd.concat([done, new], ignore_index=True) if len(done) else new
    if len(combined):
        save(combined.drop_duplicates(["ticker", "end_ts"]), candle_path)
    empty_path.write_text("\n".join(sorted(no_candles)) + ("\n" if no_candles else ""))
    remaining = len(todo) - fetched
    status = "complete" if remaining == 0 and failed == 0 else "incomplete: run again"
    annotate(
        "notice",
        f"{args.series}: fetched {fetched}, failed {failed}, "
        f"remaining {remaining}, {len(finished)} done before; {status}",
    )


def annotate(level, message):
    """Print a GitHub Actions annotation (readable through the API) and the text."""
    print(message)
    print(f"::{level}::" + message.replace("%", "%25").replace("\n", "%0A"))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - report any failure as an annotation
        annotate("error", traceback.format_exc()[-3000:])
        sys.exit(1)
