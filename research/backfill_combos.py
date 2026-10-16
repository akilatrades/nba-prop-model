"""Download Kalshi's NBA same-game combo (parlay) markets and their trades.

Stage 1 lists every market in KXMVENBASINGLEGAME (historical endpoint, plus
settled ones still on the live endpoint) with its legs, result, volume and
last price. Stage 2 fetches the trade prints of a fixed random sample of the
combos that traded (seed 0), until --minutes runs out; it is resumable.

Writes data/kalshi_history/combos_markets.parquet and combos_trades.parquet.
Prints counts only.
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

from kalshi.api import fetch_markets, trades

SERIES = "KXMVENBASINGLEGAME"
OUT = Path("data/kalshi_history")
FIELDS = [
    "ticker",
    "event_ticker",
    "mve_collection_ticker",
    "created_time",
    "open_time",
    "close_time",
    "expected_expiration_time",
    "result",
    "settlement_value_dollars",
    "volume_fp",
    "last_price_dollars",
    "status",
]


def annotate(level, message):
    print(message)
    print(f"::{level}::" + message.replace("%", "%25").replace("\n", "%0A"))


def save(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def list_markets():
    rows = []
    for source, historical, status in [
        ("historical", True, None),
        ("live", False, "settled"),
    ]:
        got = fetch_markets(SERIES, status=status, historical=historical)
        for m in got:
            row = {k: m.get(k) for k in FIELDS}
            row["legs"] = json.dumps(m.get("mve_selected_legs") or [])
            row["n_legs"] = len(m.get("mve_selected_legs") or [])
            row["source"] = source
            rows.append(row)
        print(f"{source}: {len(got)}")
    frame = pd.DataFrame(rows).drop_duplicates("ticker", keep="last")
    volume = pd.to_numeric(frame.volume_fp, errors="coerce").fillna(0)
    counts = {
        "listed": len(frame),
        "by_source": frame.source.value_counts().to_dict(),
        "by_legs": frame.n_legs.value_counts().sort_index().to_dict(),
        "traded_settled": int(((volume > 0) & frame.result.isin(["yes", "no"])).sum()),
    }
    (OUT / "combos_counts.json").write_text(
        json.dumps(counts, indent=1, default=str) + "\n"
    )
    # Only combos that traded and settled are kept (the full list is 1.8 million rows).
    frame = frame[(volume > 0) & frame.result.isin(["yes", "no"])]
    save(frame, OUT / "combos_markets.parquet")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=240)
    parser.add_argument("--sample", type=int, default=60000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    deadline = time.monotonic() + 60 * args.minutes
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "combos_markets.parquet"
    markets = pd.read_parquet(path) if path.exists() else list_markets()
    volume = pd.to_numeric(markets.volume_fp, errors="coerce").fillna(0)
    traded = markets[(volume > 0) & markets.result.isin(["yes", "no"])]
    rng = np.random.default_rng(0)
    order = traded.ticker.to_numpy()[rng.permutation(len(traded))][: args.sample]
    source = dict(zip(markets.ticker, markets.source))
    tpath = OUT / "combos_trades.parquet"
    dpath = OUT / "combos_trades_done.txt"
    old = pd.read_parquet(tpath) if tpath.exists() else pd.DataFrame()
    done = set(dpath.read_text().split()) if dpath.exists() else set()
    todo = [t for t in order if t not in done]
    print(
        f"{len(markets)} combos listed, {len(traded)} traded and settled, {len(todo)} to fetch"
    )
    rows, failed = [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        it = iter(todo)
        pending = {}

        def submit():
            t = next(it, None)
            if t is not None and time.monotonic() < deadline:
                hist = source.get(t) == "historical"
                pending[pool.submit(trades, t, 0, 2_000_000_000, hist)] = t

        for _ in range(args.workers * 3):
            submit()
        while pending:
            future = next(as_completed(list(pending)))
            ticker = pending.pop(future)
            try:
                rows.extend(future.result())
                done.add(ticker)
            except (OSError, ValueError, KeyError):
                failed += 1
            submit()
    new = pd.DataFrame(rows)
    combined = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(combined):
        save(combined, tpath)
    dpath.write_text("\n".join(sorted(done)) + "\n")
    n_legs = markets.n_legs.value_counts().sort_index().to_dict()
    annotate(
        "notice",
        f"combos: {len(markets)} listed ({markets.source.value_counts().to_dict()}), "
        f"{len(traded)} traded and settled; legs {n_legs}; trades fetched for "
        f"{len(done)} combos ({len(combined)} prints), failed {failed}, "
        f"remaining {len(order) - len(done)}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - report any failure as an annotation
        annotate("error", traceback.format_exc()[-3000:])
        sys.exit(1)
