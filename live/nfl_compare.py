"""Score the NFL model against Kalshi prices once games have been played.

Reads every saved Kalshi snapshot (data/kalshi) and model ladder
(outputs/projections/ladders), matches markets to model prices made before
kickoff, adds results, and writes outputs/market/.

Run from the repository root: python -m live.nfl_compare
"""

from pathlib import Path

import pandas as pd

from markets.nfl.data import load
from markets.nfl.market import compare, summarize

SNAPSHOTS = Path("data/nfl/kalshi")
LADDERS = Path("outputs/nfl/projections/ladders")
OUT = Path("outputs/nfl/market")


def read_all(folder):
    files = sorted(folder.rglob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def main():
    snapshots, ladders = read_all(SNAPSHOTS), read_all(LADDERS)
    if snapshots.empty or ladders.empty:
        print("Need at least one Kalshi snapshot and one model ladder.")
        return
    games, players = load()
    compared = compare(snapshots, ladders, games, players)
    OUT.mkdir(parents=True, exist_ok=True)
    compared.to_csv(
        OUT / "comparison.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    summary = summarize(compared)
    summary.round(4).to_csv(OUT / "summary.csv", index=False)
    print(f"Matched {len(compared):,} markets; {compared.over.notna().sum():,} settled")
    if not summary.empty:
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
