"""Save a snapshot of Kalshi's open NFL player-prop markets.

Runs on a schedule in GitHub Actions (see .github/workflows/kalshi.yml).
Writes data/kalshi/<season>/<UTC timestamp>.csv.gz.

Run from the repository root: python -m live.nfl_log_kalshi
"""

from datetime import UTC, datetime
from pathlib import Path

from markets.nfl.kalshi import snapshot

OUT = Path("data/nfl/kalshi")


def main():
    now = datetime.now(UTC)
    frame = snapshot(now=now)
    season = now.year if now.month >= 3 else now.year - 1
    folder = OUT / str(season)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{now:%Y-%m-%dT%H%MZ}.csv.gz"
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    counts = frame.groupby("series").size()
    print(f"Saved {len(frame)} markets to {path}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
