"""Save a snapshot of Kalshi's open NBA player-prop markets.

Runs on a schedule in GitHub Actions (see .github/workflows/kalshi.yml).
Writes data/kalshi/<season>/<UTC timestamp>.csv.gz, where the season is
named by the year it ends (2026-27 is 2027), as in the box-score data.

Run from the repository root: python log_kalshi.py
"""

from datetime import UTC, datetime
from pathlib import Path

from nba_model.kalshi import snapshot

OUT = Path("data/kalshi")


def main():
    now = datetime.now(UTC)
    frame = snapshot(now=now)
    season = now.year + 1 if now.month >= 8 else now.year
    folder = OUT / str(season)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{now:%Y-%m-%dT%H%MZ}.csv.gz"
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"Saved {len(frame)} markets to {path}")
    print(frame.groupby("series").size().to_string())


if __name__ == "__main__":
    main()
