"""Project every player for the next unplayed week of the current season.

Run from the repository root: python -m live.nfl_projections [season week]
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from markets.nfl.data import load, load_injuries
from markets.nfl.marginals import FEATURE_NAMES, STATS
from markets.nfl.project import project_week

OUT = Path("outputs/nfl/projections")
# Every run keeps its own over-probability ladder, so a market comparison can
# use the version made before each game kicked off.
LADDERS = OUT / "ladders"


def next_week(games):
    upcoming = games[games.home_score.isna() & games.total_line.notna()]
    if upcoming.empty:
        return None
    first = upcoming.sort_values(["season", "week"]).iloc[0]
    return int(first.season), int(first.week)


def main():
    games, players = load()
    if len(sys.argv) == 3:
        season, week = int(sys.argv[1]), int(sys.argv[2])
    else:
        upcoming = next_week(games)
        if upcoming is None:
            print("No upcoming games with lines; nothing to project.")
            return
        season, week = upcoming
    projections, models, ladder = project_week(
        games, players, season, week, load_injuries()
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{season}_week{week:02d}.csv"
    projections.sort_values(["stat", "expected"], ascending=[True, False]).to_csv(
        path, index=False
    )

    made = datetime.now(UTC)
    ladder.insert(0, "made_utc", made.isoformat(timespec="seconds"))
    LADDERS.mkdir(parents=True, exist_ok=True)
    ladder.to_csv(
        LADDERS / f"{season}_week{week:02d}_{made:%Y-%m-%dT%H%MZ}.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    coefficients = [
        {"stat": name, "feature": feature, "coefficient": round(value, 4)}
        for name, model in models.items()
        for feature, value in zip(
            FEATURE_NAMES[name], model.coefficients(), strict=True
        )
    ]
    pd.DataFrame(coefficients).to_csv(OUT / "coefficients.csv", index=False)
    counts = projections.groupby("stat").size().reindex(list(STATS))
    print(f"Wrote {path} and a {len(ladder):,}-row probability ladder")
    print(counts.to_string())


if __name__ == "__main__":
    main()
