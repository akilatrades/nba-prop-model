"""Walk-forward backtest of the single-leg models, 2019 to the latest week.

Run from the repository root: python -m research.nfl_backtest
"""

import json
from pathlib import Path

import pandas as pd

from markets.nfl.chart import plot_calibration
from markets.nfl.data import load, load_injuries, player_games
from markets.nfl.evaluate import (
    by_line,
    by_season,
    calibration,
    injury_effect,
    summarize,
    walk_forward,
)
from markets.nfl.features import build_features, model_rows

OUT = Path("outputs/nfl/backtest")
FIRST_TEST_SEASON = 2019


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    games, players = load()
    injuries = load_injuries()
    rows = model_rows(build_features(player_games(games, players), injuries))
    seasons = range(FIRST_TEST_SEASON, int(rows.season.max()) + 1)
    results = walk_forward(rows, seasons)
    # Same backtest without the injury report, to measure what it adds.
    without = walk_forward(
        model_rows(build_features(player_games(games, players))), seasons
    )
    injury_effect(results, without, rows).round(4).to_csv(
        OUT / "injury_effect.csv", index=False
    )

    summary = summarize(results).round(4)
    lines = by_line(results).round(4)
    calib = calibration(results).round(4)
    seasonal = by_season(results).round(4)
    summary.to_csv(OUT / "summary.csv", index=False)
    lines.to_csv(OUT / "by_line.csv", index=False)
    calib.to_csv(OUT / "calibration.csv", index=False)
    seasonal.to_csv(OUT / "season_bias.csv")
    plot_calibration(
        calib,
        OUT / "calibration.svg",
        f"Single-leg calibration, {seasons[0]}–{seasons[-1]} "
        f"({len(results):,} predictions)",
    )
    meta = {
        "test_seasons": [seasons[0], seasons[-1]],
        "predictions": len(results),
        "player_games": int(
            results.drop_duplicates(["season", "week", "player_id", "stat"]).shape[0]
        ),
        "data_retrieval": json.loads(Path("data/nfl/retrieval.json").read_text()),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(calib.to_string(index=False))


if __name__ == "__main__":
    main()
