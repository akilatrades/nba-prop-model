"""Walk-forward backtest of the single-stat models.

Development seasons (2021-22 to 2024-25) by default; ``--holdout`` runs the
2025-26 season once the model is frozen (see RESEARCH_LOG.md).

Run from the repository root: python run_backtest.py [--holdout]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from markets.nba.chart import plot_calibration
from markets.nba.evaluate import (
    by_line,
    by_season,
    calibration,
    summarize,
    walk_forward,
)
from markets.nba.features import build_features, model_rows

DEV_SEASONS = [2022, 2023, 2024, 2025]
HOLDOUT_SEASONS = [2026]
REGULAR_MINUTES = 26  # heavy-minute players: the ones Kalshi lists


def season_label(season):
    return f"{season - 1}-{str(season)[2:]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    seasons = HOLDOUT_SEASONS if args.holdout else DEV_SEASONS
    out = Path("outputs/backtest") / ("holdout" if args.holdout else "development")
    out.mkdir(parents=True, exist_ok=True)

    players = pd.read_parquet("data/player_games.parquet")
    games = pd.read_parquet("data/games.parquet")
    rows = model_rows(build_features(players, games))
    results = walk_forward(rows, seasons)
    results.to_parquet(out / "predictions.parquet", index=False)

    summary = summarize(results).round(4)
    heavy = summarize(results[results.ew_minutes >= REGULAR_MINUTES]).round(4)
    lines = by_line(results).round(4)
    calib = calibration(results).round(4)
    summary.to_csv(out / "summary.csv", index=False)
    heavy.to_csv(out / f"summary_{REGULAR_MINUTES}plus_minutes.csv", index=False)
    lines.to_csv(out / "by_line.csv", index=False)
    calib.to_csv(out / "calibration.csv", index=False)
    by_season(results).round(4).to_csv(out / "season_bias.csv")
    span = f"{season_label(seasons[0])} to {season_label(seasons[-1])}"
    if len(seasons) == 1:
        span = season_label(seasons[0])
    plot_calibration(
        calib,
        out / "calibration.svg",
        f"Calibration, {span} ({len(results):,} predictions)",
    )
    meta = {
        "seasons": seasons,
        "predictions": len(results),
        "player_games": int(results.drop_duplicates(["game_id", "player_id"]).shape[0]),
        "data_retrieval": json.loads(Path("data/retrieval.json").read_text()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(heavy.to_string(index=False))
    print(calib.to_string(index=False))


if __name__ == "__main__":
    main()
