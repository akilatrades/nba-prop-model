"""The pre-registered market test: the frozen model against Kalshi, 2025-26.

Reads data/kalshi_history/ (from the backfill workflow) and the box scores,
prices every settled Kalshi market with the model refit before each month,
and runs tests A and B exactly as written in RESEARCH_LOG.md on 2026-10-15.

``--coverage`` only matches tickers to games and players and reports counts;
it reads no prices, results or volumes, so it can be run before the model
is frozen.

Run from the repository root: python -m research.market_test [--coverage]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi.prices import price_at_tip
from kalshi.stats import test_a, test_b_summary
from markets.nba.features import build_features, model_rows
from markets.nba.kalshi import SERIES
from markets.nba.market import (
    load_history,
    match_markets,
    model_prices,
    test_b,
    trades,
    two_sided,
)

HISTORY = Path("data/kalshi_history")
OUT = Path("outputs/market_test")
SEASON = 2026
TEAMMATE_COLUMNS = ("vacated_", "returning_")


def without_teammates(rows):
    rows = rows.copy()
    for column in rows.columns:
        if column.startswith(TEAMMATE_COLUMNS):
            rows[column] = 0.0
    return rows


def coverage(markets, matched, eligible):
    in_season = matched.season_type.isin(["regular", "play-in", "playoffs"])
    keys = pd.MultiIndex.from_frame(eligible[["game_id", "player_id"]])
    priced = pd.MultiIndex.from_frame(
        matched[["game_id", "player_id"]].fillna(-1).astype("int64")
    ).isin(keys)
    return {
        "markets_listed": len(markets),
        "by_series": markets.series.value_counts().to_dict(),
        "game_found": int(matched.game_id.notna().sum()),
        "regular_playin_playoffs": int(in_season.sum()),
        "player_found": int((matched.player_id.notna() & in_season).sum()),
        "model_can_price": int((priced & in_season).sum()),
        "unmatched_team_codes": sorted(
            set(markets.home[matched.game_id.isna()])
            | set(markets.away[matched.game_id.isna()])
        )[:40],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    players = pd.read_parquet("data/player_games.parquet")
    games = pd.read_parquet("data/games.parquet")
    rows = model_rows(build_features(players, games))
    markets, candles = load_history(HISTORY)
    matched = match_markets(markets, players)
    eligible = rows[rows.season == SEASON]
    report = coverage(markets, matched, eligible)
    print(json.dumps(report, indent=2, default=str))
    (OUT / "coverage.json").write_text(json.dumps(report, indent=2, default=str))
    if args.coverage:
        return

    sample = matched[
        matched.result.isin(["yes", "no"])
        & matched.player_id.notna()
        & matched.season_type.isin(["regular", "play-in", "playoffs"])
    ].copy()
    sample["player_id"] = sample.player_id.astype("int64")
    sample["game_id"] = sample.game_id.astype("int64")
    sample["over"] = sample.result.eq("yes").astype(float)
    requests = sample[["ticker", "game_id", "player_id", "stat", "line"]]

    priced = model_prices(rows, requests, SEASON)
    plain = model_prices(rows, requests, SEASON, transform=without_teammates)
    plain = plain[["ticker", "p_model"]].rename(columns={"p_model": "p_no_teammates"})

    results, frames = {}, {}
    for label, hours in [("at_tip", 0), ("two_hours_before", 2)]:
        quotes = price_at_tip(sample, candles, hours_before=hours)
        frame = (
            sample.merge(quotes, on="ticker")
            .merge(priced, on="ticker")
            .merge(plain, on="ticker")
        )
        frame = frame[two_sided(frame)].copy()
        frame["mid"] = (frame.yes_bid + frame.yes_ask) / 2
        frames[label] = frame
        a = test_a(frame)
        b, passes, made = test_b(frame)
        results[label] = {"test_a": a, "test_b": b, "test_b_passes": passes}
        if label == "at_tip":
            made.to_csv(OUT / "trades.csv.gz", index=False)
            frame.to_csv(OUT / "priced_markets.csv.gz", index=False)

    main_frame = frames["at_tip"]
    no_team = main_frame.assign(p_model=main_frame.p_no_teammates)
    results["without_teammate_features"] = {
        "test_a": test_a(no_team),
        "test_b": test_b(no_team)[0],
    }
    by_stat = {}
    for stat in SERIES.values():
        part = main_frame[main_frame.stat == stat]
        if len(part) > 100:
            by_stat[stat] = {
                "test_a": test_a(part),
                "test_b": test_b_summary(trades(part)),
            }
    results["by_stat"] = by_stat
    volume = pd.to_numeric(main_frame.volume_fp, errors="coerce")
    by_volume = {}
    for label, part in [
        ("volume below median", main_frame[volume < volume.median()]),
        ("volume at or above median", main_frame[volume >= volume.median()]),
    ]:
        by_volume[label] = test_b_summary(trades(part))
    results["by_volume"] = by_volume
    results["accuracy"] = {
        "markets": len(main_frame),
        "brier_model": float(((main_frame.p_model - main_frame.over) ** 2).mean()),
        "brier_market_mid": float(((main_frame.mid - main_frame.over) ** 2).mean()),
        "kalshi_result_matches_box_score": float(
            ((main_frame.actual > main_frame.line) == (main_frame.over == 1)).mean()
        ),
    }
    text = json.dumps(results, indent=2, default=_plain)
    (OUT / "results.json").write_text(text + "\n")
    print(text)


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
