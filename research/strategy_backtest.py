"""Exploratory findings and the 2026-27 rule replayed on 2025-26.

Everything here was run after the pre-registered market test failed, on the
same season, so it is in-sample: it shows why the rule was chosen and what a
$500 bankroll would have done if the rule had been known in advance. The
real test is the 2026-27 season (RESEARCH_LOG.md).

Reads outputs/market_test/priced_markets.csv.gz and data/kalshi_history/.
Writes outputs/strategy/.

Run from the repository root: python run_strategy_backtest.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi.prices import price_at_tip
from kalshi.stats import clustered_mean_t
from kalshi.strategy import (
    bootstrap,
    historical_trades,
    qualifying,
    settle,
    simulate,
    summarize_path,
)
from markets.nba.market import load_history

OUT = Path("outputs/strategy")
BANKROLL = 500.0
SPLIT = pd.Timestamp("2026-02-01", tz="America/New_York")


def quotes_function(priced, candles):
    cache = {}

    def quotes_at(hours):
        if hours not in cache:
            cache[hours] = price_at_tip(priced[["ticker", "tip_utc"]], candles, hours)
        return cache[hours]

    return quotes_at


def stats(trades):
    mean, t = clustered_mean_t(trades.pnl, trades.game_id)
    early = trades[pd.to_datetime(trades.tip_utc, utc=True) < SPLIT]
    late = trades[pd.to_datetime(trades.tip_utc, utc=True) >= SPLIT]
    return {
        "trades": len(trades),
        "cents_per_contract": 100 * mean,
        "t": t,
        "cents_at_mid_before_costs": 100 * trades.gross_at_mid.mean(),
        "avg_cost_cents": 100 * trades.cost.mean(),
        "before_feb_cents": 100 * early.pnl.mean(),
        "from_feb_cents": 100 * late.pnl.mean(),
        "share_yes": float((trades.side == "yes").mean()),
    }


def exploratory_grid(priced, quotes_at):
    """Profit by hours before tip-off and by spread, single look, 5c edge."""
    rows = []
    for hours in [0, 2, 4, 6]:
        q = priced.drop(columns=["yes_bid", "yes_ask"]).merge(
            quotes_at(hours)[["ticker", "yes_bid", "yes_ask"]], on="ticker"
        )
        q["mid"] = (q.yes_bid + q.yes_ask) / 2
        for label, column in [("model", "p_model"), ("no teammates", "p_no_teammates")]:
            t = settle(qualifying(q.assign(p_model=q[column]), max_spread=1.0))
            t["bucket"] = pd.cut(
                t.spread,
                [0, 0.015, 0.025, 0.045, 1],
                labels=["1c", "2c", "3-4c", "5c+"],
            )
            for bucket, g in t.groupby("bucket", observed=True):
                mean, tt = clustered_mean_t(g.pnl, g.game_id)
                rows.append(
                    {
                        "hours_before_tip": hours,
                        "features": label,
                        "spread": bucket,
                        "trades": len(g),
                        "cents_per_contract": 100 * mean,
                        "t": tt,
                        "cents_at_mid": 100 * g.gross_at_mid.mean(),
                    }
                )
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    priced = pd.read_csv("outputs/market_test/priced_markets.csv.gz", low_memory=False)
    priced["tip_utc"] = pd.to_datetime(priced.tip_utc, utc=True)
    priced = priced.drop(columns=["end_ts", "price", "volume"])
    _, candles = load_history(Path("data/kalshi_history"))
    quotes_at = quotes_function(priced, candles)

    grid = exploratory_grid(priced, quotes_at)
    grid.round(3).to_csv(OUT / "exploratory_grid.csv", index=False)

    results = {}
    trades = {}
    for label, column in [("model", "p_model"), ("no_teammates", "p_no_teammates")]:
        t = historical_trades(priced.assign(p_model=priced[column]), quotes_at)
        trades[label] = t
        results[label] = {"rule": stats(t)}
        replay = simulate(t, BANKROLL)
        results[label]["replay_500"] = summarize_path(replay, BANKROLL)
        roi = (t.pnl / t.cost).mean()
        roi_mid = (t.gross_at_mid / t.cost).mean()
        results[label]["rule"]["return_per_dollar_pct"] = 100 * roi
        scenarios = {
            "as_in_2025_26": 0.0,
            "half_the_edge": -roi / 2,
            "no_edge_after_costs": -roi,
            "model_no_better_than_market": -roi_mid,
        }
        boot = {}
        for name, shift in scenarios.items():
            b = bootstrap(t, paths=2000, bankroll=BANKROLL, roi_shift=shift)
            boot[name] = {
                "return_per_dollar_pct": 100 * (roi + shift),
                "median_end": float(b.end.median()),
                "p10_end": float(b.end.quantile(0.10)),
                "p90_end": float(b.end.quantile(0.90)),
                "chance_of_loss": float((b.end < BANKROLL).mean()),
                "chance_down_half": float((b.end < BANKROLL / 2).mean()),
                "median_max_drawdown_pct": float(b.max_drawdown_pct.median()),
            }
        results[label]["bootstrap_500"] = boot
        segments = {}
        for name, part in [
            ("regular_season", t[t.season_type == "regular"]),
            ("play_in_and_playoffs", t[t.season_type != "regular"]),
        ]:
            part_roi = (part.pnl / part.cost).mean()
            seg = {"rule": stats(part)}
            seg["rule"]["return_per_dollar_pct"] = 100 * part_roi
            seg["replay_500"] = summarize_path(simulate(part, BANKROLL), BANKROLL)
            for scenario, shift in [
                ("as_in_2025_26", 0.0),
                ("half_the_edge", -part_roi / 2),
                ("no_edge_after_costs", -part_roi),
            ]:
                b = bootstrap(part, paths=2000, bankroll=BANKROLL, roi_shift=shift)
                seg[scenario] = {
                    "median_end": float(b.end.median()),
                    "p10_end": float(b.end.quantile(0.10)),
                    "p90_end": float(b.end.quantile(0.90)),
                    "chance_of_loss": float((b.end < BANKROLL).mean()),
                    "median_max_drawdown_pct": float(b.max_drawdown_pct.median()),
                }
            segments[name] = seg
        results[label]["by_part_of_season"] = segments
        if label == "model":
            replay.to_csv(OUT / "replay_500.csv", index=False)
            t.to_csv(OUT / "trades_2025_26.csv.gz", index=False)

    text = json.dumps(results, indent=2, default=_plain)
    (OUT / "results.json").write_text(text + "\n")
    print(grid.round(2).to_string(index=False))
    print(text)


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
