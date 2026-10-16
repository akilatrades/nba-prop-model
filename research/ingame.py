"""The in-game test (RESEARCH_LOG.md, 2026-10-15): live game state vs Kalshi.

1. Training states every six game minutes for 2023-24 and 2024-25.
2. Calibration check: fit on 2023-24, score 2024-25.
3. Fit on both seasons; at every hourly Kalshi candle during a 2025-26 game,
   price the market from the game state at that moment; run tests A and B.

Downloads play-by-play from the hoopR data repository into data/pbp/ if
missing (about 22 MB a season).

Run from the repository root: python -m research.ingame [--calibration-only]
"""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi.stats import clustered_mean_t, test_a
from kalshi.strategy import one_per_player, qualifying, settle
from markets.nba.ingame import STATS, InGameModel, game_states, load_pbp
from markets.nba.market import load_history

PBP = Path("data/pbp")
PBP_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/"
    "nba/pbp/parquet/play_by_play_{season}.parquet"
)
OUT = Path("outputs/ingame")
CHECKPOINTS = [360.0 * k for k in range(1, 8)]  # every six game minutes
SPLIT = pd.Timestamp("2026-02-01", tz="America/New_York")
LINES = {
    "points": [9.5, 19.5, 29.5],
    "rebounds": [5.5, 9.5],
    "assists": [3.5, 7.5],
    "threes": [1.5, 3.5],
}


def pregame(predictions):
    """Pre-game expected value and recent minutes per player-game and stat."""
    p = predictions.drop_duplicates(["game_id", "player_id", "stat"])
    wide = p.pivot_table(index=["game_id", "player_id"], columns="stat", values="mean")
    wide.columns = [f"pregame_{c}" for c in wide.columns]
    minutes = p.groupby(["game_id", "player_id"]).ew_minutes.first()
    return wide.join(minutes).reset_index()


def training_states(season, predictions, box):
    pbp = load_pbp(PBP / f"play_by_play_{season}.parquet")
    pre = pregame(predictions)
    rows = []
    for gid, game in pbp.groupby("game_id"):
        players = pre[pre.game_id == gid]
        if players.empty:
            continue
        st = game_states(game, list(players.player_id), CHECKPOINTS)
        if st.empty:
            continue
        st["game_id"] = gid
        rows.append(st)
    states = pd.concat(rows, ignore_index=True).merge(pre, on=["game_id", "player_id"])
    final = box[["game_id", "player_id", *STATS]]
    states = states.merge(final, on=["game_id", "player_id"])
    for stat in STATS:
        states[f"{stat}_remaining"] = (states[stat] - states[f"{stat}_so_far"]).clip(
            lower=0
        )
    states["season"] = season
    needed = ["elapsed", "home_margin", "minutes_so_far", "ew_minutes"]
    return states.dropna(subset=needed)


def with_pregame_mean(rows, stat):
    return rows.assign(pregame_mean=rows[f"pregame_{stat}"])


def fit_models(states):
    return {s: InGameModel(s).fit(with_pregame_mean(states, s)) for s in STATS}


def calibration(models, states):
    out = []
    for stat in STATS:
        rows = with_pregame_mean(states, stat)
        for line in LINES[stat]:
            threshold = line + 0.5
            open_ = rows[rows[f"{stat}_so_far"] < threshold]
            p = models[stat].prob_at_least(open_, threshold)
            happened = (open_[stat] >= threshold).astype(float)
            out.append(
                {
                    "stat": stat,
                    "threshold": threshold,
                    "states": len(open_),
                    "predicted": float(p.mean()),
                    "actual": float(happened.mean()),
                    "brier": float(((p - happened) ** 2).mean()),
                }
            )
    return pd.DataFrame(out)


def live_snapshots(priced, candles, pbp, pre, lag=0):
    """Kalshi quotes during games, with the game state at the quote's time.

    ``lag`` seconds: use the state as it was that long before the quote, to
    see how much the result depends on knowing plays the instant they happen.
    """
    c = candles.merge(priced[["ticker", "game_id", "player_id", "stat"]], on="ticker")
    bounds = pbp.dropna(subset=["ts"]).groupby("game_id").ts.agg(["min", "max"])
    c = c.merge(bounds, left_on="game_id", right_index=True)
    c = c[(c.end_ts > c["min"] + 300) & (c.end_ts < c["max"] - 300)]
    c = c[(c.yes_bid > 0) & (c.yes_ask < 1) & (c.yes_ask > c.yes_bid)]
    states = []
    for gid, group in c.groupby("game_id"):
        game = pbp[pbp.game_id == gid].dropna(subset=["ts"])
        roster = list(group.player_id.unique())
        times = [t - lag for t in sorted(group.end_ts.unique())]
        st = game_states(game, roster, times, by="ts")
        if not st.empty:
            states.append(st.assign(game_id=gid))
    states = pd.concat(states, ignore_index=True)
    states["end_ts"] = states.pop("time") + lag
    snap = c.merge(states, on=["game_id", "player_id", "end_ts"])
    return snap.merge(pre, on=["game_id", "player_id"])


def download_pbp(seasons=(2024, 2025, 2026)):
    PBP.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        path = PBP / f"play_by_play_{season}.parquet"
        if not path.exists():
            urllib.request.urlretrieve(PBP_URL.format(season=season), path)


def price_snapshots(snap, meta, models):
    snap = snap.merge(meta.drop(columns=["game_id", "player_id", "stat"]), on="ticker")
    threshold = snap.line + 0.5
    parts = []
    for stat in STATS:
        part = snap[snap.stat == stat].copy()
        rows = with_pregame_mean(part, stat)
        part["p_model"] = models[stat].prob_at_least(rows, threshold.loc[part.index])
        part["reached"] = part[f"{stat}_so_far"] >= threshold.loc[part.index]
        parts.append(part)
    snap = pd.concat(parts)
    snap = snap[~snap.reached & (snap.fouls_so_far < 6)].copy()
    snap["mid"] = (snap.yes_bid + snap.yes_ask) / 2
    return snap


def first_trades(snap):
    taken, trades = set(), []
    for _, group in snap.sort_values("end_ts").groupby("end_ts", sort=True):
        t = one_per_player(qualifying(group, max_spread=1.0, min_edge=0.05))
        keep = [k not in taken for k in zip(t.game_id, t.player_id, t.stat)]
        t = t[np.array(keep, dtype=bool)]
        taken.update(zip(t.game_id, t.player_id, t.stat))
        trades.append(t)
    return settle(pd.concat(trades, ignore_index=True))


def trade_summary(t):
    if len(t) < 2:
        return {"trades": len(t)}
    mean, tt = clustered_mean_t(t.pnl, t.game_id)
    regular = t[t.season_type == "regular"]
    reg_mean, reg_t = (
        clustered_mean_t(regular.pnl, regular.game_id)
        if len(regular) > 1
        else (np.nan, np.nan)
    )
    tip = pd.to_datetime(t.tip_utc, utc=True)
    return {
        "trades": len(t),
        "games": int(t.game_id.nunique()),
        "cents_per_contract": 100 * mean,
        "t": tt,
        "regular_season_cents": 100 * reg_mean,
        "regular_season_t": reg_t,
        "playoffs_cents": 100 * t.loc[t.season_type != "regular", "pnl"].mean(),
        "before_feb_cents": 100 * t.loc[tip < SPLIT, "pnl"].mean(),
        "from_feb_cents": 100 * t.loc[tip >= SPLIT, "pnl"].mean(),
        "share_yes": float((t.side == "yes").mean()),
        "median_spread_cents": 100 * float((t.yes_ask - t.yes_bid).median()),
        "median_hour_volume": float(pd.to_numeric(t.volume, errors="coerce").median())
        if "volume" in t
        else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    download_pbp()
    box = pd.read_parquet("data/player_games.parquet")
    dev = pd.read_parquet("outputs/backtest/development/predictions.parquet")
    hold = pd.read_parquet("outputs/backtest/holdout/predictions.parquet")
    train = {s: training_states(s, dev[dev.season == s], box) for s in (2024, 2025)}

    check = calibration(fit_models(train[2024]), train[2025])
    check.round(4).to_csv(OUT / "calibration_2024_25.csv", index=False)
    print(check.round(3).to_string(index=False))
    if args.calibration_only:
        return

    models = fit_models(pd.concat(train.values(), ignore_index=True))
    priced = pd.read_csv("outputs/market_test/priced_markets.csv.gz", low_memory=False)
    priced["tip_utc"] = pd.to_datetime(priced.tip_utc, utc=True)
    _, candles = load_history(Path("data/kalshi_history"))
    pbp = load_pbp(PBP / "play_by_play_2026.parquet")
    pre = pregame(hold)
    meta = priced.drop(
        columns=["yes_bid", "yes_ask", "end_ts", "price", "volume", "mid"]
    )
    snap = price_snapshots(live_snapshots(priced, candles, pbp, pre), meta, models)
    snap.to_parquet(OUT / "snapshots.parquet", index=False)
    results = {"snapshots": len(snap), "markets": int(snap.ticker.nunique())}
    results["test_a"] = test_a(snap)
    results["brier_market_mid"] = float(((snap.mid - snap.over) ** 2).mean())
    results["brier_model"] = float(((snap.p_model - snap.over) ** 2).mean())
    t = first_trades(snap)
    results["test_b"] = trade_summary(t)
    b = results["test_b"]
    results["test_b_passes"] = bool(
        b["cents_per_contract"] > 0
        and b["t"] >= 2.0
        and b["regular_season_cents"] > 0
        and b["before_feb_cents"] > 0
        and b["from_feb_cents"] > 0
    )
    robustness = {}
    for lag in (60, 180, 300):
        lagged = price_snapshots(
            live_snapshots(priced, candles, pbp, pre, lag), meta, models
        )
        robustness[f"state_{lag}s_before_quote"] = {
            "test_a_model_t": test_a(lagged)["t_model"],
            "trades": trade_summary(first_trades(lagged)),
        }
    t["left_min"] = (2880 - t.elapsed) / 60
    for label, part in [
        ("first_half", t[t.left_min > 24]),
        ("third_quarter", t[(t.left_min <= 24) & (t.left_min > 12)]),
        ("fourth_quarter", t[t.left_min <= 12]),
        ("spread_under_6c", t[(t.yes_ask - t.yes_bid) < 0.055]),
        ("spread_6c_plus", t[(t.yes_ask - t.yes_bid) >= 0.055]),
        ("player_on_court", t[t.on_court.astype(bool)]),
        ("player_on_bench", t[~t.on_court.astype(bool)]),
    ]:
        robustness[label] = trade_summary(part)
    results["robustness"] = robustness
    t.to_csv(OUT / "trades.csv.gz", index=False)
    text = json.dumps(results, indent=2, default=_plain)
    (OUT / "results.json").write_text(text + "\n")
    print(text)


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


if __name__ == "__main__":
    main()
