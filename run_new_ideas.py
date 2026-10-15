"""Tests of the three new ideas on 2025-26 (RESEARCH_LOG.md, 2026-10-15).

1. Maker: rest limit orders on the model's side, filled from trade prints.
2. Sportsbooks: see scripts/probe_espn_props.py (needs stored book lines).
3. Structure: ladder arbitrage, and double-doubles and triple-doubles priced
   from Kalshi's own 10+ markets with fitted correlations.

Writes outputs/new_ideas/results.json and supporting tables.

Run from the repository root: python run_new_ideas.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_model.correlation import (
    double_double,
    independent_double_double,
    triple_yes,
)
from nba_model.kalshi import (
    COMBO_SERIES,
    SERIES,
    SETTLED_FIELDS,
    fee_per_contract,
    parse,
)
from nba_model.maker import fills, first_orders, settle_orders
from nba_model.market import clustered_mean_t, load_history, match_markets, price_at_tip
from nba_model.strategy import one_per_player, qualifying, settle

HISTORY = Path("data/kalshi_history")
OUT = Path("outputs/new_ideas")
SPLIT = pd.Timestamp("2026-02-01", tz="America/New_York")
LOOKS = (5, 2)


def seconds(times):
    return pd.to_datetime(times, utc=True).map(lambda t: int(t.timestamp()))


def summary(frame, column, filled=None):
    f = frame if filled is None else frame[filled]
    if len(f) < 2:
        return {"positions": len(f)}
    mean, t = clustered_mean_t(f[column], f.game_id)
    tip = pd.to_datetime(f.tip_utc, utc=True)
    regular = f[f.season_type == "regular"]
    reg_mean, reg_t = (
        clustered_mean_t(regular[column], regular.game_id)
        if len(regular) > 1
        else (np.nan, np.nan)
    )
    return {
        "positions": len(f),
        "games": int(f.game_id.nunique()),
        "cents_per_contract": 100 * mean,
        "t": t,
        "regular_season_cents": 100 * reg_mean,
        "regular_season_t": reg_t,
        "playoffs_cents": 100 * f.loc[f.season_type != "regular", column].mean(),
        "before_feb_cents": 100 * f.loc[tip < SPLIT, column].mean(),
        "from_feb_cents": 100 * f.loc[tip >= SPLIT, column].mean(),
    }


def passes(s):
    return bool(
        s.get("cents_per_contract", -1) > 0
        and s.get("t", 0) >= 2.0
        and s.get("regular_season_cents", -1) > 0
        and s.get("before_feb_cents", -1) > 0
        and s.get("from_feb_cents", -1) > 0
    )


def load_trades():
    frames = [
        pd.read_parquet(p)
        for s in SERIES
        if (p := HISTORY / f"trades_{s}.parquet").exists()
    ]
    t = pd.concat(frames, ignore_index=True)
    t["ts"] = seconds(t.created_time)
    return t[~t.is_block_trade]


def idea_maker(priced, candles):
    trades = load_trades()
    looks = []
    for hours in LOOKS:
        q = price_at_tip(priced[["ticker", "tip_utc"]], candles, hours)
        q = priced.drop(columns=["yes_bid", "yes_ask"], errors="ignore").merge(
            q[["ticker", "end_ts", "yes_bid", "yes_ask"]], on="ticker"
        )
        q["tip_ts"] = seconds(q.tip_utc)
        looks.append((hours, q))
    out = {}
    for label, share in [("maker_fee_quarter", 0.25), ("no_maker_fee", 0.0)]:
        placed = first_orders(looks, fee_share=share)
        settled = settle_orders(placed, fills(placed, trades), fee_share=share)
        result = {
            "orders": len(settled),
            "fill_rate": float(settled.filled.mean()),
            "filled": summary(settled, "pnl_if_filled", settled.filled),
            "all_orders_as_if_filled": summary(settled, "pnl_if_filled"),
            "unfilled_orders_as_if_filled": summary(
                settled, "pnl_if_filled", ~settled.filled
            ),
            "fill_rate_by_side": settled.groupby("side").filled.mean().to_dict(),
            "fill_rate_improved_vs_joined": settled.groupby("improved")
            .filled.mean()
            .rename(index=str)
            .to_dict(),
        }
        result["passes"] = passes(result["filled"])
        out[label] = result
        if share == 0.25:
            settled.to_csv(OUT / "maker_orders.csv.gz", index=False)
    return out


def idea_ladder_arbitrage(markets, candles, players):
    m = match_markets(markets, players)
    m["key"] = m.ticker.str.rsplit("-", n=1).str[0]
    c = candles.merge(m[["ticker", "key", "line", "game_id"]], on="ticker")
    c = c[(c.yes_bid > 0) & (c.yes_ask < 1) & (c.yes_ask > c.yes_bid)]
    snapshots, rows = 0, []
    for (key, ts), g in c.groupby(["key", "end_ts"]):
        if len(g) < 2:
            continue
        snapshots += 1
        g = g.sort_values("line")
        asks, bids = g.yes_ask.to_numpy(), g.yes_bid.to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                no_cost = 1 - bids[j]
                cost = (
                    asks[i]
                    + fee_per_contract(asks[i])
                    + no_cost
                    + fee_per_contract(no_cost)
                )
                if cost < 1:
                    rows.append({"key": key, "end_ts": ts, "profit": 1 - cost})
    arb = pd.DataFrame(rows, columns=["key", "end_ts", "profit"])
    first = arb.sort_values("end_ts").drop_duplicates("key")
    return {
        "snapshots_with_two_or_more_lines": snapshots,
        "snapshots_with_arbitrage": len(arb),
        "player_stat_games_with_arbitrage": int(arb.key.nunique()),
        "profit_per_pair_first_opportunity_cents": 100 * float(first.profit.mean())
        if len(first)
        else None,
    }


def combo_markets(players):
    frames = []
    for series in COMBO_SERIES:
        m, c = (
            HISTORY / f"markets_{series}.parquet",
            HISTORY / f"candles_{series}.parquet",
        )
        if m.exists() and c.exists():
            frames.append((series, pd.read_parquet(m), pd.read_parquet(c)))
    return frames


def idea_combos(priced, candles, players, rhos, holdout):
    """Double- and triple-doubles from Kalshi's own 10+ markets plus correlation."""
    results = {}
    tens = priced[priced.line == 9.5][
        ["ticker", "game_id", "player_id", "stat", "tip_utc"]
    ]
    model = holdout[holdout.line == 9.5].pivot_table(
        index=["game_id", "player_id"], columns="stat", values="p_model"
    )
    for series, markets, combo_candles in combo_markets(players):
        markets = parse(markets[["series", *SETTLED_FIELDS]])
        m = match_markets(markets, players)
        m = m[m.result.isin(["yes", "no"]) & m.player_id.notna()].copy()
        m["player_id"] = m.player_id.astype("int64")
        m["game_id"] = m.game_id.astype("int64")
        m["over"] = m.result.eq("yes").astype(float)
        m["stat"] = COMBO_SERIES[series]
        looks, accuracy = [], None
        for hours in LOOKS:
            q = price_at_tip(m[["ticker", "tip_utc"]], combo_candles, hours)
            x = m.merge(q[["ticker", "yes_bid", "yes_ask"]], on="ticker")
            marg = price_at_tip(tens[["ticker", "tip_utc"]], candles, hours)
            marg = tens.merge(marg[["ticker", "yes_bid", "yes_ask"]], on="ticker")
            two = (
                (marg.yes_bid > 0) & (marg.yes_ask < 1) & (marg.yes_ask > marg.yes_bid)
            )
            marg = marg[two].assign(mid=(marg.yes_bid + marg.yes_ask) / 2)
            wide = marg.pivot_table(
                index=["game_id", "player_id"], columns="stat", values="mid"
            )
            probs = {}
            source_kalshi = 0
            for stat in ["points", "rebounds", "assists"]:
                key = pd.MultiIndex.from_frame(x[["game_id", "player_id"]])
                k = (
                    wide[stat].reindex(key).to_numpy()
                    if stat in wide
                    else np.full(len(x), np.nan)
                )
                mdl = (
                    model[stat].reindex(key).to_numpy()
                    if stat in model
                    else np.full(len(x), np.nan)
                )
                source_kalshi += np.isfinite(k).sum()
                probs[stat] = np.where(np.isfinite(k), k, mdl)
            okp = (
                np.isfinite(probs["points"])
                & np.isfinite(probs["rebounds"])
                & np.isfinite(probs["assists"])
            )
            x = x[okp].copy()
            pp, pr, pa = (probs[s][okp] for s in ["points", "rebounds", "assists"])
            if series == "KXNBA2D":
                x["p_model"] = double_double(pp, pr, pa, rhos)
                x["p_independent"] = independent_double_double(pp, pr, pa)
            else:
                x["p_model"] = triple_yes(pp, pr, pa, rhos)
                x["p_independent"] = pp * pr * pa
            x["mid"] = (x.yes_bid + x.yes_ask) / 2
            looks.append((hours, x))
            if hours == LOOKS[-1]:
                ok = (x.yes_bid > 0) & (x.yes_ask < 1) & (x.yes_ask > x.yes_bid)
                a = x[ok]
                accuracy = {
                    "markets": len(a),
                    "share_of_inputs_from_kalshi": float(
                        source_kalshi / max(3 * okp.size, 1)
                    ),
                    "outcome_rate": float(a.over.mean()),
                    "kalshi_mid_rate": float(a.mid.mean()),
                    "fair_rate": float(a.p_model.mean()),
                    "independent_rate": float(a.p_independent.mean()),
                    "brier_kalshi_mid": float(((a.mid - a.over) ** 2).mean()),
                    "brier_fair": float(((a.p_model - a.over) ** 2).mean()),
                    "brier_independent": float(
                        ((a.p_independent - a.over) ** 2).mean()
                    ),
                }
        taken, trades = set(), []
        for hours, x in looks:
            t = one_per_player(qualifying(x, max_spread=1.0, min_edge=0.05))
            t = t[[k not in taken for k in zip(t.game_id, t.player_id, t.stat)]]
            taken.update(zip(t.game_id, t.player_id, t.stat))
            trades.append(t.assign(hours_before=hours))
        t = settle(pd.concat(trades, ignore_index=True))
        res = {"accuracy": accuracy, "trades": summary(t, "pnl")}
        res["passes"] = passes(res["trades"]) if series == "KXNBA2D" else "report only"
        results[series] = res
        t.to_csv(OUT / f"trades_{series}.csv.gz", index=False)
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    players = pd.read_parquet("data/player_games.parquet")
    priced = pd.read_csv("outputs/market_test/priced_markets.csv.gz", low_memory=False)
    priced["tip_utc"] = pd.to_datetime(priced.tip_utc, utc=True)
    priced = priced.drop(columns=["end_ts", "price", "volume"])
    markets, candles = load_history(HISTORY)
    rhos = json.loads(Path("outputs/correlation/copula.json").read_text())
    rhos = [
        rhos["rho_points_rebounds"],
        rhos["rho_points_assists"],
        rhos["rho_rebounds_assists"],
    ]
    holdout = pd.read_parquet("outputs/backtest/holdout/predictions.parquet")

    results = {}
    if any((HISTORY / f"trades_{s}.parquet").exists() for s in SERIES):
        results["idea1_maker"] = idea_maker(priced, candles)
    results["idea3_ladder_arbitrage"] = idea_ladder_arbitrage(markets, candles, players)
    results["idea3_combos"] = idea_combos(priced, candles, players, rhos, holdout)
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
