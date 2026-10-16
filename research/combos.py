"""Test P3 (RESEARCH_LOG.md): do takers of Kalshi's NBA same-game combos overpay?

Writes outputs/parlays/combos.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/kalshi_history")
OUT = Path("outputs/parlays")
SEASON = (pd.Timestamp("2025-10-01").date(), pd.Timestamp("2026-06-30").date())
SPLIT = pd.Timestamp("2026-02-01").date()
PROP_SERIES = ("KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT")


def ratio_t(profit, paid, clusters):
    """Dollar-weighted return sum(profit)/sum(paid) and its t, clustered."""
    g = pd.DataFrame({"profit": profit, "paid": paid, "c": clusters}).groupby("c").sum()
    mean = g.profit.sum() / g.paid.sum()
    resid = g.profit - mean * g.paid
    n = len(g)
    se = np.sqrt((resid**2).sum() * n / (n - 1)) / g.paid.sum()
    return float(mean), float(mean / se) if se > 0 else np.nan


def load():
    m = pd.read_parquet(DATA / "combos_markets.parquet")
    t = pd.read_parquet(DATA / "combos_trades.parquet")
    m["game"] = m.mve_collection_ticker.fillna(m.event_ticker)
    code = m.game.str.split("-").str[1].str[:7]
    m["date"] = pd.to_datetime(code, format="%y%b%d", errors="coerce").dt.date
    m = m[m.result.isin(["yes", "no"]) & m.date.notna()]
    m = m[(m.date >= SEASON[0]) & (m.date <= SEASON[1])]
    t = t.merge(m[["ticker", "game", "date", "result", "n_legs", "legs"]], on="ticker")
    t["side"] = t.taker_outcome_side
    t = t[t.side.isin(["yes", "no"])].copy()
    t["price"] = np.where(t.side == "yes", t.yes_price, 1 - t.yes_price)
    t = t[(t.price > 0) & (t.price < 1)]
    t["paid"] = t.price * t["count"]
    t["payout"] = np.where(t.side == t.result, 1.0, 0.0) * t["count"]
    t["profit"] = t.payout - t.paid
    t["ts"] = pd.to_datetime(t.created_time, utc=True, format="ISO8601")
    return m, t


def summary(t):
    mean, tstat = ratio_t(t.profit, t.paid, t.game)
    first, second = t[t.date < SPLIT], t[t.date >= SPLIT]
    return {
        "trades": len(t),
        "combos": int(t.ticker.nunique()),
        "games": int(t.game.nunique()),
        "dollars_paid": float(t.paid.sum()),
        "avg_price": float(np.average(t.price, weights=t["count"])),
        "hit_rate_contracts": float(t.payout.sum() / t["count"].sum()),
        "taker_return_per_dollar": mean,
        "t": tstat,
        "first_half": ratio_t(first.profit, first.paid, first.game)[0]
        if len(first)
        else None,
        "second_half": ratio_t(second.profit, second.paid, second.game)[0]
        if len(second)
        else None,
    }


def leg_margin(t):
    """For props-only combos: quoted price against the product of the legs' mids."""
    candles = pd.concat(
        [pd.read_parquet(DATA / f"candles_{s}.parquet") for s in PROP_SERIES],
        ignore_index=True,
    )
    candles = candles[(candles.yes_bid > 0) & (candles.yes_ask < 1)]
    candles["mid"] = (candles.yes_bid + candles.yes_ask) / 2
    by_ticker = {k: g.sort_values("end_ts") for k, g in candles.groupby("ticker")}
    rows = []
    yes_trades = t[t.side == "yes"]
    for r in yes_trades.itertuples():
        legs = json.loads(r.legs)
        if not legs or not all(
            str(x["market_ticker"]).startswith(PROP_SERIES) for x in legs
        ):
            continue
        ts = int(r.ts.timestamp())
        product = 1.0
        for leg in legs:
            g = by_ticker.get(leg["market_ticker"])
            if g is None:
                product = np.nan
                break
            before = g[g.end_ts <= ts]
            if before.empty:
                product = np.nan
                break
            mid = float(before.mid.iloc[-1])
            product *= mid if leg["side"] == "yes" else 1 - mid
        if np.isfinite(product):
            rows.append(
                {
                    "price": r.price,
                    "product": product,
                    "hit": float(r.result == "yes"),
                    "count": r.count,
                    "n_legs": r.n_legs,
                    "game": r.game,
                }
            )
    x = pd.DataFrame(rows)
    if x.empty:
        return {"combos_priced": 0}
    w = x["count"]
    return {
        "trades_priced": len(x),
        "avg_quote": float(np.average(x.price, weights=w)),
        "avg_product_of_leg_mids": float(np.average(x["product"], weights=w)),
        "hit_rate": float(np.average(x.hit, weights=w)),
        "quote_over_product": float(np.average(x.price / x["product"], weights=w)),
        "by_legs": {
            int(k): {
                "trades": len(g),
                "quote": float(np.average(g.price, weights=g["count"])),
                "product": float(np.average(g["product"], weights=g["count"])),
                "hit": float(np.average(g.hit, weights=g["count"])),
            }
            for k, g in x.groupby("n_legs")
        },
    }


def main():
    m, t = load()
    res = {
        "combos_listed_in_season": len(m),
        "taker_side_counts": t.side.value_counts().to_dict(),
        "all_takers": summary(t),
        "by_side": {s: summary(g) for s, g in t.groupby("side")},
        "by_legs": {
            str(k if k < 5 else "5+"): summary(g)
            for k, g in t.assign(n=np.minimum(t.n_legs, 5)).groupby("n")
        },
    }
    a = res["all_takers"]
    res["overpay_passes"] = bool(
        a["taker_return_per_dollar"] < 0
        and a["t"] <= -2
        and (a["first_half"] or 0) < 0
        and (a["second_half"] or 0) < 0
    )
    res["leg_margin_props_only"] = leg_margin(t)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "combos.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
