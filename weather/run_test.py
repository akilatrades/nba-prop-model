"""Run the Miami daily-high test once, as fixed in RESEARCH_LOG.md (2026-10-15).

Writes weather/out/results.json and the trade lists. Usage:
    python weather/run_test.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import forecast as fc
import wx

from nba_model.maker import maker_fee, orders
from nba_model.market import clustered_logit, clustered_mean_t, logit

OUT = Path("weather/out")
START, END = pd.Timestamp("2024-05-01"), pd.Timestamp("2026-10-14")
LAST_YEAR = pd.Timestamp("2025-10-15")
TAKER_EDGE = 0.05
MAKER_EDGE = 0.03
FEE_SHARE = 0.25
STAKE, DAY_CAP, MAX_DOLLARS, MAX_CONTRACTS = 0.01, 0.30, 25.0, 100


def load():
    cli = wx.load_cli()
    cli.index = pd.to_datetime(cli.index)
    daily = fc.daily_forecasts()
    model = {}
    for lead, frame in daily.items():
        frame.index = pd.to_datetime(frame.index)
        model[lead] = fc.rolling_model(frame, cli)
    m = wx.load_markets()
    m["date"] = pd.to_datetime(m.date)
    m = m[m.yes.notna() & m.date.between(START, END)]
    c = pd.read_parquet(wx.DATA / "kalshi_candles.parquet")
    c = c.merge(m[["ticker", "date", "lo", "hi", "yes", "strike_type"]], on="ticker")
    c = c[(c.yes_bid > 0) & (c.yes_ask < 1) & (c.yes_ask > c.yes_bid)].copy()
    start = c.date.dt.tz_localize("UTC") + pd.Timedelta(hours=5)
    c["h"] = (
        pd.to_datetime(c.end_ts, unit="s", utc=True) - start
    ).dt.total_seconds() / 3600
    c["mid"] = (c.yes_bid + c.yes_ask) / 2
    return cli, model, m, c


def forecast_prob(rows, model, lead):
    r = model[lead]
    mu = rows.date.map(r.mu)
    sigma = rows.date.map(r.sigma)
    return pd.Series(
        fc.prob_between(mu, sigma, rows.lo, rows.hi), index=rows.index
    ).where(mu.notna())


def price_families(c, model):
    # Forecast family: day 2 before 02:00 UTC on D (h < -3), day 1 from then to h = 8.
    f = c[c.h <= 8].copy()
    p2 = forecast_prob(f, model, 2)
    p1 = forecast_prob(f, model, 1)
    f["p_model"] = np.where(f.h < -3, p2, p1)
    f = f.dropna(subset=["p_model"])

    # On-the-day family: h = 9 to 23 on the hour.
    d = c[c.h.between(9, 23) & (c.h == c.h.round())].copy()
    d["h"] = d.h.astype(int)
    states = pd.read_parquet(OUT / "obs_states.parquet")
    states["date"] = pd.to_datetime(states.date)
    states = fc.intraday_frame(states, model[1].mu)
    d = d.merge(
        states[["date", "h", "base", "src", "gap_bin", "mh", "m6"]], on=["date", "h"]
    )
    parts = []
    for month in sorted(d.date.dt.to_period("M").unique()):
        first = month.to_timestamp()
        pooled, cells = fc.offset_table(
            states[states.date < first - pd.Timedelta(days=1)]
        )
        rows = d[d.date.dt.to_period("M") == month].copy()
        rows["p_day"] = fc.intraday_probs(
            rows, pooled, cells, rows.lo.to_numpy(), rows.hi.to_numpy()
        )
        parts.append(rows)
    d = pd.concat(parts)
    d["p_fc"] = forecast_prob(d, model, 1)
    d["p_model"] = fc.blend(d.p_day, d.p_fc)
    d = d.dropna(subset=["p_model"])
    return {"forecast": f, "on_the_day": d}


def test_a(rows):
    beta, se = clustered_logit(
        rows.yes, np.column_stack([logit(rows.mid), logit(rows.p_model)]), rows.date
    )
    brier_model = float(((rows.p_model - rows.yes) ** 2).mean())
    brier_mid = float(((rows.mid - rows.yes) ** 2).mean())
    return {
        "candles": len(rows),
        "markets": int(rows.ticker.nunique()),
        "dates": int(rows.date.nunique()),
        "coef_market": float(beta[1]),
        "t_market": float(beta[1] / se[1]),
        "coef_model": float(beta[2]),
        "t_model": float(beta[2] / se[2]),
        "brier_model": brier_model,
        "brier_mid": brier_mid,
        "passes": bool(beta[2] / se[2] >= 2.0),
    }


def taker_trades(rows):
    r = rows.copy()
    yes_edge = r.p_model - r.yes_ask - wx.fee(r.yes_ask)
    no_price = 1 - r.yes_bid
    no_edge = (1 - r.p_model) - no_price - wx.fee(no_price)
    r["side"] = np.where(yes_edge >= no_edge, "yes", "no")
    r["price"] = np.where(r.side == "yes", r.yes_ask, no_price).round(4)
    r["edge"] = np.maximum(yes_edge, no_edge)
    r = r[r.edge >= TAKER_EDGE].sort_values("end_ts")
    r = r.drop_duplicates("ticker", keep="first").copy()
    r["value"] = np.where(r.side == "yes", r.yes, 1 - r.yes)
    r["pnl"] = r.value - r.price - wx.fee(r.price)
    return r


def resting_orders(rows, trades):
    q = rows.copy()
    o = orders(q, min_edge=MAKER_EDGE, fee_share=FEE_SHARE).reset_index(drop=True)
    o["order_id"] = np.arange(len(o))
    keys = ["order_id", "ticker", "end_ts", "side", "yes_level", "improved"]
    tickers = o.ticker.unique()
    filled = []
    for i in range(0, len(tickers), 300):
        chunk = set(tickers[i : i + 300])
        t = trades[trades.ticker.isin(chunk)].merge(
            o.loc[o.ticker.isin(chunk), keys], on="ticker"
        )
        t = t[(t.ts > t.end_ts) & (t.ts <= t.end_ts + 3600)]
        level = t.yes_level
        same = (t.yes_price - level).abs() < 1e-6
        hit_yes = (t.side == "yes") & (t.taker_outcome_side == "no")
        hit_yes &= (t.yes_price < level - 1e-6) | (same & t.improved)
        hit_no = (t.side == "no") & (t.taker_outcome_side == "yes")
        hit_no &= (t.yes_price > level + 1e-6) | (same & t.improved)
        filled.append(t[hit_yes | hit_no].groupby("order_id")["count"].sum())
    filled = pd.concat(filled) if filled else pd.Series(dtype=float)
    o["traded_through"] = o.order_id.map(filled).fillna(0.0)
    o["value"] = np.where(o.side == "yes", o.yes, 1 - o.yes)
    o["pnl"] = o.value - o.price - maker_fee(o.price, FEE_SHARE)
    return o


def summary(t, split):
    mean, tstat = clustered_mean_t(t.pnl, t.date)
    first, second = t[t.date < split], t[t.date >= split]
    last = t[t.date >= LAST_YEAR]
    out = {
        "positions": len(t),
        "dates": int(t.date.nunique()),
        "cents_per_contract": 100 * float(mean) if len(t) else None,
        "t": float(tstat) if len(t) > 1 else None,
        "avg_price": float(t.price.mean()) if len(t) else None,
        "first_half_cents": 100 * float(first.pnl.mean()) if len(first) else None,
        "second_half_cents": 100 * float(second.pnl.mean()) if len(second) else None,
        "last_12m_cents": 100 * float(last.pnl.mean()) if len(last) else None,
        "last_12m_positions": len(last),
    }
    out["passes"] = bool(
        len(t) > 1
        and mean > 0
        and tstat >= 2.0
        and out["first_half_cents"] is not None
        and out["first_half_cents"] > 0
        and out["second_half_cents"] is not None
        and out["second_half_cents"] > 0
        and out["last_12m_cents"] is not None
        and out["last_12m_cents"] > 0
    )
    return out


def simulate(days, bankroll=500.0, maker=False, roi_shift=0.0):
    """days: list of (price, cap, value) arrays per day. Returns final bankroll path."""
    path = []
    for price, cap, value in days:
        start = bankroll
        scale = min(1.0, DAY_CAP / (STAKE * len(price)))
        budget = np.minimum(STAKE * start * scale, MAX_DOLLARS)
        size = np.minimum(np.floor(budget / price + 1e-9), MAX_CONTRACTS)
        size = np.minimum(size, np.floor(cap))
        rate = FEE_SHARE * 0.07 if maker else 0.07
        fee = np.ceil(np.round(100 * rate * size * price * (1 - price), 6)) / 100
        fee = np.where(size > 0, fee, 0.0)
        profit = float(np.sum(size * (value + roi_shift * price - price) - fee))
        bankroll = start + profit
        path.append((profit, bankroll, int((size > 0).sum()), int(size.sum())))
        if bankroll <= 1:
            break
    return path


def day_arrays(t, cap_column):
    return [
        (
            g.price.to_numpy(float),
            g[cap_column].to_numpy(float),
            g.value.to_numpy(float),
        )
        for _, g in t.sort_values("date").groupby("date")
    ]


def bootstrap(t, cap_column, maker, paths=2000, seed=0):
    recent = t[t.date >= LAST_YEAR]
    days = day_arrays(recent, cap_column)
    if not days:
        return None
    per_dollar = float((recent.pnl / recent.price).mean())
    rng = np.random.default_rng(seed)
    out = {}
    for label, shift in [
        ("full_edge", 0.0),
        ("half_edge", -per_dollar / 2),
        ("no_edge", -per_dollar),
    ]:
        ends, contracts = [], []
        for _ in range(paths):
            picks = rng.integers(0, len(days), len(days))
            path = simulate([days[i] for i in picks], maker=maker, roi_shift=shift)
            ends.append(path[-1][1])
            contracts.append(sum(p[3] for p in path))
        ends = np.array(ends)
        out[label] = {
            "median_end": float(np.median(ends)),
            "p10_end": float(np.percentile(ends, 10)),
            "p90_end": float(np.percentile(ends, 90)),
            "share_losing": float((ends < 500).mean()),
            "median_contracts": float(np.median(contracts)),
        }
    out["trading_days_in_last_12m"] = len(days)
    out["note"] = "each path redraws the last 12 months' trading days with replacement"
    return out


def breakdowns(t):
    t = t.copy()
    t["year"] = t.date.dt.year
    t["kind"] = np.where(t.strike_type == "between", "range", "tail")
    t["price_bin"] = pd.cut(t.price, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    t["hour_bin"] = pd.cut(t.h, [-40, -3, 8, 13, 19, 24])
    out = {}
    for column in ["year", "kind", "side", "price_bin", "hour_bin"]:
        g = t.groupby(column, observed=True).pnl.agg(["size", "mean"])
        out[column] = {
            str(k): [int(v["size"]), round(100 * v["mean"], 2)] for k, v in g.iterrows()
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _cli, model, _markets, candles = load()
    trades = pd.read_parquet(wx.DATA / "kalshi_trades.parquet")
    families = price_families(candles, model)
    split = START + (END - START) / 2
    results = {
        "test_dates": [str(START.date()), str(END.date())],
        "split": str(split.date()),
    }
    # Volume in the hour after each candle, as a cap on taker size.
    vol = trades[["ticker", "ts", "count"]]
    for name, rows in families.items():
        a = test_a(rows)
        b = taker_trades(rows)
        hour = b[["ticker", "end_ts"]].merge(vol, on="ticker")
        hour = hour[(hour.ts > hour.end_ts) & (hour.ts <= hour.end_ts + 3600)]
        b["next_hour_volume"] = b.set_index(["ticker", "end_ts"]).index.map(
            hour.groupby(["ticker", "end_ts"])["count"].sum()
        )
        b["next_hour_volume"] = b.next_hour_volume.fillna(0.0)
        c = resting_orders(rows, trades)
        filled = c[c.traded_through >= 1].copy()
        res = {
            "test_a": a,
            "test_b": summary(b, split),
            "test_c": summary(filled, split)
            | {"orders": len(c), "fill_rate": float((c.traded_through >= 1).mean())},
            "breakdown_b": breakdowns(b) if len(b) else {},
            "breakdown_c": breakdowns(filled) if len(filled) else {},
        }
        if res["test_b"]["passes"]:
            res["bankroll_b"] = bootstrap(b, "next_hour_volume", maker=False)
        if res["test_c"]["passes"]:
            res["bankroll_c"] = bootstrap(filled, "traded_through", maker=True)
        results[name] = res
        keep = [
            "ticker",
            "date",
            "end_ts",
            "h",
            "side",
            "price",
            "p_model",
            "mid",
            "yes",
            "pnl",
        ]
        b[keep + ["next_hour_volume"]].to_csv(OUT / f"taker_{name}.csv.gz", index=False)
        filled[keep + ["traded_through"]].to_csv(
            OUT / f"resting_{name}.csv.gz", index=False
        )
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
