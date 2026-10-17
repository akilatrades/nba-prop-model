"""Desk-style improvements, replayed on the 2025-26 NBA "no" seller
(RESEARCH_LOG.md, 2026-10-17, "Desk studies"). Offline; nothing live changes.

1. Value per quote: a fill-probability model from what is known at posting
   (spread, price level, open interest, hour's volume, hours to tip) and a
   replay that spends the daily cap by expected value per dollar-hour instead
   of by model edge.
2. Inventory: is a night of "no" fills one position? Correlate the day's
   paper profit with league-wide scoring that night.
3. Sizing under uncertainty: the daily cap set by a shrunk, running estimate
   of the edge (fractional Kelly on the posterior) instead of a fixed 30%.
4. Attribution: on the fills with a sportsbook quote, how much of the profit
   is the premium (sold above the books' probability) and how much residual.

    python -m research.desk      # writes outputs/desk/*
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kalshi.stats import clustered_mean_t
from kalshi.strategy import MAX_CONTRACTS, MAX_ORDER_DOLLARS

ORDERS = Path("outputs/decompose/orders_no.csv.gz")
CANDLES = [
    Path(f"data/kalshi_history/candles_{s}.parquet")
    for s in ("KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT")
]
BOOK = Path("outputs/bookline/fills_with_book.csv.gz")
PLAYERS = Path("data/player_games.parquet")
OUT = Path("outputs/desk")
FLOW = 0.5
FEE_SHARE = 0.25


def load_orders():
    o = pd.read_csv(ORDERS, low_memory=False)
    o = o[o.model_edge.astype(float) >= 0].copy()
    o["filled"] = o.filled.astype(str).str.lower().eq("true")
    c = pd.concat(
        [pd.read_parquet(p) for p in CANDLES if p.exists()], ignore_index=True
    )
    c = c.rename(columns={"volume": "hour_volume"})[
        ["ticker", "end_ts", "hour_volume", "open_interest"]
    ]
    o = o.merge(c, on=["ticker", "end_ts"], how="left")
    o["day"] = (
        pd.to_datetime(o.tip_utc, utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.astype(str)
    )
    for col in (
        "price",
        "yes_bid",
        "yes_ask",
        "edge",
        "pnl_if_filled",
        "filled_contracts",
        "hours_before",
    ):
        o[col] = pd.to_numeric(o[col], errors="coerce")
    o["hour_volume"] = o.hour_volume.fillna(0.0)
    o["open_interest"] = o.open_interest.fillna(0.0)
    return o


# --- 1. fill probability and value per quote --------------------------------------


def features(o):
    return pd.DataFrame(
        {
            "spread": (o.yes_ask - o.yes_bid),
            "level": o.price,
            "level_sq": o.price**2,
            "log_oi": np.log1p(o.open_interest),
            "hours": o.hours_before,
        }
    )


def fill_model(o):
    """Logistic fill model fit on the first half of the season, scored on the second."""
    days = sorted(o.day.unique())
    cut = days[len(days) // 2]
    train, test = o[o.day < cut], o[o.day >= cut]
    model = LogisticRegression(max_iter=1000).fit(features(train), train.filled)
    p = model.predict_proba(features(test))[:, 1]
    # calibration by decile
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    cal = (
        pd.DataFrame({"p": p, "filled": test.filled.to_numpy(), "q": q})
        .groupby("q")
        .agg(predicted=("p", "mean"), actual=("filled", "mean"), n=("p", "size"))
    )
    coef = dict(zip(features(train).columns, model.coef_[0].round(3)))
    return model, test.assign(p_fill=p), cal, coef, cut


def replay(
    frame, start, rank, day_cap=0.30, stake=0.01, game_cap=0.10, flow=FLOW, seed=0
):
    """Bankroll replay with one ordering rule; returns path and per-day frame."""
    bank, path, days = start, [start], []
    for day, g in frame.groupby("day"):
        g = (
            g.sort_values(rank, ascending=False)
            if rank
            else g.sample(frac=1, random_state=seed)
        )
        room, per, profit, n_placed, n_filled = day_cap * bank, {}, 0.0, 0, 0
        for r in g.itertuples():
            budget = min(
                stake * bank,
                MAX_ORDER_DOLLARS,
                room,
                game_cap * bank - per.get(r.game_id, 0.0),
            )
            n = min(math.floor(budget / r.price + 1e-9), MAX_CONTRACTS)
            if n < 1:
                continue
            room -= n * r.price
            per[r.game_id] = per.get(r.game_id, 0.0) + n * r.price
            n_placed += 1
            got = min(n, math.floor(flow * r.filled_contracts))
            if got > 0:
                n_filled += 1
                profit += got * r.pnl_if_filled
        bank = max(bank + profit, 0.0)
        path.append(bank)
        days.append(
            {
                "day": day,
                "profit": profit,
                "placed": n_placed,
                "filled": n_filled,
                "bank": bank,
            }
        )
    return np.array(path), pd.DataFrame(days)


def drawdown(path):
    peak = np.maximum.accumulate(path)
    return float(((peak - path) / peak).max())


def study_value_per_quote(o):
    model, test, cal, coef, cut = fill_model(o)
    test = test.assign(
        ev_edge=test.edge,  # current rule: rank by model edge
        ev_quote=test.p_fill * test.edge,  # expected value per contract posted
        ev_per_dollar_hour=test.p_fill
        * test.edge
        / (test.price * test.hours_before.clip(lower=0.5)),
    )
    out = {
        "train_until": cut,
        "test_orders": int(len(test)),
        "coefficients": coef,
        "calibration": cal.round(3).reset_index().to_dict("records"),
    }
    for rank, label in [
        ("ev_edge", "rank by model edge (current)"),
        ("ev_quote", "rank by P(fill) x edge"),
        ("ev_per_dollar_hour", "rank by P(fill) x edge per dollar-hour"),
        (None, "random order (control)"),
    ]:
        path, days = replay(test, 500.0, rank)
        out[label] = {
            "end": round(float(path[-1])),
            "max_drawdown": round(drawdown(path), 3),
            "orders": int(days.placed.sum()),
            "fills": int(days.filled.sum()),
            "profit_per_100_orders": round(
                100 * float(days.profit.sum() / max(days.placed.sum(), 1)), 2
            ),
        }
    return out


# --- 2. inventory: one position per night? -----------------------------------------


def study_inventory(o):
    path, days = replay(o, 500.0, "edge")
    p = pd.read_parquet(PLAYERS)
    p["day"] = p.tip_utc.dt.tz_convert("America/New_York").dt.date.astype(str)
    season = p[p.day.isin(set(days.day))]
    nightly = season.groupby("day").agg(
        points=("points", "sum"), games=("game_id", "nunique")
    )
    nightly["points_per_game"] = nightly.points / nightly.games
    # Expected points per game that night from each team's season average to date
    nightly = nightly.join(days.set_index("day")[["profit", "filled"]], how="inner")
    nightly["pnl_per_fill"] = nightly.profit / nightly.filled.replace(0, np.nan)
    x = nightly.points_per_game - nightly.points_per_game.rolling(
        20, min_periods=5
    ).mean().shift(1)
    valid = nightly[x.notna() & nightly.pnl_per_fill.notna()]
    xv = x[valid.index]
    corr = float(np.corrcoef(xv, valid.pnl_per_fill)[0, 1]) if len(valid) > 5 else None
    beta = float(np.polyfit(xv, valid.pnl_per_fill, 1)[0]) if len(valid) > 5 else None
    hi = valid[xv > xv.quantile(0.75)]
    lo = valid[xv < xv.quantile(0.25)]
    return {
        "nights": int(len(valid)),
        "corr_pnl_vs_scoring_surprise": round(corr, 3) if corr is not None else None,
        "dollars_per_fill_per_extra_point_per_game": round(beta, 3)
        if beta is not None
        else None,
        "pnl_per_fill_high_scoring_nights": round(float(hi.pnl_per_fill.mean()), 3)
        if len(hi)
        else None,
        "pnl_per_fill_low_scoring_nights": round(float(lo.pnl_per_fill.mean()), 3)
        if len(lo)
        else None,
    }


# --- 3. sizing under uncertainty -------------------------------------------------------


def replay_adaptive(
    o,
    start=500.0,
    prior_n=500,
    target=0.03,
    base_cap=0.30,
    max_cap=0.70,
    min_cap=0.10,
    flow=FLOW,
):
    """Daily cap = base_cap x (shrunk running edge / target), clamped.

    The running edge is the mean per-contract profit of fills so far, shrunk
    toward zero with the weight of ``prior_n`` imaginary zero-edge fills.
    """
    bank, path, pnl, n = start, [start], 0.0, 0
    caps = []
    for day, g in o.groupby("day"):
        post = (pnl / (n + prior_n)) if n else 0.0  # shrunk mean per contract
        cap = (
            float(np.clip(base_cap * post / target, min_cap, max_cap))
            if n
            else base_cap
        )
        caps.append(cap)
        g = g.sort_values("edge", ascending=False)
        room, per, profit = cap * bank, {}, 0.0
        for r in g.itertuples():
            budget = min(
                0.01 * bank,
                MAX_ORDER_DOLLARS,
                room,
                0.10 * bank - per.get(r.game_id, 0.0),
            )
            k = min(math.floor(budget / r.price + 1e-9), MAX_CONTRACTS)
            if k < 1:
                continue
            room -= k * r.price
            per[r.game_id] = per.get(r.game_id, 0.0) + k * r.price
            got = min(k, math.floor(flow * r.filled_contracts))
            if got > 0:
                profit += got * r.pnl_if_filled
                pnl += got * r.pnl_if_filled
                n += got
        bank = max(bank + profit, 0.0)
        path.append(bank)
    return np.array(path), caps


def study_sizing(o):
    out = {}
    fixed, _ = replay(o, 500.0, "edge", day_cap=0.30)
    out["fixed 30% cap"] = {
        "end": round(float(fixed[-1])),
        "max_drawdown": round(drawdown(fixed), 3),
    }
    for prior in (250, 500, 1000):
        path, caps = replay_adaptive(o, prior_n=prior)
        out[f"adaptive cap, prior {prior} zero-edge fills"] = {
            "end": round(float(path[-1])),
            "max_drawdown": round(drawdown(path), 3),
            "cap_after_30_days": round(caps[min(30, len(caps) - 1)], 2),
            "cap_final": round(caps[-1], 2),
        }
    # Stress: edge halved and no edge
    m = o.pnl_if_filled.mean()
    for k, label in [(0.5, "edge halved"), (0.0, "no edge")]:
        oh = o.copy()
        oh["pnl_if_filled"] = oh.pnl_if_filled - (1 - k) * m
        f, _ = replay(oh, 500.0, "edge", day_cap=0.30)
        a, _ = replay_adaptive(oh, prior_n=500)
        out[f"stress {label}: fixed 30%"] = {
            "end": round(float(f[-1])),
            "max_drawdown": round(drawdown(f), 3),
        }
        out[f"stress {label}: adaptive"] = {
            "end": round(float(a[-1])),
            "max_drawdown": round(drawdown(a), 3),
        }
    return out


# --- 4. attribution --------------------------------------------------------------------


def study_attribution():
    if not BOOK.exists():
        return {"note": "no book sample"}
    b = pd.read_csv(BOOK, low_memory=False)
    b = b[b.p_book.notna()].copy()
    b["premium"] = (
        b.book_edge
    )  # (1 - p_book) - price - fee: what selling above the books' probability is worth
    b["residual"] = b.pnl_if_filled - b.premium
    total = float(b.pnl_if_filled.sum())
    mean_r, t_r = clustered_mean_t(b.residual, b.game_id)
    return {
        "covered_fills": int(len(b)),
        "profit_cents_per_contract": round(100 * float(b.pnl_if_filled.mean()), 2),
        "premium_cents": round(100 * float(b.premium.mean()), 2),
        "residual_cents": round(100 * float(mean_r), 2),
        "residual_t": round(float(t_r), 2) if np.isfinite(t_r) else None,
        "share_of_profit_from_premium": round(float(b.premium.sum() / total), 2)
        if total
        else None,
    }


def main():
    o = load_orders()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "value_per_quote": study_value_per_quote(o),
        "inventory": study_inventory(o),
        "sizing": study_sizing(o),
        "attribution": study_attribution(),
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
