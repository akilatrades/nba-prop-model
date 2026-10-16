"""Parlay test (RESEARCH_LOG.md, 2026-10-15): run once as written there.

Writes outputs/parlays/results.json and the parlay lists.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi.stats import clustered_mean_t

OUT = Path("outputs/parlays")
PRICED = Path("outputs/market_test/priced_markets.csv.gz")
MIN_Q = 0.70
MIN_EDGE = 0.03
SPLIT = pd.Timestamp("2026-02-01").date()
STAKE = 0.02


def fee(price):
    price = np.asarray(price, dtype=float)
    return 0.07 * price * (1 - price)


def legs():
    p = pd.read_csv(PRICED, low_memory=False)
    p = p[p.over.isin([0, 1]) & p.p_model.notna()]
    p = p[(p.yes_bid > 0) & (p.yes_ask < 1) & (p.yes_ask > p.yes_bid)].copy()
    yes = p.p_model >= 0.5
    p["side"] = np.where(yes, "yes", "no")
    p["q"] = np.where(yes, p.p_model, 1 - p.p_model)
    p["cost"] = np.where(yes, p.yes_ask, 1 - p.yes_bid)
    p["edge"] = p.q - p.cost - fee(p.cost)
    p["hit"] = np.where(yes, p.over, 1 - p.over).astype(int)
    p["mid_side"] = np.where(
        yes, (p.yes_bid + p.yes_ask) / 2, 1 - (p.yes_bid + p.yes_ask) / 2
    )
    p["date"] = (
        pd.to_datetime(p.tip_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    )
    return p


def qualifying(p):
    q = p[(p.q >= MIN_Q) & (p.edge >= MIN_EDGE)]
    q = q.sort_values(["q", "edge"], ascending=False)
    return q.drop_duplicates(["game_id", "player_id"]).copy()


def price_parlay(group):
    price = float(np.prod(group.cost))
    paid = price + float(fee(price))
    payout = float(group.hit.prod())
    return {
        "date": group.date.iloc[0],
        "legs": len(group),
        "tickers": "|".join(group.ticker),
        "price": price,
        "paid": paid,
        "q_joint": float(np.prod(group.q)),
        "mid_joint": float(np.prod(group.mid_side)),
        "hit": payout,
        "ret": payout / paid - 1,
        "regular": bool((group.season_type == "regular").all()),
    }


def nightly(q, n_legs):
    rows = []
    for _, day in q.groupby("date"):
        best = day.sort_values(["q", "edge"], ascending=False).drop_duplicates(
            "game_id"
        )
        if len(best) >= n_legs:
            rows.append(price_parlay(best.head(n_legs)))
    return pd.DataFrame(rows)


def all_combos(q, n_legs, cap=200_000):
    rows = []
    for _, day in q.groupby("date"):
        idx = list(day.index)
        for combo in itertools.combinations(idx, n_legs):
            g = day.loc[list(combo)]
            if g.game_id.nunique() == n_legs:
                rows.append(price_parlay(g))
                if len(rows) >= cap:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def summarize(t):
    if len(t) < 2:
        return {"parlays": len(t)}
    mean, tstat = clustered_mean_t(t.ret, t.date)
    first, second = t[t.date < SPLIT], t[t.date >= SPLIT]
    reg = t[t.regular]
    out = {
        "parlays": len(t),
        "dates": int(t.date.nunique()),
        "hit_rate": float(t.hit.mean()),
        "model_joint": float(t.q_joint.mean()),
        "price_joint": float(t.price.mean()),
        "return_per_dollar": float(mean),
        "t": float(tstat),
        "first_half": float(first.ret.mean()) if len(first) else None,
        "second_half": float(second.ret.mean()) if len(second) else None,
        "regular_season": float(reg.ret.mean()) if len(reg) else None,
    }
    out["passes"] = bool(
        mean > 0
        and tstat >= 2
        and (out["first_half"] or 0) > 0
        and (out["second_half"] or 0) > 0
        and (out["regular_season"] or 0) > 0
    )
    return out


def singles(q, parlays):
    """The same legs as single bets, one dollar each."""
    used = set(itertools.chain.from_iterable(t.split("|") for t in parlays.tickers))
    s = q[q.ticker.isin(used)].copy()
    s["ret"] = s.hit / (s.cost + fee(s.cost)) - 1
    mean, tstat = clustered_mean_t(s.ret, s.date)
    return {
        "legs": len(s),
        "hit_rate": float(s.hit.mean()),
        "model_q": float(s.q.mean()),
        "avg_cost": float(s.cost.mean()),
        "return_per_dollar": float(mean),
        "t": float(tstat),
    }


def same_game_pairs(p):
    """Correlation within a game: both legs hitting against the product of probabilities."""
    rows = []
    cols = ["game_id", "player_id", "team", "stat", "q", "hit", "mid_side"]
    one = p.sort_values("q", ascending=False).drop_duplicates(
        ["game_id", "player_id", "stat"]
    )
    for _, g in one[cols].groupby("game_id"):
        r = g.to_dict("records")
        for a, b in itertools.combinations(r, 2):
            kind = (
                "same player"
                if a["player_id"] == b["player_id"]
                else ("teammates" if a["team"] == b["team"] else "opponents")
            )
            rows.append(
                {
                    "kind": kind,
                    "both": a["hit"] * b["hit"],
                    "q_product": a["q"] * b["q"],
                    "mid_product": a["mid_side"] * b["mid_side"],
                    "resid": (a["hit"] - a["q"]) * (b["hit"] - b["q"]),
                    "game_id": a["game_id"],
                }
            )
    t = pd.DataFrame(rows)
    out = {}
    for kind, g in t.groupby("kind"):
        mean, tstat = clustered_mean_t(g.resid, g.game_id)
        out[kind] = {
            "pairs": len(g),
            "both_hit": float(g.both.mean()),
            "model_product": float(g.q_product.mean()),
            "price_product": float(g.mid_product.mean()),
            "residual_covariance": float(mean),
            "t": float(tstat),
        }
    return out


def bootstrap(t, paths=2000, seed=0):
    rng = np.random.default_rng(seed)
    ret = t.ret.to_numpy()
    paid = t.paid.to_numpy()
    mean = ret.mean()
    out = {}
    for label, shift in [
        ("full_edge", 0.0),
        ("half_edge", -mean / 2),
        ("no_edge", -mean),
    ]:
        ends = []
        for _ in range(paths):
            bank = 500.0
            for i in rng.integers(0, len(ret), len(ret)):
                n = np.floor(STAKE * bank / paid[i])
                bank += n * paid[i] * (ret[i] + shift)
                if bank <= 1:
                    break
            ends.append(bank)
        ends = np.array(ends)
        out[label] = {
            "median_end": float(np.median(ends)),
            "p10_end": float(np.percentile(ends, 10)),
            "p90_end": float(np.percentile(ends, 90)),
            "share_losing": float((ends < 500).mean()),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = legs()
    q = qualifying(p)
    results = {
        "legs_priced": len(p),
        "legs_qualifying": len(q),
        "qualifying_singles": {
            "hit_rate": float(q.hit.mean()),
            "model_q": float(q.q.mean()),
            "avg_cost": float(q.cost.mean()),
            "return_per_dollar": float((q.hit / (q.cost + fee(q.cost)) - 1).mean()),
            "t": float(
                clustered_mean_t(q.hit / (q.cost + fee(q.cost)) - 1, q.game_id)[1]
            ),
        },
    }
    for n in (2, 3):
        t = nightly(q, n)
        results[f"nightly_{n}_leg"] = summarize(t) | {
            "same_legs_as_singles": singles(q, t)
        }
        if results[f"nightly_{n}_leg"].get("passes"):
            results[f"nightly_{n}_leg"]["bankroll"] = bootstrap(t)
        t.to_csv(OUT / f"nightly_{n}_leg.csv", index=False)
        combos = all_combos(q, n)
        results[f"all_{n}_leg_combinations"] = summarize(combos)
    results["same_game_pairs"] = same_game_pairs(p)
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
