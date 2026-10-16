"""Recompute the market test's headline numbers with separate code.

Reads outputs/market_test/priced_markets.csv.gz (one row per priced market)
and recomputes Test B's P&L and t, and Test A's coefficients, without
importing markets.nba.market.

Run from the repository root: python -m research.verify_market_test
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

frame = pd.read_csv("outputs/market_test/priced_markets.csv.gz", low_memory=False)
reported = json.loads(Path("outputs/market_test/results.json").read_text())["at_tip"]


def fee(price):
    return 0.07 * price * (1 - price)


rows = []
for r in frame.itertuples():
    yes_cost, no_cost = r.yes_ask, 1 - r.yes_bid
    if r.p_model - yes_cost - fee(yes_cost) >= 0.05:
        rows.append((r.game_id, r.tip_utc, r.over - yes_cost - fee(yes_cost)))
    elif (1 - r.p_model) - no_cost - fee(no_cost) >= 0.05:
        rows.append((r.game_id, r.tip_utc, (1 - r.over) - no_cost - fee(no_cost)))
t = pd.DataFrame(rows, columns=["game_id", "tip_utc", "pnl"])
n = len(t)
mean = t.pnl.mean()
by_game = (t.pnl - mean).groupby(t.game_id).sum()
g = len(by_game)
se = np.sqrt((by_game**2).sum() * g / (g - 1)) / n
print(f"Test B: {n} trades, {mean:+.4f} per contract, t {mean / se:.2f}")
print(
    f"  reported: {reported['test_b']['all']['trades']} trades, "
    f"{reported['test_b']['all']['pnl_per_contract']:+.4f}, "
    f"t {reported['test_b']['all']['t']:.2f}"
)


def clip_logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


X = np.column_stack(
    [np.ones(len(frame)), clip_logit(frame.mid), clip_logit(frame.p_model)]
)
y = frame.over.to_numpy()


def nll(beta):
    z = X @ beta
    return np.sum(np.logaddexp(0, z) - y * z)


beta = minimize(nll, np.zeros(3), method="BFGS").x
print(f"Test A coefficients (market, model): {beta[1]:.3f}, {beta[2]:.3f}")
print(
    f"  reported: {reported['test_a']['coef_market']:.3f}, "
    f"{reported['test_a']['coef_model']:.3f}"
)
