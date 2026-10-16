import numpy as np
import pandas as pd
import pytest

from markets.nfl.marginals import STATS, NaiveModel, StatModel


def synthetic_qbs(n=20000, seed=0):
    """QB games where passing yards depend on implied points, plus noise."""
    rng = np.random.default_rng(seed)
    implied = rng.uniform(16, 30, n)
    mean = 240 * (implied / 22) ** 0.8
    yards = rng.gamma(shape=12, scale=mean / 12)
    tds = rng.binomial(8, np.clip(mean / 240 * 1.6 / 8, 0, 1))
    return pd.DataFrame(
        {
            # Recent seasons only, so the whole sample feeds the spread layer.
            "season": rng.integers(2021, 2025, n),
            "is_starting_qb": True,
            "pos_group": "QB",
            "implied_total": implied,
            "p_win": rng.uniform(0.2, 0.8, n),
            "ew_team_attempts": rng.uniform(30, 38, n),
            "ypa": rng.uniform(6, 8, n),
            "pass_td_rate": rng.uniform(0.03, 0.06, n),
            "ew_passing_yards": rng.uniform(180, 300, n),
            "def_pass_allowed": rng.uniform(200, 260, n),
            "passing_yards": yards,
            "passing_tds": tds,
        }
    )


def test_probabilities_are_calibrated_on_new_data():
    train, test = synthetic_qbs(seed=1), synthetic_qbs(seed=2)
    model = StatModel(STATS["passing_yards"]).fit(train)
    for line in [199.5, 249.5, 299.5]:
        p = model.prob_over(test, line)
        assert p.mean() == pytest.approx((test.passing_yards > line).mean(), abs=0.02)


def test_prices_fall_as_the_line_rises_and_quantiles_rise():
    model = StatModel(STATS["passing_yards"]).fit(synthetic_qbs())
    rows = synthetic_qbs(n=50, seed=3)
    p = np.column_stack([model.prob_over(rows, line) for line in [150, 250, 350]])
    assert (np.diff(p, axis=1) <= 0).all()
    q = model.quantiles(rows, [0.1, 0.5, 0.9])
    assert (np.diff(q, axis=1) >= 0).all()


def test_binomial_touchdowns_pick_enough_chances():
    train = synthetic_qbs()
    model = StatModel(STATS["passing_tds"]).fit(train)
    assert model.chances >= train.passing_tds.max()
    p1, p2 = model.prob_over(train, 0.5), model.prob_over(train, 1.5)
    assert p1.mean() == pytest.approx((train.passing_tds >= 1).mean(), abs=0.02)
    assert (p2 <= p1).all()


def test_naive_baseline_uses_recent_average():
    rows = synthetic_qbs(n=200)
    naive = NaiveModel(STATS["passing_yards"]).fit(rows)
    assert naive.mean(rows) == pytest.approx(rows.ew_passing_yards.to_numpy())
