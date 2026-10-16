"""Statistics shared by every market test: clustered t, clustered logistic regression."""

import numpy as np
import pandas as pd


def clustered_mean_t(values, clusters):
    """Mean and its t statistic with standard errors clustered by ``clusters``."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    mean = values.mean()
    sums = pd.Series(values - mean).groupby(np.asarray(clusters)).sum()
    g = len(sums)
    se = np.sqrt((sums**2).sum() * g / (g - 1)) / n
    return mean, mean / se if se > 0 else np.nan


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    return np.log(p / (1 - p))


def clustered_logit(y, x, clusters, iterations=50):
    """Logistic regression (with intercept) and game-clustered standard errors."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)])
    beta = np.zeros(X.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-X @ beta))
        w = p * (1 - p)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    bread = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    scores = pd.DataFrame(X * (y - p)[:, None]).groupby(np.asarray(clusters)).sum()
    meat = scores.to_numpy().T @ scores.to_numpy()
    g = len(scores)
    cov = bread @ meat @ bread * g / (g - 1)
    se = np.sqrt(np.diag(cov))
    return beta, se


def test_a(frame):
    beta, se = clustered_logit(
        frame.over,
        np.column_stack([logit(frame.mid), logit(frame.p_model)]),
        frame.game_id,
    )
    return {
        "markets": len(frame),
        "games": frame.game_id.nunique(),
        "coef_market": beta[1],
        "t_market": beta[1] / se[1],
        "coef_model": beta[2],
        "t_model": beta[2] / se[2],
        "passes": bool(beta[2] / se[2] >= 2.0),
    }


def test_b_summary(t, column="pnl"):
    mean, t_stat = clustered_mean_t(t[column], t.game_id)
    return {
        "trades": len(t),
        "games": t.game_id.nunique(),
        "pnl_per_contract": mean,
        "t": t_stat,
        "total": t[column].sum(),
        "avg_cost": t.cost.mean(),
        "share_yes": (t.side == "yes").mean(),
    }
