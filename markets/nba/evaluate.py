"""Walk-forward backtest: predict each month using only earlier games.

Before each calendar month the model is refit on every earlier game with a
betting line (2020-21 onward) and predicts that month, mirroring how it would
be refreshed during a season. Nothing from the predicted month is used.
"""

import numpy as np
import pandas as pd

from markets.nba.marginals import STATS, NaiveModel, StatModel

FIRST_TRAIN_SEASON = 2021
QUANTILES = [0.1, 0.5, 0.9]


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def month_of(rows):
    local = rows.tip_utc.dt.tz_convert("America/New_York")
    return local.dt.tz_localize(None).dt.to_period("M")


def monthly_models(rows, seasons, transform=None):
    """Yield (test rows, {stat: (model, baseline)}) for each month in turn.

    Each month's models are fit on every eligible game before that month.
    ``transform`` optionally changes the rows before fitting and predicting
    (used to switch features off for comparison).
    """
    rows = rows[rows.season >= FIRST_TRAIN_SEASON].copy()
    rows["month"] = month_of(rows)
    for season in seasons:
        months = sorted(rows.loc[rows.season == season, "month"].unique())
        for month in months:
            test = rows[(rows.season == season) & (rows.month == month)]
            train = rows[rows.month < month]
            if transform is not None:
                train, test = transform(train), transform(test)
            models = {
                name: (StatModel(spec).fit(train), NaiveModel(spec).fit(train))
                for name, spec in STATS.items()
            }
            yield test, models


def walk_forward(rows, seasons, transform=None):
    """One output row per (player-game, stat, line) with both models' prices."""
    records = []
    for test, models in monthly_models(rows, seasons, transform):
        for name, (model, naive) in models.items():
            y = test[name].to_numpy(float)
            mean, naive_mean = model.mean(test), naive.mean(test)
            below = y[:, None] <= model.quantiles(test, QUANTILES, mean)
            for line in STATS[name].lines:
                records.append(
                    pd.DataFrame(
                        {
                            "season": test.season.to_numpy(),
                            "month": test.month.astype(str).to_numpy(),
                            "game_id": test.game_id.to_numpy(),
                            "player_id": test.player_id.to_numpy(),
                            "ew_minutes": test.ew_minutes.to_numpy(),
                            "stat": name,
                            "line": line,
                            "actual": y,
                            "over": (y > line).astype(float),
                            "mean": mean,
                            "naive_mean": naive_mean,
                            "p_model": model.prob_over(test, line, mean),
                            "p_naive": naive.prob_over(test, line, naive_mean),
                            "at_or_below_q10": below[:, 0],
                            "at_or_below_q50": below[:, 1],
                            "at_or_below_q90": below[:, 2],
                        }
                    )
                )
    return pd.concat(records, ignore_index=True)


def summarize(results):
    """Accuracy and calibration by stat, comparing the model with the baseline."""
    out = []
    for name, group in results.groupby("stat", sort=False):
        first = group[group.line == group.line.min()]
        brier = ((group.p_model - group.over) ** 2).mean()
        brier_naive = ((group.p_naive - group.over) ** 2).mean()
        loss = log_loss(group.over, group.p_model).mean()
        loss_naive = log_loss(group.over, group.p_naive).mean()
        out.append(
            {
                "stat": name,
                "label": STATS[name].label,
                "player_games": len(first),
                "mae": (first.actual - first["mean"]).abs().mean(),
                "mae_naive": (first.actual - first.naive_mean).abs().mean(),
                "brier": brier,
                "brier_naive": brier_naive,
                "brier_improvement_pct": 100 * (1 - brier / brier_naive),
                "log_loss": loss,
                "log_loss_naive": loss_naive,
                "log_loss_improvement_pct": 100 * (1 - loss / loss_naive),
                "at_or_below_q10": first.at_or_below_q10.mean(),
                "at_or_below_q50": first.at_or_below_q50.mean(),
                "at_or_below_q90": first.at_or_below_q90.mean(),
            }
        )
    return pd.DataFrame(out)


def by_line(results):
    def brier(column):
        return lambda p: ((p - results.loc[p.index, "over"]) ** 2).mean()

    return (
        results.groupby(["stat", "line"], sort=False)
        .agg(
            n=("over", "size"),
            actual_rate=("over", "mean"),
            model_rate=("p_model", "mean"),
            brier=("p_model", brier("p_model")),
            brier_naive=("p_naive", brier("p_naive")),
        )
        .reset_index()
    )


def calibration(results, bins=10):
    """Predicted chance vs how often it happened, pooled across stats and lines."""
    edges = np.linspace(0, 1, bins + 1)
    band = np.clip(np.digitize(results.p_model, edges) - 1, 0, bins - 1)
    table = (
        results.assign(band=band)
        .groupby("band")
        .agg(n=("over", "size"), predicted=("p_model", "mean"), actual=("over", "mean"))
    )
    table["range"] = [f"{edges[b]:.1f}-{edges[b + 1]:.1f}" for b in table.index]
    return table.reset_index(drop=True)[["range", "n", "predicted", "actual"]]


def by_season(results):
    """Average predicted minus actual rate: positive means the model ran high."""
    return (
        results.groupby(["stat", "season"], sort=False)
        .apply(lambda g: g.p_model.mean() - g.over.mean(), include_groups=False)
        .unstack("season")
    )
