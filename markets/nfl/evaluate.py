"""Walk-forward backtest: predict each block of weeks using only earlier games.

The model is refit before every block of four weeks (weeks 1-4, 5-8, 9-12
and 13-18) on all games played before that block, mirroring how it would
be refreshed during a season. Nothing from the predicted weeks is used.
"""

import numpy as np
import pandas as pd

from markets.nfl.marginals import STATS, NaiveModel, StatModel

BLOCKS = [(1, 4), (5, 8), (9, 12), (13, 18)]
QUANTILES = [0.1, 0.5, 0.9]


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def walk_forward(rows, seasons):
    """One output row per (player-game, stat, line) with both models' prices."""
    records = []
    for season in seasons:
        for first, last in BLOCKS:
            test = rows[(rows.season == season) & rows.week.between(first, last)]
            if test.empty:
                continue
            train = rows[
                (rows.season < season) | ((rows.season == season) & (rows.week < first))
            ]
            for name, spec in STATS.items():
                model = StatModel(spec).fit(train)
                naive = NaiveModel(spec).fit(train)
                target = test[spec.eligible(test) & test[name].notna()]
                y = target[name].to_numpy(float)
                mean, naive_mean = model.mean(target), naive.mean(target)
                at_or_below = y[:, None] <= model.quantiles(target, QUANTILES, mean)
                for line in spec.lines:
                    records.append(
                        pd.DataFrame(
                            {
                                "season": season,
                                "week": target.week.to_numpy(),
                                "player_id": target.player_id.to_numpy(),
                                "stat": name,
                                "line": line,
                                "actual": y,
                                "over": (y > line).astype(float),
                                "mean": mean,
                                "naive_mean": naive_mean,
                                "p_model": model.prob_over(target, line, mean),
                                "p_naive": naive.prob_over(target, line, naive_mean),
                                "at_or_below_q10": at_or_below[:, 0],
                                "at_or_below_q50": at_or_below[:, 1],
                                "at_or_below_q90": at_or_below[:, 2],
                            }
                        )
                    )
    return pd.concat(records, ignore_index=True)


def summarize(results):
    """Accuracy and calibration by stat, comparing the model with the baseline."""
    rows = []
    for name, group in results.groupby("stat", sort=False):
        first_line = group[group.line == group.line.min()]
        brier = ((group.p_model - group.over) ** 2).mean()
        brier_naive = ((group.p_naive - group.over) ** 2).mean()
        loss = log_loss(group.over, group.p_model).mean()
        loss_naive = log_loss(group.over, group.p_naive).mean()
        rows.append(
            {
                "stat": name,
                "label": STATS[name].label,
                "player_games": len(first_line),
                "mae": (first_line.actual - first_line["mean"]).abs().mean(),
                "mae_naive": (first_line.actual - first_line.naive_mean).abs().mean(),
                "brier": brier,
                "brier_naive": brier_naive,
                "brier_improvement_pct": 100 * (1 - brier / brier_naive),
                "log_loss": loss,
                "log_loss_naive": loss_naive,
                "log_loss_improvement_pct": 100 * (1 - loss / loss_naive),
                "at_or_below_q10": first_line.at_or_below_q10.mean(),
                "at_or_below_q50": first_line.at_or_below_q50.mean(),
                "at_or_below_q90": first_line.at_or_below_q90.mean(),
            }
        )
    return pd.DataFrame(rows)


def by_line(results):
    return (
        results.groupby(["stat", "line"], sort=False)
        .agg(
            n=("over", "size"),
            actual_rate=("over", "mean"),
            model_rate=("p_model", "mean"),
            brier=(
                "p_model",
                lambda p: ((p - results.loc[p.index, "over"]) ** 2).mean(),
            ),
            brier_naive=(
                "p_naive",
                lambda p: ((p - results.loc[p.index, "over"]) ** 2).mean(),
            ),
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


def injury_effect(with_injuries, without_injuries, rows, thresholds=(0.1, 0.2)):
    """How the injury adjustment changed predictions when teammates were out.

    Compares two backtests that differ only in whether the injury report is
    used, on player-games where at least ``threshold`` of the team's targets
    (carries, for rushing) belonged to teammates ruled out that week.
    """
    key = ["season", "week", "player_id", "stat", "line"]
    merged = with_injuries.merge(
        without_injuries[key + ["p_model", "mean"]], on=key, suffixes=("", "_without")
    ).merge(
        rows[["season", "week", "player_id", "vacated_targets", "vacated_carries"]],
        on=["season", "week", "player_id"],
    )
    merged["vacated"] = merged.vacated_targets.where(
        merged.stat != "rushing_yards", merged.vacated_carries
    )
    out = []
    for stat in ["receiving_yards", "receptions", "rushing_yards"]:
        for threshold in thresholds:
            group = merged[(merged.stat == stat) & (merged.vacated >= threshold)]
            first = group[group.line == group.line.min()]
            brier_new = ((group.p_model - group.over) ** 2).mean()
            brier_old = ((group.p_model_without - group.over) ** 2).mean()
            out.append(
                {
                    "stat": stat,
                    "vacated_at_least": threshold,
                    "player_games": len(first),
                    "actual_average": first.actual.mean(),
                    "expected_without_injuries": first.mean_without.mean(),
                    "expected_with_injuries": first["mean"].mean(),
                    "brier_without": brier_old,
                    "brier_with": brier_new,
                    "brier_improvement_pct": 100 * (1 - brier_new / brier_old),
                    "mae_without": (first.actual - first.mean_without).abs().mean(),
                    "mae_with": (first.actual - first["mean"]).abs().mean(),
                }
            )
    return pd.DataFrame(out)
