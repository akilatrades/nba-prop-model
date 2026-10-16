"""Single-leg models: a full range of outcomes for each player stat.

Each stat has two layers, as in the NFL model.

1. **Expected value.** A Poisson regression (log link) on pre-game features:
   expected minutes (recent minutes, starts, games missed, teammates out or
   back, back-to-backs, blowout risk), the player's per-minute rate shrunk
   toward his position, his recent form, the team's implied points from the
   betting line, and what the opponent allows. On the log scale a
   coefficient of 1.0 on minutes and on the rate would mean
   "minutes x rate" taken literally.
2. **Spread around it.** The empirical distribution of actual / expected in
   the training data, kept separately for ten bands of expected value within
   three tiers of recent minutes. The
   chance of reaching a line L is the share of training ratios above
   L / expected. This keeps the real shape of outcomes (including zero-three
   nights and big scoring games) without assuming a formula.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_BANDS = 10
SEASON_HALF_LIFE = 3  # a season three years older counts half as much
SPREAD_SEASONS = 3  # outcome spread comes from the latest three seasons
# The spread is kept separately for bench, rotation and heavy-minute players:
# the same expected value is less certain for a player who plays less.
MINUTE_TIERS = [20, 30]


def _tier(rows):
    return np.searchsorted(MINUTE_TIERS, rows.ew_minutes.to_numpy(float), side="right")


def _log(x, floor=0.1):
    return np.log(np.maximum(np.asarray(x, dtype=float), 0) + floor)


def _out(share):
    """log(1 / (1 - share)): 1.0 means volume handed out in proportion to usage."""
    return -np.log(1 - np.clip(np.asarray(share, dtype=float), 0, 0.95))


def _is(rows, position):
    return (rows.position == position).astype(float)


def minutes_block(rows):
    starter = rows.ew_starter.to_numpy(float)
    blowout = np.abs(rows.team_spread.to_numpy(float))
    return [
        _log(rows.ew_minutes, 1.0),
        _log(rows.last_minutes, 1.0),
        starter,
        np.log1p(rows.games_missed.fillna(0)),
        _out(rows.vacated_same_share_minutes),
        _out(rows.vacated_share_minutes - rows.vacated_same_share_minutes),
        _out(rows.returning_share_minutes),
        (rows.rest_days <= 1).astype(float),
        rows.playoffs.to_numpy(float),
        rows.home.astype(float),
        blowout,
        blowout * starter,
        np.log1p(rows.n_prior),
    ]


def context_block(rows, stat):
    return [
        np.log(rows.implied_points),
        np.log(rows[f"opp_factor_{stat}"]),
        _out(rows[f"vacated_same_share_{stat}"]),
        _out(rows[f"vacated_share_{stat}"] - rows[f"vacated_same_share_{stat}"]),
        _out(rows[f"returning_share_{stat}"]),
    ]


def points_features(rows):
    return np.column_stack(
        minutes_block(rows)
        + context_block(rows, "points")
        + [
            np.log(rows.rate_points),
            np.log(rows.rate_fta),
            _log(rows.ew_points, 1.0),
        ]
    )


def rebounds_features(rows):
    return np.column_stack(
        minutes_block(rows)
        + context_block(rows, "rebounds")
        + [
            np.log(rows.rate_rebounds),
            _log(rows.ew_rebounds, 0.5),
            _is(rows, "C"),
            _is(rows, "F"),
        ]
    )


def assists_features(rows):
    return np.column_stack(
        minutes_block(rows)
        + context_block(rows, "assists")
        + [
            np.log(rows.rate_assists),
            _log(rows.ew_assists, 0.5),
            _is(rows, "G"),
        ]
    )


def threes_features(rows):
    return np.column_stack(
        minutes_block(rows)
        + context_block(rows, "threes")
        + [
            np.log(rows.rate_fg3a),
            np.log(rows.three_pct),
            _log(rows.ew_threes, 0.25),
        ]
    )


MINUTES_NAMES = [
    "log recent minutes",
    "log minutes last game",
    "recent share of starts",
    "log(1 + team games missed)",
    "teammates out, same position: minutes to hand out",
    "teammates out, other positions: minutes to hand out",
    "regulars returning: their minutes",
    "back-to-back",
    "playoffs",
    "home",
    "size of the spread",
    "size of the spread x starter",
    "log(1 + games played)",
]


def context_names(stat):
    return [
        "log implied team points",
        f"log opponent {stat} allowed vs league",
        f"teammates out, same position: {stat} to hand out",
        f"teammates out, other positions: {stat} to hand out",
        f"regulars returning: their {stat}",
    ]


@dataclass
class StatSpec:
    name: str
    features: object
    label: str
    lines: list
    feature_names: list = field(default_factory=list)


STATS = {
    "points": StatSpec(
        "points",
        points_features,
        "Points",
        [9.5, 14.5, 19.5, 24.5, 29.5, 34.5],
        MINUTES_NAMES
        + context_names("points")
        + ["log points per minute", "log free throws per minute", "log recent points"],
    ),
    "rebounds": StatSpec(
        "rebounds",
        rebounds_features,
        "Rebounds",
        [3.5, 5.5, 7.5, 9.5, 11.5],
        MINUTES_NAMES
        + context_names("rebounds")
        + ["log rebounds per minute", "log recent rebounds", "is C", "is F"],
    ),
    "assists": StatSpec(
        "assists",
        assists_features,
        "Assists",
        [1.5, 3.5, 5.5, 7.5, 9.5],
        MINUTES_NAMES
        + context_names("assists")
        + ["log assists per minute", "log recent assists", "is G"],
    ),
    "threes": StatSpec(
        "threes",
        threes_features,
        "Threes made",
        [0.5, 1.5, 2.5, 3.5, 4.5],
        MINUTES_NAMES
        + context_names("threes")
        + ["log three attempts per minute", "log three-point %", "log recent threes"],
    ),
}


class StatModel:
    """Expected value plus outcome distribution for one stat."""

    def __init__(self, spec):
        self.spec = spec
        self.glm = make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000)
        )

    def fit(self, rows):
        rows = rows[rows[self.spec.name].notna()]
        y = rows[self.spec.name].to_numpy(float)
        age = rows.season.max() - rows.season.to_numpy()
        weight = 0.5 ** (age / SEASON_HALF_LIFE)
        self.glm.fit(
            self.spec.features(rows),
            np.maximum(y, 0),
            poissonregressor__sample_weight=weight,
        )
        self._fit_spread(rows, y)
        self.n_train = len(rows)
        return self

    def _fit_spread(self, rows, y):
        recent = (rows.season > rows.season.max() - SPREAD_SEASONS).to_numpy()
        mean = self.mean(rows)[recent]
        tier = _tier(rows)[recent]
        ratio = y[recent] / mean
        self.edges, self.ratios = [], []
        for t in range(len(MINUTE_TIERS) + 1):
            here = tier == t
            edges = np.quantile(mean[here], np.linspace(0, 1, N_BANDS + 1)[1:-1])
            band = np.searchsorted(edges, mean[here])
            self.edges.append(edges)
            self.ratios.append(
                [np.sort(ratio[here][band == b]) for b in range(N_BANDS)]
            )

    def mean(self, rows):
        return self.glm.predict(self.spec.features(rows))

    def _cells(self, rows, mean):
        """(minutes tier, band of expected value) for each row."""
        tier = _tier(rows)
        band = np.empty(len(mean), dtype=int)
        for t in range(len(MINUTE_TIERS) + 1):
            here = tier == t
            band[here] = np.searchsorted(self.edges[t], mean[here])
        return tier, band

    def prob_over(self, rows, line, mean=None):
        """P(stat > line) for each row; ``line`` is a number or an array."""
        mean = self.mean(rows) if mean is None else mean
        line = np.broadcast_to(np.asarray(line, dtype=float), mean.shape)
        tier, band = self._cells(rows, mean)
        threshold = line / mean
        out = np.empty(len(mean))
        for t in range(len(MINUTE_TIERS) + 1):
            for b in range(N_BANDS):
                here = (tier == t) & (band == b)
                if here.any():
                    ratios = self.ratios[t][b]
                    above = np.searchsorted(ratios, threshold[here], side="right")
                    out[here] = 1 - above / len(ratios)
        return out

    def quantiles(self, rows, qs, mean=None):
        mean = self.mean(rows) if mean is None else mean
        tier, band = self._cells(rows, mean)
        out = np.empty((len(mean), len(qs)))
        for t in range(len(MINUTE_TIERS) + 1):
            for b in range(N_BANDS):
                here = (tier == t) & (band == b)
                if here.any():
                    q = np.quantile(self.ratios[t][b], qs)
                    out[here] = mean[here, None] * q[None, :]
        return out

    def coefficients(self):
        scaler, glm = self.glm.named_steps.values()
        return glm.coef_ / scaler.scale_


class NaiveModel(StatModel):
    """Baseline: the player's own recent average, with the same spread layer."""

    def fit(self, rows):
        rows = rows[rows[self.spec.name].notna()]
        self._fit_spread(rows, rows[self.spec.name].to_numpy(float))
        self.n_train = len(rows)
        return self

    def mean(self, rows):
        return np.maximum(rows[f"ew_{self.spec.name}"].to_numpy(float), 0.05)


def fit_all(rows, model=StatModel):
    return {name: model(spec).fit(rows) for name, spec in STATS.items()}
