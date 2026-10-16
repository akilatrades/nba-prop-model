"""Single-leg models: a full range of outcomes for each player stat.

Each stat has two layers.

1. **Expected value.** A Poisson regression (log link) on pre-game features:
   expected volume (the player's recent share of team targets, carries or
   attempts times the team's recent volume), shrunk efficiency, the team's
   implied points and win probability. On the log scale the volume and
   efficiency coefficients say how much the model trusts each piece; 1.0
   would mean "volume x efficiency" taken literally.
2. **Spread around it.**
   - Yards and receptions: the empirical distribution of actual / expected
     in the training data, kept separately for ten bands of expected value.
     The chance of going over a line L is the share of training ratios above
     L / expected. This keeps the real shape of outcomes, including zero-yard
     games and long tails, without assuming a formula.
   - Touchdowns: a count distribution with the expected count. Anytime TDs
     fit a Poisson well. Passing TDs vary less than a Poisson allows, so they
     use a binomial whose number of "chances" is fitted by likelihood.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binom, poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_BANDS = 10
SEASON_HALF_LIFE = 3  # a season three years older counts half as much
SPREAD_SEASONS = 4  # outcome spread comes from the latest four seasons
BINOMIAL_CHANCES = [7, 8, 9, 10, 12, 15, 20, 30]


def _log(x, floor=0.1):
    return np.log(np.maximum(np.asarray(x, dtype=float), 0) + floor)


def _exp_targets(rows):
    return rows.ew_target_share * rows.ew_team_targets


def _exp_carries(rows):
    return rows.ew_carry_share * rows.ew_team_carries


def _redistribution(rows, column):
    """Vacated share from players in the same position group, and from others.

    Each enters as log(1 / (1 - share)); spreading the missing volume over
    everyone in proportion to their usage would give both a coefficient of 1.
    """
    same = rows[f"{column}_same"]
    other = rows[column] - same
    return np.column_stack([-np.log(1 - same), -np.log(1 - other)])


def _is(rows, group):
    return (rows.pos_group == group).astype(float)


def receiving_features(rows):
    return np.column_stack(
        [
            _log(_exp_targets(rows)),
            np.log(rows.ypt.clip(lower=0.5)),
            np.log(rows.implied_total),
            rows.p_win,
            _is(rows, "TE"),
            _is(rows, "RB"),
            _is(rows, "RB") * rows.p_win,
            np.log1p(rows.n_prior),
            _log(rows.ew_receiving_yards, 1.0),
            np.log(rows.def_pass_allowed),
            _redistribution(rows, "vacated_targets"),
        ]
    )


def reception_features(rows):
    return np.column_stack(
        [
            _log(_exp_targets(rows)),
            np.log(rows.catch_rate.clip(lower=0.2)),
            np.log(rows.implied_total),
            rows.p_win,
            _is(rows, "TE"),
            _is(rows, "RB"),
            _is(rows, "RB") * rows.p_win,
            _log(rows.ew_receptions, 0.5),
            np.log(rows.def_pass_allowed),
            _redistribution(rows, "vacated_targets"),
        ]
    )


def rushing_features(rows):
    return np.column_stack(
        [
            _log(_exp_carries(rows)),
            np.log(rows.ypc.clip(lower=1.0)),
            np.log(rows.implied_total),
            rows.p_win,
            _log(rows.ew_rushing_yards, 1.0),
            np.log(rows.def_rush_allowed),
            _redistribution(rows, "vacated_carries"),
        ]
    )


def passing_features(rows):
    return np.column_stack(
        [
            np.log(rows.ew_team_attempts),
            np.log(rows.ypa.clip(lower=3.0)),
            np.log(rows.implied_total),
            rows.p_win,
            _log(rows.ew_passing_yards, 10.0),
            np.log(rows.def_pass_allowed),
        ]
    )


def passing_td_features(rows):
    return np.column_stack(
        [
            np.log(rows.ew_team_attempts),
            np.log(rows.pass_td_rate.clip(lower=0.01)),
            np.log(rows.implied_total),
            rows.p_win,
            np.log(rows.def_pass_allowed),
        ]
    )


def skill_td_features(rows):
    return np.column_stack(
        [
            _log(_exp_targets(rows)),
            _log(_exp_carries(rows)),
            np.log(rows.td_per_touch.clip(lower=0.005)),
            np.log(rows.implied_total),
            _is(rows, "TE"),
            _is(rows, "RB"),
            _log(rows.ew_skill_tds, 0.05),
        ]
    )


def skill_rows(rows):
    return rows.pos_group.isin(["WR", "TE", "RB"])


def rb_rows(rows):
    return rows.pos_group == "RB"


def qb_rows(rows):
    return rows.is_starting_qb.astype(bool)


@dataclass
class StatSpec:
    name: str
    eligible: object
    features: object
    spread: str  # "empirical", "poisson" or "binomial"
    label: str
    lines: list = field(default_factory=list)


STATS = {
    "receiving_yards": StatSpec(
        "receiving_yards",
        skill_rows,
        receiving_features,
        "empirical",
        "Receiving yards (WR, TE, RB)",
        [19.5, 39.5, 59.5, 79.5, 99.5],
    ),
    "receptions": StatSpec(
        "receptions",
        skill_rows,
        reception_features,
        "empirical",
        "Receptions (WR, TE, RB)",
        [1.5, 3.5, 5.5, 7.5],
    ),
    "rushing_yards": StatSpec(
        "rushing_yards",
        rb_rows,
        rushing_features,
        "empirical",
        "Rushing yards (RB)",
        [29.5, 49.5, 69.5, 89.5],
    ),
    "passing_yards": StatSpec(
        "passing_yards",
        qb_rows,
        passing_features,
        "empirical",
        "Passing yards (starting QB)",
        [199.5, 249.5, 299.5],
    ),
    "passing_tds": StatSpec(
        "passing_tds",
        qb_rows,
        passing_td_features,
        "binomial",
        "Passing TDs (starting QB)",
        [0.5, 1.5, 2.5],
    ),
    "skill_tds": StatSpec(
        "skill_tds",
        skill_rows,
        skill_td_features,
        "poisson",
        "Anytime TD, rushing or receiving (WR, TE, RB)",
        [0.5],
    ),
}


# Column names for each feature matrix above, in order (for reporting).
FEATURE_NAMES = {
    "receiving_yards": [
        "log expected targets",
        "log yards per target",
        "log implied team points",
        "win probability",
        "is TE",
        "is RB",
        "is RB x win probability",
        "log games played",
        "log recent receiving yards",
        "log opponent pass yards allowed",
        "teammates out, same position: targets to redistribute",
        "teammates out, other positions: targets to redistribute",
    ],
    "receptions": [
        "log expected targets",
        "log catch rate",
        "log implied team points",
        "win probability",
        "is TE",
        "is RB",
        "is RB x win probability",
        "log recent receptions",
        "log opponent pass yards allowed",
        "teammates out, same position: targets to redistribute",
        "teammates out, other positions: targets to redistribute",
    ],
    "rushing_yards": [
        "log expected carries",
        "log yards per carry",
        "log implied team points",
        "win probability",
        "log recent rushing yards",
        "log opponent rush yards allowed",
        "teammates out, same position: carries to redistribute",
        "teammates out, other positions: carries to redistribute",
    ],
    "passing_yards": [
        "log team pass attempts",
        "log yards per attempt",
        "log implied team points",
        "win probability",
        "log recent passing yards",
        "log opponent pass yards allowed",
    ],
    "passing_tds": [
        "log team pass attempts",
        "log TD rate per attempt",
        "log implied team points",
        "win probability",
        "log opponent pass yards allowed",
    ],
    "skill_tds": [
        "log expected targets",
        "log expected carries",
        "log TDs per touch",
        "log implied team points",
        "is TE",
        "is RB",
        "log recent TDs",
    ],
}


class StatModel:
    """Expected value plus outcome distribution for one stat."""

    def __init__(self, spec):
        self.spec = spec
        self.glm = make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000)
        )

    def fit(self, rows):
        rows = rows[self.spec.eligible(rows) & rows[self.spec.name].notna()]
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
        if self.spec.spread == "binomial":
            mean = self.mean(rows)
            candidates = [n for n in BINOMIAL_CHANCES if n >= y.max()]
            fits = {
                n: binom.logpmf(y, n, np.clip(mean / n, 1e-9, 1 - 1e-9)).sum()
                for n in candidates
            }
            self.chances = max(fits, key=fits.get)
            return
        if self.spec.spread != "empirical":
            return
        recent = (rows.season > rows.season.max() - SPREAD_SEASONS).to_numpy()
        mean = self.mean(rows)[recent]
        ratio = y[recent] / mean
        self.edges = np.quantile(mean, np.linspace(0, 1, N_BANDS + 1)[1:-1])
        band = np.searchsorted(self.edges, mean)
        self.ratios = [np.sort(ratio[band == b]) for b in range(N_BANDS)]

    def mean(self, rows):
        return self.glm.predict(self.spec.features(rows))

    def _bands(self, mean):
        return np.searchsorted(self.edges, mean)

    def prob_over(self, rows, line, mean=None):
        """P(stat > line) for each row; ``line`` is a number or an array."""
        mean = self.mean(rows) if mean is None else mean
        line = np.broadcast_to(np.asarray(line, dtype=float), mean.shape)
        if self.spec.spread == "poisson":
            return poisson.sf(np.floor(line), mean)
        if self.spec.spread == "binomial":
            n = self.chances
            return binom.sf(np.floor(line), n, np.clip(mean / n, 0, 1))
        bands = self._bands(mean)
        threshold = line / mean
        out = np.empty(len(mean))
        for b in range(N_BANDS):
            here = bands == b
            if here.any():
                ratios = self.ratios[b]
                above = np.searchsorted(ratios, threshold[here], side="right")
                out[here] = 1 - above / len(ratios)
        return out

    def quantiles(self, rows, qs, mean=None):
        mean = self.mean(rows) if mean is None else mean
        if self.spec.spread == "poisson":
            return np.column_stack([poisson.ppf(q, mean) for q in qs])
        if self.spec.spread == "binomial":
            p = np.clip(mean / self.chances, 0, 1)
            return np.column_stack([binom.ppf(q, self.chances, p) for q in qs])
        bands = self._bands(mean)
        out = np.empty((len(mean), len(qs)))
        for b in range(N_BANDS):
            here = bands == b
            if here.any():
                out[here] = mean[here, None] * np.quantile(self.ratios[b], qs)[None, :]
        return out

    def coefficients(self):
        scaler, glm = self.glm.named_steps.values()
        return glm.coef_ / scaler.scale_


class NaiveModel(StatModel):
    """Baseline: the player's own recent average, with the same spread layer."""

    def fit(self, rows):
        rows = rows[self.spec.eligible(rows) & rows[self.spec.name].notna()]
        self._fit_spread(rows, rows[self.spec.name].to_numpy(float))
        self.n_train = len(rows)
        return self

    def mean(self, rows):
        return np.maximum(rows[f"ew_{self.spec.name}"].to_numpy(float), 0.05)


def fit_all(rows, model=StatModel):
    return {name: model(spec).fit(rows) for name, spec in STATS.items()}
