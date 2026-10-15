"""Joint probabilities for double-doubles and triple-doubles.

Points, rebounds and assists of 10 or more are not independent: a player who
gets more minutes than expected tends to gain on all three. Each pair is
linked by a Gaussian copula: the events are thresholds on correlated normal
variables, so P(both) = Phi2(Phi^-1(pA), Phi^-1(pB); rho). The three
correlations are fitted by maximum likelihood on past model probabilities
and results.

Double-double here means at least two of points, rebounds and assists of 10
or more (steals and blocks rarely reach 10 and are ignored).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import owens_t
from scipy.stats import multivariate_normal, norm

PAIRS = [("points", "rebounds"), ("points", "assists"), ("rebounds", "assists")]
EPS = 1e-6


def bvn_cdf(h, k, rho):
    """P(X <= h, Y <= k) for standard normals with correlation rho (Owen's T)."""
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    h = np.where(np.abs(h) < 1e-9, 1e-9, h)
    k = np.where(np.abs(k) < 1e-9, 1e-9, k)
    root = np.sqrt(1 - rho**2)
    t1 = owens_t(h, (k - rho * h) / (h * root))
    t2 = owens_t(k, (h - rho * k) / (k * root))
    beta = np.where(h * k < 0, 0.5, 0.0)
    return np.clip(0.5 * (norm.cdf(h) + norm.cdf(k)) - t1 - t2 - beta, 0, 1)


def joint_yes(pa, pb, rho):
    """P(A and B) for events with probabilities pa, pb linked by rho."""
    pa = np.clip(pa, EPS, 1 - EPS)
    pb = np.clip(pb, EPS, 1 - EPS)
    # A happens when the latent variable is above Phi^-1(1 - pa): by symmetry
    # P(A and B) = P(X <= Phi^-1(pa), Y <= Phi^-1(pb)).
    return bvn_cdf(norm.ppf(pa), norm.ppf(pb), rho)


def fit_rho(pa, pb, ya, yb):
    """Maximum-likelihood copula correlation from probabilities and outcomes."""
    pa = np.clip(np.asarray(pa, float), 1e-4, 1 - 1e-4)
    pb = np.clip(np.asarray(pb, float), 1e-4, 1 - 1e-4)
    ya = np.asarray(ya, bool)
    yb = np.asarray(yb, bool)

    def nll(rho):
        p11 = joint_yes(pa, pb, rho)
        cells = np.select(
            [ya & yb, ya & ~yb, ~ya & yb], [p11, pa - p11, pb - p11], 1 - pa - pb + p11
        )
        return -np.sum(np.log(np.clip(cells, 1e-12, 1)))

    return float(minimize_scalar(nll, bounds=(-0.9, 0.95), method="bounded").x)


def triple_yes(pa, pb, pc, rhos):
    """P(A and B and C) with the three pairwise correlations (ab, ac, bc)."""
    rab, rac, rbc = rhos
    cov = np.array([[1, rab, rac], [rab, 1, rbc], [rac, rbc, 1]])
    x = norm.ppf(np.clip(np.column_stack([pa, pb, pc]), EPS, 1 - EPS))
    out = multivariate_normal(mean=np.zeros(3), cov=cov, allow_singular=True).cdf(x)
    return np.clip(np.atleast_1d(out), 0, 1)


def double_double(pp, pr, pa, rhos):
    """P(at least two of points, rebounds, assists reach 10)."""
    rpr, rpa, rra = rhos
    both = joint_yes(pp, pr, rpr) + joint_yes(pp, pa, rpa) + joint_yes(pr, pa, rra)
    return np.clip(both - 2 * triple_yes(pp, pr, pa, rhos), 0, 1)


def independent_double_double(pp, pr, pa):
    return pp * pr + pp * pa + pr * pa - 2 * pp * pr * pa
