import numpy as np
import pytest
from scipy.stats import multivariate_normal

from markets.nba.correlation import (
    bvn_cdf,
    double_double,
    fit_rho,
    independent_double_double,
    joint_yes,
)


def test_bvn_matches_scipy():
    rng = np.random.default_rng(0)
    for rho in [-0.5, 0.0, 0.4, 0.8]:
        h, k = rng.normal(size=4), rng.normal(size=4)
        ref = [
            multivariate_normal(cov=[[1, rho], [rho, 1]]).cdf([a, b])
            for a, b in zip(h, k)
        ]
        assert np.allclose(bvn_cdf(h, k, rho), ref, atol=1e-6)


def test_zero_correlation_is_independence():
    assert joint_yes(0.3, 0.4, 0.0) == pytest.approx(0.12, abs=1e-6)
    dd = double_double(np.array([0.5]), np.array([0.4]), np.array([0.2]), [0, 0, 0])
    assert dd[0] == pytest.approx(independent_double_double(0.5, 0.4, 0.2), abs=1e-5)


def test_fit_rho_recovers_simulated_correlation():
    rng = np.random.default_rng(1)
    z = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], size=40000)
    pa, pb = np.full(40000, 0.3), np.full(40000, 0.2)
    from scipy.stats import norm

    ya, yb = z[:, 0] < norm.ppf(pa), z[:, 1] < norm.ppf(pb)
    assert fit_rho(pa, pb, ya, yb) == pytest.approx(0.4, abs=0.04)
