"""Checks for the pooled negative-binomial regressions and the placebo shuffle."""

import numpy as np
import pandas as pd

from t6 import nbmodels as N

RNG = np.random.default_rng(1)


def _simulate(n=8000, alpha=1.0):
    X = np.column_stack([np.ones(n), RNG.normal(size=(n, 3))])
    beta = np.array([0.4, 0.5, -0.3, 0.2])
    mu = np.exp(X @ beta)
    r = 1 / alpha
    y = RNG.negative_binomial(r, r / (r + mu)).astype(float)
    return X, y, beta


def test_irls_nb_recovers_coefficients():
    X, y, beta = _simulate()
    est = N.irls_nb(X, y, alpha=1.0)
    assert np.allclose(est, beta, atol=0.08)


def test_alpha_is_estimated():
    X, y, _ = _simulate(alpha=0.5)
    _, alpha = N.fit(X, y)
    assert 0.4 < alpha < 0.65


def test_cluster_covariance_is_psd_and_positive():
    X, y, _ = _simulate(n=3000)
    beta = N.irls_nb(X, y, 1.0)
    groups = RNG.integers(0, 40, size=len(y))
    cov = N.cluster_cov(X, y, beta, 1.0, groups)
    assert np.allclose(cov, cov.T) and (np.diag(cov) > 0).all()
    assert np.linalg.eigvalsh(cov).min() > -1e-9


def test_loglik_improves_with_a_true_predictor():
    X, y, _ = _simulate()
    full = N.nb_llf(X, y, N.irls_nb(X, y, 1.0), 1.0)
    reduced = N.nb_llf(X[:, :2], y, N.irls_nb(X[:, :2], y, 1.0), 1.0)
    assert full > reduced


def test_year_shuffle_keeps_values_and_within_year_order():
    n_years = 5
    D = {
        "keys": pd.DataFrame({"area": ["A", "B"]}),
        "Y": np.zeros((2, 12 * n_years)),
        "clim": {v: RNG.normal(size=(2, 12 * n_years)) for v in N.VARS},
    }
    out = N._permute_years(D, np.random.default_rng(3))
    for v in N.VARS:
        for i in range(2):
            a = D["clim"][v][i].reshape(n_years, 12)
            b = out[v][i].reshape(n_years, 12)
            assert sorted(map(tuple, a)) == sorted(
                map(tuple, b)
            )  # same years, only reordered
    assert not np.array_equal(out[N.VARS[0]], D["clim"][N.VARS[0]])
    # same permutation for all variables within a country: years stay aligned across variables
    a0 = D["clim"][N.VARS[0]][0].reshape(n_years, 12)
    a1 = D["clim"][N.VARS[1]][0].reshape(n_years, 12)
    b0, b1 = (out[v][0].reshape(n_years, 12) for v in N.VARS[:2])
    for k in range(n_years):
        i0 = int(np.flatnonzero((a0 == b0[k]).all(1))[0])
        assert np.array_equal(a1[i0], b1[k])
