"""Validation of the baseline against a reference GLM and against simulated benchmarks."""

import numpy as np
import pytest
import statsmodels.api as sm

from t6 import baseline

RNG = np.random.default_rng(42)


def _simulate(n_months=84, level=20.0, amp=0.5, theta=8.0, rng=RNG):
    """Seasonal negative-binomial series with known mean (overdispersed by 1/theta)."""
    cal = np.arange(n_months) % 12 + 1
    mu = level * np.exp(amp * np.sin(2 * np.pi * (cal - 1) / 12))
    y = rng.negative_binomial(theta, theta / (theta + mu))
    return y.astype(float), cal, mu


def test_irls_matches_statsmodels():
    y, cal, _ = _simulate()
    t = (np.arange(len(y)) - len(y)) / 12.0
    X = baseline.design(t, cal, n_harm=1, trend=True)
    beta, _, _ = baseline.irls_poisson(X, y)
    ref = sm.GLM(y, X, family=sm.families.Poisson()).fit().params
    assert np.allclose(beta, ref, atol=1e-4)


def test_false_alarm_rate_is_controlled_under_the_null():
    """On series drawn from a seasonal NB, nominal 1% tail should not be badly exceeded."""
    hits, tests = 0, 0
    for _ in range(40):
        y, cal, _ = _simulate(n_months=72 + 60)
        for i in range(72, len(y)):
            w = slice(i - 60, i)
            t = (np.arange(i - 60, i) - i) / 12.0
            p = baseline.fit_predict(y[w], t, cal[w], 0.0, cal[i], n_harm=1, trend=True)
            hits += baseline.upper_p(y[i], p.mu, p.var) < 0.01
            tests += 1
    assert hits / tests < 0.02  # allows some slack over the nominal 1%


def test_injected_outbreak_is_detected():
    y, cal, mu = _simulate(n_months=72)
    i = len(y) - 1
    y[i] = mu[i] * 6
    w = slice(i - 60, i)
    t = (np.arange(i - 60, i) - i) / 12.0
    p = baseline.fit_predict(y[w], t, cal[w], 0.0, cal[i], n_harm=1, trend=True)
    assert baseline.upper_p(y[i], p.mu, p.var) < 1e-3


def test_past_peak_is_downweighted():
    """A huge past wave must not push the expected count of a normal month far upward."""
    y, cal, mu = _simulate(n_months=72)
    clean = y.copy()
    y[30:33] = mu[30:33] * 15
    i = len(y) - 1
    w = slice(i - 60, i)
    t = (np.arange(i - 60, i) - i) / 12.0
    with_wave = baseline.fit_predict(y[w], t, cal[w], 0.0, cal[i], 1, True).mu
    without = baseline.fit_predict(clean[w], t, cal[w], 0.0, cal[i], 1, True).mu
    assert with_wave < 2.0 * without


def test_no_overshoot_the_season_after_a_wave():
    """A one-off wave must not be predicted again at the same season next year."""
    rng = np.random.default_rng(7)
    ratios = []
    for _ in range(20):
        y, cal, _mu = _simulate(n_months=72, level=5.0, amp=0.8, theta=5.0, rng=rng)
        clean = y.copy()
        y[41:44] = (
            np.maximum(y[41:44], 1) * 200
        )  # wave in the 6th year of data, months 42-44
        i = 53  # same season, one year after the wave
        w = slice(i - 48, i)
        t = (np.arange(i - 48, i) - i) / 12.0
        with_wave = baseline.fit_predict(y[w], t, cal[w], 0.0, cal[i], 2, True).mu
        without = baseline.fit_predict(clean[w], t, cal[w], 0.0, cal[i], 2, True).mu
        ratios.append(with_wave / without)
    assert np.median(ratios) < 1.5


def test_recurring_seasonal_peak_is_kept():
    """A peak that repeats every year is baseline, not a wave: screening must not remove it."""
    y, cal, mu = _simulate(n_months=72, level=20.0, amp=1.0, theta=20.0)
    i = 71
    w = slice(i - 60, i)
    t = (np.arange(i - 60, i) - i) / 12.0
    p = baseline.fit_predict(y[w], t, cal[w], 0.0, cal[i], 1, False)
    assert 0.6 < p.mu / mu[i] < 1.6


def test_poisson_limit_when_no_overdispersion():
    size, prob = baseline.nb_params(5.0, 5.0)
    assert size > 1e8 and prob == pytest.approx(1.0, abs=1e-6)


def test_sparse_window_does_not_explode():
    """Two non-zero months in 60 (real ASF window, Estonia 2025) used to give mu = exp(20)."""
    y = np.zeros(60)
    y[57], y[59] = 15.0, 6.0
    cal = np.arange(60) % 12 + 1
    t = (np.arange(60) - 60) / 12.0
    for harm, trend in ((1, True), (2, True), (0, True)):
        p = baseline.fit_predict(y, t, cal, 0.0, 6, harm, trend)
        assert np.isfinite(p.mu) and p.mu < 1.0 and np.isfinite(p.var)
