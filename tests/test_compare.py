"""Checks for the model-comparison metrics on synthetic data, and a look-ahead guard for the features."""

import numpy as np
import pandas as pd
from scipy import stats

from t6 import baseline, compare, deep

RNG = np.random.default_rng(7)


def _nb(mu, theta, n, rng=RNG):
    return rng.negative_binomial(theta, theta / (theta + mu), size=n).astype(float)


def test_log_score_matches_scipy_and_survives_tiny_poisson_mean():
    y, mu, var = (
        np.array([0.0, 2.0, 7.0]),
        np.array([1.5, 2.0, 4.0]),
        np.array([3.0, 5.0, 9.0]),
    )
    size, prob = baseline.nb_params(mu, var)
    assert np.allclose(
        compare.log_score(y, mu, var), stats.nbinom.logpmf(y, size, prob)
    )
    # prob rounds to 1 here and scipy returns -inf; the true Poisson log pmf is about -61.8
    ls = compare.log_score([3.0], [2.06e-9], [2.06e-9])
    assert np.isfinite(ls).all() and abs(ls[0] + 61.8) < 0.1


def test_true_distribution_scores_best_and_is_calibrated():
    mu, theta, n = 6.0, 2.0, 20000
    y = _nb(mu, theta, n)
    var = mu + mu**2 / theta
    good = compare.log_score(y, np.full(n, mu), np.full(n, var))
    shifted = compare.log_score(
        y, np.full(n, 2 * mu), np.full(n, 2 * mu + (2 * mu) ** 2 / theta)
    )
    assert good.mean() > shifted.mean()
    z = baseline.dunn_smyth(y, np.full(n, mu), np.full(n, var))
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05
    z_poisson = baseline.dunn_smyth(
        y, np.full(n, mu), np.full(n, mu)
    )  # ignores the overdispersion
    assert z_poisson.std() > 1.3
    assert (baseline.upper_p(y, np.full(n, mu), np.full(n, var)) < 0.01).mean() < 0.015


def test_bootstrap_ci_excludes_zero_for_a_consistent_gain_only():
    n_series, per = 40, 30
    g = np.repeat(np.arange(n_series), per)
    noise = RNG.normal(0, 1, n_series * per)
    mean, lo, hi = compare.boot_mean(0.3 + noise, g)
    assert lo > 0 and abs(mean - 0.3) < 0.1
    mean, lo, hi = compare.boot_mean(noise, g)
    assert lo < 0 < hi


def test_bootstrap_resamples_series_not_rows():
    """Series-level offsets make rows within a series dependent: the series CI must be wider."""
    n_series, per = 30, 50
    g = np.repeat(np.arange(n_series), per)
    d = np.repeat(RNG.normal(0, 1, n_series), per) + RNG.normal(0, 0.1, n_series * per)
    _, lo_s, hi_s = compare.boot_mean(d, g)
    _, lo_r, hi_r = compare.boot_mean(d, np.arange(len(d)))
    assert (hi_s - lo_s) > 2 * (hi_r - lo_r)


def test_win_rate_counts_series_not_rows():
    g = np.array([0] * 10 + [1] * 2 + [2] * 2)
    d = np.array([1.0] * 10 + [-1.0] * 2 + [-1.0] * 2)
    assert np.isclose(compare.win_rate(d, g), 1 / 3)


def test_lag1_autocorrelation_of_residuals():
    n_series, n = 30, 80
    iid = RNG.normal(size=(n_series, n))
    ar = np.zeros_like(iid)
    for t in range(1, n):
        ar[:, t] = 0.7 * ar[:, t - 1] + RNG.normal(size=n_series)
    g = np.repeat(np.arange(n_series), n)
    assert abs(compare.mean_acf1(iid.ravel(), g)) < 0.1
    assert compare.mean_acf1(ar.ravel(), g) > 0.5


def _synthetic_frame(n_series=40, months=36):
    """Truth is NB with a per-series mean; 'baseline' ignores the series mean, 'ml' knows it."""
    mus = np.exp(RNG.normal(1.0, 1.0, n_series))
    rows = []
    idx = pd.date_range("2020-01-01", periods=months, freq="MS")
    for s, mu in enumerate(mus):
        y = _nb(mu, 3.0, months)
        rows.append(
            pd.DataFrame(
                {
                    "sid": s,
                    "did": s % 5,
                    "level": "country",
                    "month": idx,
                    "y": y,
                    "mu_ml": mu,
                    "var_ml": mu + mu**2 / 3,
                    "mu_baseline": mus.mean(),
                    "var_baseline": mus.mean() + mus.mean() ** 2 / 3,
                    "age": 40,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df["bmu"], df["ratio"] = df["mu_baseline"], df["y"] / df["mu_baseline"]
    df["lag1"] = df.groupby("sid")["y"].shift(1).fillna(0)
    df["prev12"] = df.groupby("sid")["y"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).mean()
    )
    for m in ("ml", "baseline"):
        df[f"ls_{m}"] = compare.log_score(df["y"], df[f"mu_{m}"], df[f"var_{m}"])
        df[f"z_{m}"] = baseline.dunn_smyth(df["y"], df[f"mu_{m}"], df[f"var_{m}"])
        df[f"p_{m}"] = baseline.upper_p(df["y"], df[f"mu_{m}"], df[f"var_{m}"])
    return df


def test_compare_table_and_decision_rule_on_synthetic_models():
    df = _synthetic_frame()
    res = compare.compare(df, ["baseline", "ml"])
    gain = res[
        (res["period"] == "2020-2024")
        & (res["model"] == "ml")
        & (res["kind"] == "all")
        & (res["metric"] == "dlogscore")
        & (res["ref"] == "best_stat")
    ].iloc[0]
    assert gain["lo"] > 0 and gain["hi"] >= gain["value"] >= gain["lo"]
    sd = res[
        (res["period"] == "2020-2024")
        & (res["kind"] == "all")
        & (res["model"] == "ml")
        & (res["metric"] == "sd_z")
    ].iloc[0]
    assert 0.9 < sd["value"] < 1.15
    n = res[
        (res["period"] == "2020-2024")
        & (res["kind"] == "all")
        & (res["metric"] == "n_series")
    ].iloc[0]
    assert n["n"] == len(df[df["month"] <= "2024-12-31"])
    verdict = compare.decide(res, ["baseline", "ml"]).iloc[0]
    assert bool(verdict["gain_2020_24_ci_above_0"]) and verdict["verdict"] in {
        "supplement",
        "drop",
    }
    assert not verdict[
        "alternative"
    ]  # no model explains flags without a parametric form


def test_features_use_only_data_before_the_target_month():
    """Scrambling every count from month t0 on must leave the inputs for target month t0 unchanged."""
    rng = np.random.default_rng(3)
    S, T = 25, 200
    Y = rng.poisson(rng.uniform(0.2, 8, (S, 1)), (S, T)).astype(float)
    Y[:, : rng.integers(0, 40)] = 0
    lvl = rng.integers(0, 3, S)
    sp = deep.split(2018, T)
    t0 = int(sp.test[0])
    Y2 = Y.copy()
    Y2[:, t0:] = rng.poisson(50, (S, T - t0))
    out = []
    for A in (Y, Y2):
        fn = deep.first_nonzero(A)
        sets = deep.lstm_sets(A, fn, lvl, sp)
        gb = deep.gbm_sets(A, fn, lvl, sp)
        keep = sets["test"][1] == t0
        gkeep = gb["test"][1] == t0
        out.append(
            (
                sets["test"][0][keep],
                sets["test"][2][keep],
                sets["test"][3][keep],
                gb["test"][2][gkeep],
                sets["train"][2],
                sets["val"][2],
            )
        )
    for a, b in zip(*out):
        assert np.array_equal(a, b)
    assert len(out[0][0]) > 0
