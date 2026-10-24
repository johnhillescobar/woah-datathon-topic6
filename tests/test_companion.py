"""Checks for the companion layer: no look-ahead in the forward forecast, interval ordering, alert rule,
agreement bookkeeping and the ensemble / adoption arithmetic."""
import numpy as np
import pandas as pd

from t6 import companion as K

RNG = np.random.default_rng(11)


def _panel(n_series=36, n_obs=96):
    """Seasonal Poisson counts plus one placeholder column for the month to forecast."""
    mu = np.exp(RNG.normal(0.8, 0.8, (n_series, 1))) * (1 + 0.5 * np.sin(2 * np.pi * np.arange(n_obs + 1) / 12))
    Y = RNG.poisson(mu).astype(float)
    Y[:, :3] = 0
    Y[np.arange(n_series), RNG.integers(3, 8, n_series)] = 2  # every series has a first count early on
    lvl = RNG.integers(0, 3, n_series)
    return Y, lvl


def test_forward_forecast_ignores_the_unobserved_month():
    Y, lvl = _panel()
    other = Y.copy()
    other[:, -1] = RNG.poisson(500, len(Y))  # the month being forecast
    s1, mu1, th1 = K.forward_gbm(Y, lvl)
    s2, mu2, th2 = K.forward_gbm(other, lvl)
    assert np.array_equal(s1, s2) and np.array_equal(mu1, mu2) and np.array_equal(th1, th2)
    changed = Y.copy()
    changed[:, -2] = changed[:, -2] + 25  # the last observed month does matter
    assert not np.allclose(K.forward_gbm(changed, lvl)[1], mu1)


def test_intervals_are_ordered_integers_and_cover():
    mu = np.r_[1e-3, 0.05, 0.3, 1.0, 4.0, 40.0, 500.0, RNG.uniform(0.01, 200, 50)]
    for theta in (0.03, 0.5, 5.0, 1000.0):
        iv = K.intervals(mu, np.full(len(mu), theta))
        assert (iv["lo95"] <= iv["lo80"]).all() and (iv["hi80"] <= iv["hi95"]).all()
        assert (iv["lo80"] <= mu).all() and (mu <= iv["hi80"]).all()
        assert (iv["lo95"] >= 0).all() and np.array_equal(iv.to_numpy(), np.round(iv.to_numpy()))
    n, m, th = 40000, 6.0, 1.5
    y = RNG.negative_binomial(th, th / (th + m), n)
    iv = K.intervals(np.array([m]), np.array([th])).iloc[0]
    assert ((y >= iv["lo80"]) & (y <= iv["hi80"])).mean() >= 0.79
    assert ((y >= iv["lo95"]) & (y <= iv["hi95"])).mean() >= 0.94


def _rolling(n=400):
    months = pd.date_range("2018-01-01", periods=n // 4, freq="MS")
    keys = pd.DataFrame({"level": "country", "disease": np.repeat(list("abcd"), len(months)),
                         "area": "XXX", "month": np.tile(months, 4)})
    mu = RNG.uniform(0.2, 5, len(keys))
    y = RNG.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
    y[RNG.choice(len(keys), 12, replace=False)] += 60  # injected surprises
    return keys.assign(metric="outbreaks", y=y, mu=mu, theta=2.0, var=mu + mu**2 / 2.0, status="tested")


def test_companion_alerts_follow_the_project_rule():
    r = _rolling()
    scored = K.companion_alerts(r, r[K.KEY])
    hit = scored[scored["alert"]]
    assert len(hit) > 0
    assert (hit["q"] < 0.05).all() and (hit["y"] >= 3).all() and (hit["ratio"] >= 2).all()
    assert len(scored) == len(r)


def test_agreement_counts_partition_the_flagged_months():
    r = _rolling()
    scored = K.companion_alerts(r, r[K.KEY])
    comp = scored[scored["alert"]]
    extra = scored[~scored["alert"]].sample(30, random_state=1)
    stat = {"ref": pd.concat([comp.iloc[: len(comp) // 2], extra])[K.KEY]}
    ag = K.agreement(scored, stat)
    a = ag[(ag["level"] == "all")]
    assert (a["n_both"] + a["n_only_companion"] == a["n_companion"]).all()
    assert (a["n_both"] + a["n_only_statistical"] == a["n_statistical"]).all()
    assert a["n_companion"].sum() == len(comp) and a["n_statistical"].sum() == len(stat["ref"])
    assert (a["n_scored"].sum() == len(scored))


def test_ensemble_is_geometric_in_the_mean_and_harmonic_in_theta():
    base = _rolling(40)[K.KEY + ["y"]]
    g = base.assign(mu=4.0, theta=2.0)
    l = base.assign(mu=9.0, theta=8.0)
    e = K.ensemble_frame(g, l)
    assert np.allclose(e["mu"], 6.0)
    assert np.allclose(1 / e["theta"], 0.5 * (1 / 2.0 + 1 / 8.0))
    assert np.allclose(e["var"], 6.0 + 36.0 / e["theta"])


def test_adoption_conditions_read_the_table():
    def row(period, model, metric, value, lo=np.nan, ref=None):
        return {"period": period, "model": model, "ref": ref, "metric": metric, "value": value, "lo": lo,
                "hi": np.nan, "n": 1}
    good = pd.DataFrame([row("2020-2024", "m", "dlogscore", 0.1, 0.05, "gbm"), row("2025-2026", "m", "dlogscore", 0.2, -0.1, "gbm"),
                         row("2020-2024", "m", "sd_z", 1.0), row("2025-2026", "m", "sd_z", 1.1)])
    assert K.passes(good, "m", "gbm")["passes"]
    for bad in (good.assign(lo=good["lo"].where(good["metric"] != "dlogscore", -0.01)),
                good.assign(value=good["value"].where(~((good["period"] == "2025-2026") & (good["metric"] == "dlogscore")), -0.01)),
                good.assign(value=good["value"].where(~((good["period"] == "2025-2026") & (good["metric"] == "sd_z")), 1.4))):
        assert not K.passes(bad, "m", "gbm")["passes"]
