"""Companion layer: an "expected next month given recent activity" forecast beside the statistical alerts.

The statistical baseline is the alert reference (a "normal" level that excludes one-off waves). The
companion is the pooled one-step model that lets recent activity, including a wave under way, inform
the expectation. Run with `uv run python -m t6.companion [--model gbm|lstm|ensemble]`.

* gbm (default; needs no torch): final LightGBM fit on all targets through the last scored month, true
  forward forecast for the next month with 80% / 95% NB intervals, retrospective companion alerts on the
  rolling predictions (the project's own FDR + count + ratio rule), agreement with the statistical alerts,
  two figures, and a check that refitting on the validation block does not hurt in the rolling benchmark.
* lstm / ensemble (stage 2, needs torch only for the forward fits): paired comparison with the GBM on the
  rolling predictions and the adoption rule in reports/dl_evaluation.md; if adopted, forward forecasts
  from final LSTM fits (3 seeds, cached in data/processed/deep_cache) are added to companion_forecast.csv.
  Stage 1 rewrites that file with the GBM columns only; rerun `--model lstm` to add the LSTM ones back.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from t6 import baseline, compare, deep, detect
from t6 import config as C

TABLES = C.REPORTS / "tables"
FIGURES = C.REPORTS / "figures"
KEY = compare.KEY
PERIODS = compare.PERIODS
STAT_FILES = {"baseline": "outbreaks", "ar1": "outbreaks_ar1"}
BLUE, ORANGE, GREY, SURFACE, INK = "#2a78d6", "#eb6834", "#898781", "#fcfcfb", "#2b2a28"
SHOWN = [
    ("global", "Bluetongue virus", "all", "Bluetongue virus, global"),
    ("region", "HPAI (all hosts)", "Europe", "HPAI (all hosts), Europe"),
]


# ---- forward forecast -------------------------------------------------------------------------


def forward_split(n_obs: int) -> deep.Split:
    """Fit window for the month after the data: validation block = the last 12 observed months."""
    return deep.Split(0, n_obs - 1 - deep.VAL_MONTHS, n_obs - 1, np.array([n_obs]))


def level_codes(keys: pd.DataFrame) -> np.ndarray:
    return keys["level"].map({l: i for i, l in enumerate(deep.LEVELS)}).to_numpy()


def forward_gbm(Yx: np.ndarray, lvl: np.ndarray, refit: bool = True):
    """Forecast the month after the last observed one. Yx = observed counts plus one placeholder column
    for the month to forecast; the placeholder is never read (first-count months come from observed
    columns only). Returns (series index, expected count, theta)."""
    n_obs = Yx.shape[1] - 1
    fn = deep.first_nonzero(Yx[:, :n_obs])
    s, _, mu, th, _ = deep.gbm_fit_predict(Yx, fn, lvl, forward_split(n_obs), refit)
    return s, mu, th


def intervals(mu, theta) -> pd.DataFrame:
    """80% / 95% NB predictive intervals as integer counts, widened where needed to contain the mean
    (a skewed count can have its mean above the 90th percentile)."""
    mu, theta = np.asarray(mu, float), np.asarray(theta, float)
    size, prob = baseline.nb_params(mu, mu + mu**2 / theta)
    q = lambda p: stats.nbinom.ppf(p, size, prob)
    return pd.DataFrame(
        {
            "lo95": np.minimum(q(0.025), np.floor(mu)),
            "lo80": np.minimum(q(0.10), np.floor(mu)),
            "hi80": np.maximum(q(0.90), np.ceil(mu)),
            "hi95": np.maximum(q(0.975), np.ceil(mu)),
        }
    )


def last_month_surprise(rolling: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Upper-tail p-value of the last observed month under that month's one-step prediction."""
    d = rolling[rolling["month"] == month].copy()
    d["last_month_p"] = baseline.upper_p(d["y"], d["mu"], d["var"])
    return d[["level", "disease", "area", "mu", "last_month_p"]].rename(
        columns={"mu": "last_month_expected"}
    )


def forecast_table(keys, Y, idx, s, mu, th, rolling) -> pd.DataFrame:
    out = keys.iloc[s].reset_index(drop=True)
    out["last_month"] = idx[-1]
    out["last_observed"] = Y[s, -1]
    out["forecast_month"] = idx[-1] + pd.DateOffset(months=1)
    out["expected"], out["theta"] = mu, th
    out = pd.concat([out, intervals(mu, th)], axis=1)
    out = out.merge(
        last_month_surprise(rolling, idx[-1]),
        how="left",
        on=["level", "disease", "area"],
    )
    cols = [
        "level",
        "disease",
        "area",
        "last_month",
        "last_observed",
        "forecast_month",
        "expected",
        "lo80",
        "hi80",
        "lo95",
        "hi95",
        "theta",
        "last_month_expected",
        "last_month_p",
    ]
    return out[cols]


# ---- rolling predictions: companion alerts and agreement --------------------------------------


def tested_keys() -> pd.DataFrame:
    """Rows the statistical models test (the common scoring set), 2018 on."""
    d = pd.read_parquet(C.PROCESSED / "tests_outbreaks.parquet")
    d = d[(d["status"] == "tested") & (d["month"] >= pd.Timestamp("2018-01-01"))]
    return d[KEY].assign(month=lambda x: x["month"].astype("datetime64[ns]"))


def read_rolling(model: str) -> pd.DataFrame:
    d = pd.read_parquet(C.PROCESSED / f"tests_outbreaks_{model}.parquet")
    d["month"] = d["month"].astype("datetime64[ns]")
    return d


def companion_alerts(rolling: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """The project's alert rule (BH FDR q < 0.05, y >= 3, y/expected >= 2; detect.score) applied to the
    companion's one-step predictions on the rows the statistical models test. The FDR family is those
    rows from 2018 on, a smaller family than the statistical alerts' 2007-2026 one."""
    d = rolling.merge(keys, on=KEY)[
        KEY + ["metric", "y", "mu", "var", "theta", "status"]
    ]
    return detect.score(d, "outbreaks")


def period_of(month: pd.Series) -> pd.Series:
    out = pd.Series("", index=month.index)
    for name, (a, b) in PERIODS.items():
        out[(month >= a) & (month <= b)] = name
    return out


def statistical_alerts() -> dict:
    out = {}
    for name, sfx in STAT_FILES.items():
        a = pd.read_csv(TABLES / f"alerts_{sfx}.csv", parse_dates=["month"])
        a["month"] = a["month"].astype("datetime64[ns]")
        out[name] = a[a["month"] >= pd.Timestamp("2018-01-01")]
    return out


def agreement(scored: pd.DataFrame, stat: dict) -> pd.DataFrame:
    """Flagged months by both / only the companion / only the statistical model, per level and period."""
    base = scored[KEY + ["alert"]].rename(columns={"alert": "companion"}).copy()
    base["period"] = period_of(base["month"])
    rows = []
    for name, a in stat.items():
        m = base.merge(a[KEY].assign(statistical=True), how="left", on=KEY)
        m["statistical"] = m["statistical"].fillna(False).astype(bool)
        assert m["statistical"].sum() == len(a), (
            f"{name} alerts outside the scored rows"
        )
        for period in PERIODS:
            for level in ("all", "global", "region", "country"):
                g = m[
                    (m["period"] == period) & ((m["level"] == level) | (level == "all"))
                ]
                c, s = g["companion"], g["statistical"]
                rows.append(
                    {
                        "reference": name,
                        "period": period,
                        "level": level,
                        "n_scored": len(g),
                        "n_both": int((c & s).sum()),
                        "n_only_companion": int((c & ~s).sum()),
                        "n_only_statistical": int((~c & s).sum()),
                        "n_companion": int(c.sum()),
                        "n_statistical": int(s.sum()),
                    }
                )
    return pd.DataFrame(rows)


def flagged_profile(
    scored: pd.DataFrame, stat: dict, ctx: pd.DataFrame
) -> pd.DataFrame:
    """What the months in each agreement class look like: size, the previous month's count, how often the
    series was already in alert last month, and the observed/expected ratios of the two models."""
    rows = []
    comp = scored[KEY + ["y", "mu", "alert"]].rename(
        columns={"mu": "mu_c", "alert": "companion"}
    )
    comp = comp.merge(ctx[KEY + ["lag1"]], on=KEY, how="left")
    for name, a in stat.items():
        ref = pd.read_parquet(C.PROCESSED / f"tests_{STAT_FILES[name]}.parquet")
        ref["month"] = ref["month"].astype("datetime64[ns]")
        m = comp.merge(
            ref[KEY + ["mu"]].rename(columns={"mu": "mu_s"}), on=KEY, how="left"
        )
        m = m.merge(a[KEY].assign(statistical=True), how="left", on=KEY)
        m["statistical"] = m["statistical"].fillna(False).astype(bool)
        prev = a[KEY].copy()
        prev["month"] = prev["month"] + pd.DateOffset(months=1)
        m = m.merge(prev.assign(prev_stat=True), how="left", on=KEY)
        m["prev_stat"] = m["prev_stat"].fillna(False).astype(bool)
        m["klass"] = np.select(
            [m["companion"] & m["statistical"], m["companion"], m["statistical"]],
            ["both", "only_companion", "only_statistical"],
            "neither",
        )
        for klass, g in m[m["klass"] != "neither"].groupby("klass"):
            rows.append(
                {
                    "reference": name,
                    "class": klass,
                    "n": len(g),
                    "median_count": g["y"].median(),
                    "median_previous_month": g["lag1"].median(),
                    "share_previous_month_ge3": (g["lag1"] >= 3).mean(),
                    "share_previous_month_statistical_alert": g["prev_stat"].mean(),
                    "median_obs_over_companion": (g["y"] / g["mu_c"]).median(),
                    "median_obs_over_statistical": (g["y"] / g["mu_s"]).median(),
                }
            )
    return pd.DataFrame(rows)


# ---- paired comparisons on the rolling predictions --------------------------------------------


def joined(frames: dict) -> pd.DataFrame:
    """Rows every frame scores (2018 on) with each model's log score, Dunn-Smyth residual and tail p."""
    frames = {
        m: d.assign(month=d["month"].astype("datetime64[ns]"))
        for m, d in frames.items()
    }
    df = compare.load(frames)
    df["sid"] = df.groupby(["level", "disease", "area"]).ngroup()
    for m in frames:
        df[f"ls_{m}"] = compare.log_score(df["y"], df[f"mu_{m}"], df[f"var_{m}"])
        df[f"z_{m}"] = baseline.dunn_smyth(
            df["y"], df[f"mu_{m}"], df[f"var_{m}"], seed=0
        )
        df[f"p_{m}"] = baseline.upper_p(df["y"], df[f"mu_{m}"], df[f"var_{m}"])
    return df


def paired_table(df: pd.DataFrame, models: list, pairs: list) -> pd.DataFrame:
    """Per period: each model's mean log score, residual sd and p<0.01 share; each (model, ref) gain with a
    95% bootstrap CI over series and the share of series where the model wins."""
    rows = []
    for period, (a, b) in PERIODS.items():
        mask = ((df["month"] >= a) & (df["month"] <= b)).to_numpy()
        n = int(mask.sum())
        for m in models:
            for metric, v in (
                ("mean_logscore", df.loc[mask, f"ls_{m}"].mean()),
                ("sd_z", df.loc[mask, f"z_{m}"].std()),
                ("share_p01", (df.loc[mask, f"p_{m}"] < 0.01).mean()),
            ):
                rows.append(
                    {
                        "period": period,
                        "model": m,
                        "ref": None,
                        "metric": metric,
                        "value": float(v),
                        "lo": np.nan,
                        "hi": np.nan,
                        "n": n,
                    }
                )
        sid = df.loc[mask, "sid"].to_numpy()
        for m, r in pairs:
            d = (df[f"ls_{m}"] - df[f"ls_{r}"]).to_numpy()[mask]
            v, lo, hi = compare.boot_mean(d, sid)
            rows.append(
                {
                    "period": period,
                    "model": m,
                    "ref": r,
                    "metric": "dlogscore",
                    "value": v,
                    "lo": lo,
                    "hi": hi,
                    "n": n,
                }
            )
            rows.append(
                {
                    "period": period,
                    "model": m,
                    "ref": r,
                    "metric": "win_rate",
                    "value": compare.win_rate(d, sid),
                    "lo": np.nan,
                    "hi": np.nan,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def _val(tab, period, model, metric, ref=None):
    q = tab[
        (tab["period"] == period) & (tab["model"] == model) & (tab["metric"] == metric)
    ]
    q = q[q["ref"].isna()] if ref is None else q[q["ref"] == ref]
    return q.iloc[0]


def ensemble_frame(gbm: pd.DataFrame, lstm: pd.DataFrame) -> pd.DataFrame:
    """Geometric mean of the two expected counts; NB dispersion from the average of 1/theta."""
    m = gbm.merge(lstm, on=KEY + ["y"], suffixes=("_g", "_l"))
    mu = np.sqrt(m["mu_g"] * m["mu_l"])
    theta = 1.0 / (0.5 / m["theta_g"] + 0.5 / m["theta_l"])
    return m[KEY + ["y"]].assign(
        mu=mu, var=mu + mu**2 / theta, theta=theta, status="tested", metric="outbreaks"
    )


def coverage(frames: dict, keys: pd.DataFrame) -> pd.DataFrame:
    """Share of observed counts inside the 80% / 95% intervals, by period and expected-count bin."""
    bins = [
        ("<0.5", 0, 0.5),
        ("0.5-2", 0.5, 2),
        ("2-10", 2, 10),
        (">10", 10, np.inf),
        ("all", 0, np.inf),
    ]
    rows = []
    for model, d in frames.items():
        d = d.merge(keys, on=KEY)
        iv = intervals(d["mu"], d["theta"])
        d = d.assign(
            in80=(d["y"] >= iv["lo80"]) & (d["y"] <= iv["hi80"]),
            in95=(d["y"] >= iv["lo95"]) & (d["y"] <= iv["hi95"]),
        )
        for period, (a, b) in PERIODS.items():
            for name, lo, hi in bins:
                g = d[
                    (d["month"] >= a)
                    & (d["month"] <= b)
                    & (d["mu"] >= lo)
                    & (d["mu"] < hi)
                ]
                rows.append(
                    {
                        "model": model,
                        "period": period,
                        "expected_bin": name,
                        "n": len(g),
                        "coverage80": g["in80"].mean(),
                        "coverage95": g["in95"].mean(),
                    }
                )
    return pd.DataFrame(rows)


def passes(tab: pd.DataFrame, model: str, ref: str) -> dict:
    """Pre-registered adoption conditions of `model` against `ref` (see reports/dl_evaluation.md)."""
    g_main, g_hold = (
        _val(tab, "2020-2024", model, "dlogscore", ref),
        _val(tab, "2025-2026", model, "dlogscore", ref),
    )
    sd = [_val(tab, p, model, "sd_z")["value"] for p in ("2020-2024", "2025-2026")]
    c = {
        "gain_2020_24_ci_above_0": bool(g_main["lo"] > 0),
        "gain_2025_26_not_negative": bool(g_hold["value"] >= 0),
        "sd_z_within_0.9_1.3": all(0.9 <= s <= 1.3 for s in sd),
    }
    return {**c, "passes": all(c.values())}


def stage2_tables(keys: pd.DataFrame):
    """Rolling comparison of GBM, LSTM and their ensemble on the rows the statistical models test."""
    gbm, lstm = read_rolling("gbm"), read_rolling("lstm")
    ens = ensemble_frame(gbm, lstm)
    ref = read_rolling("gbm").merge(keys, on=KEY)[
        KEY + ["y", "mu", "var"]
    ]  # only fixes the row set
    df = joined({"rows": ref, "gbm": gbm, "lstm": lstm, "ensemble": ens})
    tab = paired_table(
        df,
        ["gbm", "lstm", "ensemble"],
        [("lstm", "gbm"), ("ensemble", "gbm"), ("ensemble", "lstm")],
    )
    verdicts = {m: passes(tab, m, "gbm") for m in ("lstm", "ensemble")}
    if verdicts["lstm"]["passes"] and verdicts["ensemble"]["passes"]:
        e_vs_l = _val(tab, "2020-2024", "ensemble", "dlogscore", "lstm")
        adopted = "ensemble" if e_vs_l["lo"] > 0 else "lstm"
    else:
        adopted = next((m for m in ("lstm", "ensemble") if verdicts[m]["passes"]), None)
    return tab, verdicts, adopted


def refit_check(force: bool = False) -> pd.DataFrame:
    """Rolling benchmark of the GBM variant used for the forward forecast (refit on the validation block
    too, same thetas) against the benchmark GBM, so the final fit is not an unvalidated protocol."""
    path = TABLES / "companion_refit_check.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    keys, Y, idx = deep.load_wide()
    fn, lvl = deep.first_nonzero(Y), level_codes(keys)
    with ProcessPoolExecutor(
        min(len(deep.TEST_YEARS), max(1, (os.cpu_count() or 2) - 1))
    ) as ex:
        fits = list(ex.map(_gbm_year, [(Y, fn, lvl, y) for y in deep.TEST_YEARS]))
    parts = []
    for s, t, mu, th in fits:
        parts.append(
            pd.DataFrame(
                {
                    "level": keys["level"].to_numpy()[s],
                    "disease": keys["disease"].to_numpy()[s],
                    "area": keys["area"].to_numpy()[s],
                    "month": idx[t],
                    "y": Y[s, t],
                    "mu": mu,
                    "var": mu + mu**2 / th,
                    "status": "tested",
                    "metric": "outbreaks",
                }
            )
        )
    refit = pd.concat(parts, ignore_index=True)
    gbm = read_rolling("gbm")
    rows = gbm.merge(tested_keys(), on=KEY)[KEY + ["y", "mu", "var"]]
    df = joined({"rows": rows, "gbm": gbm, "gbm_refit": refit})
    tab = paired_table(df, ["gbm", "gbm_refit"], [("gbm_refit", "gbm")])
    tab.to_csv(path, index=False)
    return tab


def _gbm_year(task):
    Y, fn, lvl, year = task
    s, t, mu, th, _ = deep.gbm_fit_predict(
        Y, fn, lvl, deep.split(year, Y.shape[1]), refit=True
    )
    return s, t, mu, th


# ---- forward LSTM (stage 2) -------------------------------------------------------------------


def _lstm_forward(task):
    """Early-stopped fit (evaluated protocol) to find the epoch count, then a refit on all targets for
    that many epochs. Returns both forecasts."""
    Yx, fn, lvl, seed = task
    sp = forward_split(Yx.shape[1] - 1)
    s, _, mu1, th1, info = deep.lstm_fit_predict(Yx, fn, lvl, sp, seed)
    full = deep.Split(0, sp.val_end, sp.val_end, sp.test)
    _, _, mu2, th2, _ = deep.lstm_fit_predict(
        Yx, fn, lvl, full, seed, fixed_epochs=max(info["best_epoch"], 1)
    )
    return s, mu1, th1, mu2, th2, info["best_epoch"]


def forward_lstm(Yx: np.ndarray, lvl: np.ndarray, force: bool = False) -> dict:
    n_obs = Yx.shape[1] - 1
    fn = deep.first_nonzero(Yx[:, :n_obs])
    deep.CACHE.mkdir(parents=True, exist_ok=True)
    paths = {
        sd: deep.CACHE / f"companion_lstm_{n_obs}_s{sd}_{deep._key('lstm')}.npz"
        for sd in deep.LSTM["seeds"]
    }
    todo = [sd for sd, p in paths.items() if force or not p.exists()]
    if todo:
        with ProcessPoolExecutor(len(todo)) as ex:
            for sd, res in zip(
                todo, ex.map(_lstm_forward, [(Yx, fn, lvl, sd) for sd in todo])
            ):
                np.savez(
                    paths[sd],
                    s=res[0],
                    mu1=res[1],
                    th1=res[2],
                    mu2=res[3],
                    th2=res[4],
                    best_epoch=res[5],
                )
    fits = [np.load(p) for p in paths.values()]
    return {
        "s": fits[0]["s"],
        "mu": np.mean([f["mu2"] for f in fits], 0),
        "theta": np.mean([f["th2"] for f in fits], 0),
        "mu_evaluated_protocol": np.mean([f["mu1"] for f in fits], 0),
        "best_epochs": [int(f["best_epoch"]) for f in fits],
    }


def add_model_columns(
    fc: pd.DataFrame,
    suffix: str,
    mu: np.ndarray,
    th: np.ndarray,
    rolling: pd.DataFrame,
    month: pd.Timestamp,
) -> pd.DataFrame:
    """Append a model's forward forecast (aligned to the rows of fc) and its last-month surprise."""
    fc = fc.copy()
    fc[f"expected_{suffix}"], fc[f"theta_{suffix}"] = mu, th
    iv = intervals(mu, th)[["lo80", "hi80", "lo95", "hi95"]].add_suffix(f"_{suffix}")
    fc = pd.concat([fc, iv], axis=1)
    sur = last_month_surprise(rolling, month).rename(
        columns={"last_month_p": f"last_month_p_{suffix}"}
    )
    return fc.merge(
        sur[["level", "disease", "area", f"last_month_p_{suffix}"]],
        how="left",
        on=["level", "disease", "area"],
    )


# ---- figures ----------------------------------------------------------------------------------


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=8, width=0.6)
    ax.grid(axis="y", color=GREY, alpha=0.2, linewidth=0.5)


def figure_series(rolling, scored, fc, stat, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = pd.read_parquet(C.PROCESSED / "tests_outbreaks.parquet")
    base["month"] = base["month"].astype("datetime64[ns]")
    fig, axes = plt.subplots(
        len(SHOWN), 1, figsize=(9, 3.6 * len(SHOWN)), facecolor=SURFACE
    )
    for ax, (level, disease, area, label) in zip(np.atleast_1d(axes), SHOWN):
        sel = lambda d, *, level=level, disease=disease, area=area: d[
            (d["level"] == level) & (d["disease"] == disease) & (d["area"] == area)
        ].sort_values("month")
        r, b, sc = sel(rolling), sel(base[base["status"] == "tested"]), sel(scored)
        r = r[r["month"] >= pd.Timestamp("2018-01-01")]
        b = b[b["month"] >= pd.Timestamp("2018-01-01")]
        iv = intervals(r["mu"], r["theta"])
        _style(ax)
        ax.fill_between(
            r["month"],
            iv["lo95"],
            iv["hi95"],
            color=BLUE,
            alpha=0.15,
            linewidth=0,
            step="mid",
            label="companion 95% interval",
        )
        ax.plot(
            r["month"],
            r["mu"],
            color=BLUE,
            linewidth=1.1,
            label="companion expectation",
        )
        ax.plot(
            b["month"],
            b["mu"],
            color=GREY,
            linewidth=1.1,
            linestyle="--",
            label="statistical baseline expectation",
        )
        ax.plot(
            r["month"],
            r["y"],
            color=INK,
            linewidth=0.6,
            marker="o",
            markersize=2,
            label="observed outbreaks",
        )
        hit = sc[sc["alert"]]
        ax.scatter(
            hit["month"],
            hit["y"],
            s=22,
            color=ORANGE,
            zorder=5,
            label="companion alert",
        )
        f = fc[
            (fc["level"] == level) & (fc["disease"] == disease) & (fc["area"] == area)
        ]
        if len(f):
            f = f.iloc[0]
            ax.errorbar(
                [f["forecast_month"]],
                [f["expected"]],
                yerr=[[f["expected"] - f["lo95"]], [f["hi95"] - f["expected"]]],
                color=BLUE,
                marker="D",
                markersize=4,
                linewidth=1,
                capsize=2,
                label="next month, 95%",
            )
        ax.set_yscale(
            "function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)), np.square)
        )
        top = max(r["y"].max(), iv["hi95"].quantile(0.97)) * 1.1
        ax.set_ylim(0, top)
        ax.set_yticks(
            [t for t in (0, 10, 50, 100, 250, 500, 1000, 2000, 4000) if t <= top]
        )
        n_c, n_s = int(sc["alert"].sum()), len(sel(stat["baseline"]))
        ax.set_title(
            f"{label}: the statistical baseline flags {n_s} months since 2018, the companion {n_c}",
            loc="left",
            fontsize=10,
            color=INK,
        )
        ax.set_ylabel("outbreaks per month (square-root scale)", fontsize=8, color=INK)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    np.atleast_1d(axes)[0].legend(
        handles, labels, fontsize=7, frameon=False, loc="upper left", ncol=2
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_agreement(agree: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = agree[agree["level"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, facecolor=SURFACE)
    classes = [
        ("n_both", "flagged by both", BLUE),
        ("n_only_companion", "only companion", ORANGE),
        ("n_only_statistical", "only statistical", GREY),
    ]
    for ax, (ref, g) in zip(axes, a.groupby("reference", sort=False)):
        _style(ax)
        x = np.arange(len(PERIODS))
        for k, (col, lab, colr) in enumerate(classes):
            vals = g.set_index("period").loc[list(PERIODS), col]
            ax.bar(x + (k - 1) * 0.27, vals, width=0.25, color=colr, label=lab)
            for xi, v in zip(x + (k - 1) * 0.27, vals):
                ax.text(
                    xi, v, str(int(v)), ha="center", va="bottom", fontsize=7, color=INK
                )
        ax.set_xticks(x, list(PERIODS))
        both, stat_n = g["n_both"].sum(), g["n_statistical"].sum()
        ax.set_title(
            f"{ref} alerts: {both / stat_n:.0%} are also companion alerts; the companion adds few of its own",
            loc="left",
            fontsize=9,
            color=INK,
        )
    axes[0].set_ylabel("flagged months", fontsize=8, color=INK)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---- driver -----------------------------------------------------------------------------------


def stage1(force: bool = False):
    t0 = time.time()
    keys, Y, idx = deep.load_wide()
    Yx, lvl = np.pad(Y, ((0, 0), (0, 1))), level_codes(keys)
    s, mu, th = forward_gbm(Yx, lvl)
    rolling = read_rolling("gbm")
    fc = forecast_table(keys, Y, idx, s, mu, th, rolling)
    fc.to_csv(TABLES / "companion_forecast.csv", index=False)
    print(
        f"[gbm] forecast for {fc['forecast_month'].iloc[0]:%Y-%m}: {len(fc)} series, "
        f"{(fc['last_month_p'] < 0.01).sum()} with a last-month tail p < 0.01 ({time.time() - t0:.0f}s)"
    )

    scored = companion_alerts(rolling, tested_keys())
    stat = statistical_alerts()
    agree = agreement(scored, stat)
    prof = flagged_profile(scored, stat, compare.context())
    al = scored[scored["alert"]].drop(columns=["status", "period"])
    for name, a in stat.items():
        al = al.merge(a[KEY].assign(**{f"also_{name}": True}), how="left", on=KEY)
        al[f"also_{name}"] = al[f"also_{name}"].fillna(False).astype(bool)
    al.to_csv(TABLES / "companion_alerts.csv", index=False)
    agree.to_csv(TABLES / "companion_agreement.csv", index=False)
    prof.to_csv(TABLES / "companion_alert_profile.csv", index=False)
    print(
        f"[gbm] companion alerts {len(al)} (statistical: "
        + ", ".join(f"{k} {len(v)}" for k, v in stat.items())
        + ")"
    )
    print(agree[agree["level"] == "all"].drop(columns="level").to_string(index=False))
    print(prof.round(2).to_string(index=False))

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_series(rolling, scored, fc, stat, FIGURES / "companion_series.png")
    figure_agreement(agree, FIGURES / "companion_agreement.png")

    cov = coverage({"gbm": rolling}, tested_keys())
    cov.to_csv(TABLES / "companion_coverage.csv", index=False)
    c = cov[(cov["expected_bin"] == "all") & (cov["period"] == "2020-2024")].iloc[0]
    print(
        f"[gbm] interval coverage 2020-2024: 80% -> {c['coverage80']:.1%}, 95% -> {c['coverage95']:.1%}"
    )
    chk = refit_check(force)
    g = chk[(chk["model"] == "gbm_refit") & (chk["metric"] == "dlogscore")].set_index(
        "period"
    )
    print(
        "[gbm] refit on the validation block minus benchmark GBM (mean log score): "
        + "; ".join(
            f"{p} {r['value']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
            for p, r in g.iterrows()
        )
    )


def stage2(which: str, force: bool = False) -> None:
    fc_path = TABLES / "companion_forecast.csv"
    if not fc_path.exists():
        stage1(force)
    keys, Y, idx = deep.load_wide()
    Yx, lvl = np.pad(Y, ((0, 0), (0, 1))), level_codes(keys)
    fc = pd.read_csv(fc_path, parse_dates=["last_month", "forecast_month"])
    fc = fc[
        [c for c in fc.columns if not c.endswith(("_lstm", "_ensemble"))]
    ]  # rebuilt below
    tab, verdicts, adopted = stage2_tables(tested_keys())
    tab.to_csv(TABLES / "companion_stage2.csv", index=False)
    for p in PERIODS:
        print(
            f"{p}: "
            + "; ".join(
                f"{m} vs {r}: {_val(tab, p, m, 'dlogscore', r)['value']:+.3f} "
                f"[{_val(tab, p, m, 'dlogscore', r)['lo']:+.3f}, {_val(tab, p, m, 'dlogscore', r)['hi']:+.3f}]"
                for m, r in (("lstm", "gbm"), ("ensemble", "gbm"), ("ensemble", "lstm"))
            )
            + "; sd_z "
            + ", ".join(
                f"{m} {_val(tab, p, m, 'sd_z')['value']:.2f}"
                for m in ("gbm", "lstm", "ensemble")
            )
        )
    gbm, lstm = read_rolling("gbm"), read_rolling("lstm")
    cov = coverage(
        {"gbm": gbm, "lstm": lstm, "ensemble": ensemble_frame(gbm, lstm)}, tested_keys()
    )
    cov.to_csv(TABLES / "companion_coverage.csv", index=False)
    print(
        cov[cov["period"] == "2020-2024"]
        .pivot(
            index="expected_bin", columns="model", values=["coverage80", "coverage95"]
        )
        .round(3)
        .to_string()
    )
    dec = pd.DataFrame([{"model": m, **v} for m, v in verdicts.items()])
    dec["adopted"] = dec["model"] == adopted
    dec.to_csv(TABLES / "companion_adoption.csv", index=False)
    print(dec.to_string(index=False))
    print(f"adopted into the companion: {adopted or 'none (stay with the GBM)'}")
    if not verdicts[which]["passes"]:
        print(
            f"{which} fails the pre-registered rule: not adopted, no forward forecast made"
        )
        return
    t0 = time.time()
    fwd = forward_lstm(Yx, lvl, force)
    n_obs = Y.shape[1]
    assert (
        fwd["s"] == deep.samples(deep.first_nonzero(Y), n_obs, n_obs)[0]
    ).all()  # same rows as the GBM forecast
    fc = add_model_columns(
        fc, "lstm", fwd["mu"], fwd["theta"], read_rolling("lstm"), idx[-1]
    )
    shift = np.median(
        np.log((fwd["mu"] + 1e-3) / (fwd["mu_evaluated_protocol"] + 1e-3))
    )
    print(
        f"[lstm] forward fits (best epochs {fwd['best_epochs']}) in {time.time() - t0:.0f}s; median log ratio of "
        f"refit to evaluated-protocol forecast {shift:+.3f}"
    )
    if adopted == "ensemble":
        gm = np.sqrt(fc["expected"] * fc["expected_lstm"])
        th = 1.0 / (0.5 / fc["theta"] + 0.5 / fc["theta_lstm"])
        fc = add_model_columns(
            fc,
            "ensemble",
            gm.to_numpy(),
            th.to_numpy(),
            ensemble_frame(read_rolling("gbm"), read_rolling("lstm")),
            idx[-1],
        )
    fc.to_csv(fc_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gbm", "lstm", "ensemble"], default="gbm")
    ap.add_argument(
        "--force",
        action="store_true",
        help="refit the cached forward / refit-check models",
    )
    a = ap.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    if a.model == "gbm":
        stage1(a.force)
    else:
        stage2(a.model, a.force)


if __name__ == "__main__":
    main()
