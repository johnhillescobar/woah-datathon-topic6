"""Compare one-step-ahead predictive distributions of every model on the rows they all score.

Models are discovered from data/processed/tests_outbreaks*.parquet (no suffix = `baseline`, otherwise
the suffix: ar1, lstm, gbm, ...). Only status == 'tested' rows from 2018 on are scored, on the
intersection of all models. Metrics are reported per period, level, volume bin, history length and for
surge rows; differences to a reference model come with paired bootstrap CIs that resample whole series.
For each statistical model a `_recal` copy (same means, one NB theta per level fitted on the previous
year) is scored as a diagnostic: it shows whether a gain is just wider predictive tails. The
pre-registered decision rule is applied in `decide`.

Run with `uv run python -m t6.compare`. Writes reports/tables/model_comparison.csv (long format).
"""
import re

import numpy as np
import pandas as pd
from scipy import special, stats

from t6 import baseline, series
from t6 import config as C
from t6.deep import theta_mle

TABLES = C.REPORTS / "tables"
KEY = ["level", "disease", "area", "month"]
STAT = ("baseline", "ar1")  # statistical models
DIAG = "_recal"  # diagnostic copy of a statistical model with its dispersion refitted (not a contender)
PERIODS = {"2018-2019": ("2018-01-01", "2019-12-31"), "2020-2024": ("2020-01-01", "2024-12-31"),
           "2025-2026": ("2025-01-01", "2026-12-31")}
VOLUME = [("<0.5", 0, 0.5), ("0.5-2", 0.5, 2), ("2-10", 2, 10), (">10", 10, np.inf)]  # baseline mu
HISTORY = [("<36m", 0, 36), ("36-71m", 36, 72), (">=72m", 72, np.inf)]  # months since first count
SURGE_RATIO, SURGE_MIN = 3.0, 3  # y / baseline mu >= 3 and y >= 3
SURGE = "y/mu>=3 & y>=3"
FLOOR = -20.0  # sensitivity: per-row log score floored here
B = 2000
SD_RANGE = (0.9, 1.3)
# Needs that favour a transparent parametric model; the pooled models cannot meet the first (judgement)
EXPLAINS_FLAGS = {"lstm": False, "gbm": False}


# ---- scoring ----------------------------------------------------------------------------------

def log_score(y, mu, var):
    """NB log predictive probability of the observed count (higher is better).

    scipy's logpmf saturates to -inf when prob rounds to 1 (Poisson limit with a tiny mean); those
    rows use the same pmf written in terms of mu and size.
    """
    y = np.asarray(y, float)
    size, prob = baseline.nb_params(mu, var)
    ls = stats.nbinom.logpmf(y, size, prob)
    bad = ~np.isfinite(ls)
    if bad.any():
        r, m = size[bad], np.asarray(mu, float)[bad]
        den = np.logaddexp(np.log(r), np.log(m))
        ls[bad] = (special.gammaln(y[bad] + r) - special.gammaln(r) - special.gammaln(y[bad] + 1)
                   + r * (np.log(r) - den) + y[bad] * (np.log(m) - den))
    return ls


def boot_mean(d, g, n_boot: int = B, seed: int = 0):
    """Mean of d with a 95% percentile CI from resampling whole groups g (series or diseases)."""
    d = np.asarray(d, float)
    _, inv = np.unique(g, return_inverse=True)
    sums, n = np.bincount(inv, d), np.bincount(inv)
    w = np.random.default_rng(seed).multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=n_boot)
    m = (w @ sums) / (w @ n)
    return float(d.mean()), *map(float, np.percentile(m, [2.5, 97.5]))


def win_rate(d, g) -> float:
    """Share of groups whose mean difference is positive."""
    _, inv = np.unique(g, return_inverse=True)
    return float((np.bincount(inv, d) / np.bincount(inv) > 0).mean())


def mean_acf1(z, g, min_n: int = 8) -> float:
    """Lag-1 autocorrelation of residuals within each series (rows ordered by month), averaged."""
    z, g = np.asarray(z), np.asarray(g)
    cuts = np.flatnonzero(np.diff(g)) + 1
    acs = [np.corrcoef(a[:-1], a[1:])[0, 1] for a in np.split(z, cuts) if len(a) >= min_n]
    return float(np.nanmean(acs)) if acs else np.nan


# ---- data -------------------------------------------------------------------------------------

def discover() -> dict:
    out = {}
    for f in sorted(C.PROCESSED.glob("tests_outbreaks*.parquet")):
        name = re.sub(r"^tests_outbreaks_?", "", f.stem) or "baseline"
        out[name] = f
    return out


def context() -> pd.DataFrame:
    """Per series and month: months since the first count and mean count over the previous 12 months."""
    ob, ev = series.load_tables()
    wide = series.monthly_counts(ob, ev, "outbreaks")
    Y, idx = wide.to_numpy(), series.months()
    t = np.arange(len(idx))
    cs = np.pad(np.cumsum(Y, 1), ((0, 0), (1, 0)))
    prev12 = (cs[:, t] - cs[:, np.maximum(t - 12, 0)]) / np.maximum(np.minimum(t, 12), 1)
    out = wide.index.repeat(len(idx)).to_frame(index=False)
    out["month"] = np.tile(idx, len(wide)).astype("datetime64[ns]")
    out["age"] = (t[None, :] - (Y > 0).argmax(1)[:, None]).ravel()
    out["prev12"] = prev12.ravel()
    out["lag1"] = np.pad(Y[:, :-1], ((0, 0), (1, 0))).ravel()
    return out


def read_models(files: dict) -> dict:
    """Tested rows of every model (from 2017, so the diagnostics can fit dispersion on the year before)."""
    out = {}
    for name, f in files.items():
        d = pd.read_parquet(f)
        d = d[(d["status"] == "tested") & (d["month"] >= pd.Timestamp("2017-01-01"))].copy()
        d["month"] = pd.to_datetime(d["month"]).astype("datetime64[ns]")
        out[name] = d
    return out


def recalibrate(d: pd.DataFrame) -> pd.DataFrame:
    """Same means, but var = mu + mu^2/theta with one theta per level fitted on the previous year's rows
    (out of sample): separates 'better mean' from 'wider predictive tails'."""
    parts = []
    for year in sorted(d["month"].dt.year.unique())[1:]:
        prev, cur = d[d["month"].dt.year == year - 1], d[d["month"].dt.year == year].copy()
        th = {l: theta_mle(g["y"].to_numpy(), g["mu"].to_numpy()) for l, g in prev.groupby("level")}
        cur["theta"] = cur["level"].map(th)
        cur["var"] = cur["mu"] + cur["mu"] ** 2 / cur["theta"]
        parts.append(cur.dropna(subset=["theta"]))
    return pd.concat(parts)


def load(frames: dict, start: str = "2018-01-01") -> pd.DataFrame:
    """Inner join of every model's rows on KEY; columns y, mu_<m>, var_<m> (+ lstm seed columns)."""
    frames = dict(frames)
    frames.update({f"{m}{DIAG}": recalibrate(frames[m]) for m in STAT if m in frames})
    df = None
    for name, d in frames.items():
        d = d[d["month"] >= pd.Timestamp(start)]
        extra = [c for c in d.columns if re.match(r"(mu|theta)_seed\d", c)]
        d = d[KEY + ["y", "mu", "var"] + extra].rename(
            columns={"y": f"y_{name}", "mu": f"mu_{name}", "var": f"var_{name}",
                     **{c: f"{c}_{name}" for c in extra}})
        df = d if df is None else df.merge(d, on=KEY, how="inner")
    ys = [c for c in df.columns if c.startswith("y_")]
    if not all(np.allclose(df[ys[0]], df[c]) for c in ys[1:]):
        raise ValueError("models disagree on the observed counts")
    df = df.rename(columns={ys[0]: "y"}).drop(columns=ys[1:])
    return df.sort_values(["level", "disease", "area", "month"]).reset_index(drop=True)


def groups(models: list[str]):
    """(statistical models, dispersion-recalibrated diagnostics, learned models)."""
    stat = [m for m in models if m in STAT]
    diag = [m for m in models if m.endswith(DIAG)]
    return stat, diag, [m for m in models if m not in stat + diag]


def prepare(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["sid"] = df.groupby(["level", "disease", "area"]).ngroup()
    df["did"] = df.groupby("disease").ngroup()
    df = df.merge(context(), on=KEY, how="left", validate="one_to_one")
    df["bmu"] = df["mu_baseline"]
    df["ratio"] = df["y"] / df["bmu"]
    for m in models:
        df[f"ls_{m}"] = log_score(df["y"], df[f"mu_{m}"], df[f"var_{m}"])
        df[f"z_{m}"] = baseline.dunn_smyth(df["y"], df[f"mu_{m}"], df[f"var_{m}"], seed=0)
        df[f"p_{m}"] = baseline.upper_p(df["y"], df[f"mu_{m}"], df[f"var_{m}"])
    return df


def strata(df: pd.DataFrame):
    """(kind, name, mask) for the overall sample and every stratification."""
    yield "all", "all", np.ones(len(df), bool)
    for lvl in ("global", "region", "country"):
        yield "level", lvl, (df["level"] == lvl).to_numpy()
    for name, lo, hi in VOLUME:
        yield "volume", name, ((df["bmu"] >= lo) & (df["bmu"] < hi)).to_numpy()
    for name, lo, hi in HISTORY:
        yield "history", name, ((df["age"] >= lo) & (df["age"] < hi)).to_numpy()
    surge = ((df["ratio"] >= SURGE_RATIO) & (df["y"] >= SURGE_MIN)).to_numpy()
    yield "surge", SURGE, surge
    yield "surge", "other rows", ~surge
    # model-free surge: at least 3x the mean of the previous 12 months (floor 1)
    free = ((df["y"] >= SURGE_MIN) & (df["y"] >= 3 * np.maximum(df["prev12"], 1))).to_numpy()
    yield "surge_past", "y>=3 & y>=3x last-12m mean", free
    # jump: at least 3x last month (floor 1), the onset rows where persistence gives no head start
    jump = ((df["y"] >= SURGE_MIN) & (df["y"] >= 3 * np.maximum(df["lag1"], 1))).to_numpy()
    yield "surge_jump", "y>=3 & y>=3x last month", jump


# ---- tables -----------------------------------------------------------------------------------

def _row(period, kind, stratum, model, metric, value, n, ref=None, ref_model=None, lo=np.nan, hi=np.nan):
    return {"period": period, "kind": kind, "stratum": stratum, "model": model, "ref": ref,
            "ref_model": ref_model, "metric": metric, "value": value, "lo": lo, "hi": hi, "n": n}


def metric_rows(df, models, mask, period, kind, name):
    sub = df[mask]
    n = len(sub)
    rows = [_row(period, kind, name, "all_models", "n_series", sub["sid"].nunique(), n)]
    for m in models:
        ls, z, p = sub[f"ls_{m}"].to_numpy(), sub[f"z_{m}"].to_numpy(), sub[f"p_{m}"].to_numpy()
        err = sub[f"mu_{m}"].to_numpy() - sub["y"].to_numpy()
        for metric, v in (("mean_logscore", ls.mean()), ("median_logscore", np.median(ls)),
                          ("mean_logscore_floor20", np.maximum(ls, FLOOR).mean()),
                          ("mae", np.abs(err).mean()), ("rmse", np.sqrt((err**2).mean())),
                          ("mean_z", z.mean()), ("sd_z", z.std()), ("share_p01", (p < 0.01).mean()),
                          ("acf1", mean_acf1(z, sub["sid"].to_numpy()))):
            rows.append(_row(period, kind, name, m, metric, float(v), n))
    return rows


def diff_rows(df, models, refs, mask, period, kind, name):
    """Gain of each model over each reference: mean log-score difference with series / disease
    bootstrap CIs, win rate over series, and the floored-score version."""
    sub = df[mask]
    rows, sid, did = [], sub["sid"].to_numpy(), sub["did"].to_numpy()
    for m in models:
        for ref_label, ref_model in refs.items():
            if ref_model == m:
                continue
            d = (sub[f"ls_{m}"] - sub[f"ls_{ref_model}"]).to_numpy()
            df_ = (np.maximum(sub[f"ls_{m}"], FLOOR) - np.maximum(sub[f"ls_{ref_model}"], FLOOR)).to_numpy()
            n = len(d)
            args = dict(ref=ref_label, ref_model=ref_model)
            v, lo, hi = boot_mean(d, sid)
            rows.append(_row(period, kind, name, m, "dlogscore", v, n, lo=lo, hi=hi, **args))
            v, lo, hi = boot_mean(d, did)
            rows.append(_row(period, kind, name, m, "dlogscore_disease_boot", v, n, lo=lo, hi=hi, **args))
            v, lo, hi = boot_mean(df_, sid)
            rows.append(_row(period, kind, name, m, "dlogscore_floor20", v, n, lo=lo, hi=hi, **args))
            rows.append(_row(period, kind, name, m, "win_rate", win_rate(d, sid), n, **args))
    return rows


def seed_rows(df, period, mask, models):
    """Single-seed LSTM mean log scores, to show the seed-to-seed spread."""
    rows = []
    for m in models:
        for k in range(10):
            if f"mu_seed{k}_{m}" not in df:
                break
            mu, th = df[f"mu_seed{k}_{m}"], df[f"theta_seed{k}_{m}"]
            ls = log_score(df["y"], mu, mu + mu**2 / th)
            rows.append(_row(period, "seed_check", "all", f"{m}_seed{k}", "mean_logscore",
                             float(ls[mask].mean()), int(mask.sum())))
    return rows


def compare(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    stat, diag, learned = groups(models)
    for period, (a, b) in PERIODS.items():
        in_period = ((df["month"] >= a) & (df["month"] <= b)).to_numpy()
        if not in_period.any():
            continue
        best = max(stat, key=lambda m: df.loc[in_period, f"ls_{m}"].mean())
        refs_ml = {r: r for r in stat + diag + learned} | {"best_stat": best}
        others = [m for m in stat + diag if m != "baseline"]
        for kind, name, mask in strata(df):
            mask = mask & in_period
            if not mask.any():
                continue
            rows += metric_rows(df, models, mask, period, kind, name)
            rows += diff_rows(df, learned, refs_ml, mask, period, kind, name)
            rows += diff_rows(df, others, {"baseline": "baseline"}, mask, period, kind, name)
        rows += seed_rows(df, period, in_period, learned)
    return pd.DataFrame(rows)


# ---- decision rule ----------------------------------------------------------------------------

def _get(res, period, kind, stratum, model, metric, ref=None):
    q = res[(res["period"] == period) & (res["kind"] == kind) & (res["stratum"] == stratum)
            & (res["model"] == model) & (res["metric"] == metric)]
    q = q[q["ref"].isna()] if ref is None else q[q["ref"] == ref]
    return q.iloc[0] if len(q) else None


def decide(res: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Pre-registered rule. Supplement: CI of gain over the best statistical model excludes 0 (above) in
    2020-2024, gain >= 0 in 2025-2026, residual sd in 0.9-1.3 in both. Alternative: also gain >= 0 on surge
    rows and in the mu<2 strata (2020-2024), gain >= 0 on short histories, and explains why a month is flagged."""
    out = []
    for m in groups(models)[2]:
        g = lambda per, kind, st: _get(res, per, kind, st, m, "dlogscore", "best_stat")
        main, hold = g("2020-2024", "all", "all"), g("2025-2026", "all", "all")
        sd = [_get(res, p, "all", "all", m, "sd_z") for p in ("2020-2024", "2025-2026")]
        c = {"gain_2020_24_ci_above_0": main is not None and main["lo"] > 0,
             "gain_2025_26_not_negative": hold is not None and hold["value"] >= 0,
             "sd_z_within_0.9_1.3": all(s is not None and SD_RANGE[0] <= s["value"] <= SD_RANGE[1] for s in sd)}
        supplement = all(c.values())
        extra = {"surge_gain_not_negative": g("2020-2024", "surge", SURGE),
                 "sparse_lt0.5_gain_not_negative": g("2020-2024", "volume", "<0.5"),
                 "sparse_0.5_2_gain_not_negative": g("2020-2024", "volume", "0.5-2"),
                 "short_history_gain_not_negative": g("2020-2024", "history", "<36m")}
        for k, r in extra.items():
            c[k] = r is not None and r["value"] >= 0
        c["explains_why_flagged"] = EXPLAINS_FLAGS.get(m, False)
        alternative = supplement and all(c.values())
        out.append({"model": m, **c, "supplement": supplement, "alternative": alternative,
                    "verdict": "alternative" if alternative else "supplement" if supplement else "drop"})
    return pd.DataFrame(out)


# ---- printing ---------------------------------------------------------------------------------

def _fmt(res, period, kind, stratum, model, metric, ref=None, ci=False):
    r = _get(res, period, kind, stratum, model, metric, ref)
    if r is None:
        return "n/a"
    return f"{r['value']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]" if ci else f"{r['value']:.3f}"


def summary(res: pd.DataFrame, models: list[str]) -> None:
    ml = groups(models)[2]
    for period in PERIODS:
        n = res[(res["period"] == period) & (res["kind"] == "all") & (res["metric"] == "n_series")]
        if n.empty:
            continue
        print(f"\n== {period}: {int(n['n'].iloc[0]):,} rows, {int(n['value'].iloc[0])} series")
        print(f"{'model':9s}{'logscore':>9s}{'median':>8s}{'floor20':>9s}{'MAE':>8s}{'RMSE':>9s}"
              f"{'sd_z':>7s}{'p<.01':>7s}{'acf1':>7s}{'surgeLS':>9s}   gain vs baseline [95% CI]   win   vs best_stat")
        for m in models:
            f = lambda metric, kind="all", st="all": _fmt(res, period, kind, st, m, metric)
            gain = ("" if m == "baseline" else _fmt(res, period, "all", "all", m, "dlogscore", "baseline", True))
            win = ("" if m == "baseline" else _fmt(res, period, "all", "all", m, "win_rate", "baseline"))
            bs = (_fmt(res, period, "all", "all", m, "dlogscore", "best_stat", True) if m in ml else "")
            print(f"{m:9s}{f('mean_logscore'):>9s}{f('median_logscore'):>8s}{f('mean_logscore_floor20'):>9s}"
                  f"{f('mae'):>8s}{f('rmse'):>9s}{f('sd_z'):>7s}{f('share_p01'):>7s}{f('acf1'):>7s}"
                  f"{f('mean_logscore', 'surge', SURGE):>9s}   {gain:27s}{win:>6s}   {bs}")
    for period in ("2020-2024", "2025-2026"):
        print(f"\n== Gain over baseline by stratum, {period} (mean log score difference [95% CI])")
        for m in [m for m in models if m != "baseline"]:
            for kind in ("level", "volume", "history", "surge", "surge_past", "surge_jump"):
                parts = [f"{st}: {_fmt(res, period, kind, st, m, 'dlogscore', 'baseline', True)}"
                         for st in res[(res["period"] == period) & (res["kind"] == kind)]["stratum"].unique()]
                print(f"  {m:6s} {kind:10s} " + " | ".join(parts))
    diag = groups(models)[1]
    if diag:
        print("\n== Learned models against the dispersion-recalibrated statistical models, all rows")
        for period in PERIODS:
            print(f"  {period}: " + "; ".join(f"{m} vs {r}: {_fmt(res, period, 'all', 'all', m, 'dlogscore', r, True)}"
                                              for m in ml for r in diag))
    if len(ml) > 1:
        print("\n== Learned models against each other, all rows (mean log score difference [95% CI])")
        for period in PERIODS:
            print(f"  {period}: " + "; ".join(f"{m} vs {r}: {_fmt(res, period, 'all', 'all', m, 'dlogscore', r, True)}"
                                              for m in ml for r in ml if r < m))
    seeds = res[res["kind"] == "seed_check"]
    if len(seeds):
        print("\n== Single-seed LSTM mean log score (the tables above use the 3-seed mean)")
        print(seeds.pivot(index="model", columns="period", values="value").round(3).to_string())
    for lab, col in (("disease-cluster bootstrap", "dlogscore_disease_boot"),
                     ("log score floored at -20", "dlogscore_floor20")):
        print(f"\n== Sensitivity: gain over best statistical model, {lab}")
        for period in PERIODS:
            print(f"  {period}: " + "; ".join(f"{m}: {_fmt(res, period, 'all', 'all', m, col, 'best_stat', True)}"
                                              for m in ml))


def main() -> None:
    files = discover()
    frames = read_models(files)
    names = list(files) + [f"{m}{DIAG}" for m in STAT if m in files]
    models = sorted(names, key=lambda m: (m not in STAT, m.endswith(DIAG) is False, m != "baseline", m))
    print("models:", ", ".join(models))
    df = prepare(load(frames), models)
    print(f"scored on {len(df):,} common rows, {df['sid'].nunique()} series, "
          f"{df['month'].min():%Y-%m}..{df['month'].max():%Y-%m}")
    res = compare(df, models)
    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "model_comparison.csv", index=False)
    summary(res, models)
    verdict = decide(res, models)
    verdict.to_csv(TABLES / "model_decision.csv", index=False)
    print("\n== Decision rule\n" + (verdict.T.to_string(header=False) if len(verdict) else "no learned models found"))


if __name__ == "__main__":
    main()
