"""Run the baseline over every eligible series, control the false discovery rate, summarise.

Run with `uv run python -m t6.detect [--metrics outbreaks events] [--models baseline ar1]`.
Large per-test tables go to data/processed/, compact result tables to reports/tables/.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

from t6 import baseline, series
from t6 import config as C

TABLES = C.REPORTS / "tables"


def _spec(w: np.ndarray, length: int) -> tuple[int, bool]:
    """Model complexity grows with the information in the window.

    Non-zero months matter as much as the total: 21 outbreaks in two months cannot support a
    trend and a seasonal cycle (the fit separates and explodes).
    """
    total, active = w.sum(), int((w > 0).sum())
    if total < 20 or active < 8:
        n_harm = 0
    else:
        n_harm = 1 if total < 100 or active < 24 or length < 36 else 2
    return n_harm, bool(total >= 20 and active >= 12 and length >= 36)


def run_one(task):
    key, y, metric, model = task
    y = np.asarray(y, float)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return []
    cal = (np.arange(len(y)) + 0) % 12 + 1  # series starts in January 2005
    lagged = np.r_[0.0, np.log1p(y[:-1])]  # lagged[t] = log1p(count of month t-1)
    rows = []
    for t in range(nz[0], len(y)):
        prev = nz[nz < t]
        if len(prev) == 0:
            if y[t] > 0:
                rows.append(
                    (*key, t, y[t], np.nan, np.nan, np.nan, np.nan, "first_occurrence")
                )
            continue
        start = max(t - C.WINDOW_MONTHS, prev[0])
        length = t - start
        if length < C.MIN_WINDOW:
            continue
        w = y[start:t]
        if w.sum() < C.MIN_WINDOW_SUM[metric]:
            if y[t] > 0 and t - prev[-1] >= C.MIN_WINDOW:
                rows.append(
                    (*key, t, y[t], np.nan, np.nan, np.nan, np.nan, "reemergence")
                )
            continue
        n_harm, trend = _spec(w, length)
        if (
            model == "ar1" and trend
        ):  # persistence term only where the window can support it
            s = max(start, 1)
            tt = (np.arange(s, t) - t) / 12.0
            p = baseline.fit_predict(
                y[s:t],
                tt,
                cal[s:t],
                0.0,
                cal[t],
                n_harm,
                trend,
                lag=lagged[s:t],
                lag_next=lagged[t],
            )
        else:
            tt = (np.arange(start, t) - t) / 12.0
            # "legacy" (no season screen) exists only so simulate.py can show what the screen fixes
            p = baseline.fit_predict(
                w,
                tt,
                cal[start:t],
                0.0,
                cal[t],
                n_harm,
                trend,
                screen=model != "legacy",
            )
        rows.append((*key, t, y[t], p.mu, p.phi, p.se_eta, p.var, "tested"))
    return rows


def run_metric(metric: str, model: str) -> pd.DataFrame:
    ob, ev = series.load_tables()
    wide = series.monthly_counts(ob, ev, metric)
    idx = series.months()
    print(
        f"[{metric}/{model}] {len(wide):,} eligible series "
        f"({wide.index.get_level_values('level').value_counts().to_dict()}), "
        f"{idx[0]:%Y-%m} .. {idx[-1]:%Y-%m}"
    )
    tasks = [
        ((*k, metric), row, metric, model)
        for k, row in zip(wide.index, wide.to_numpy())
    ]
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(run_one, tasks, chunksize=8))
    cols = [
        "level",
        "disease",
        "area",
        "metric",
        "t",
        "y",
        "mu",
        "phi",
        "se_eta",
        "var",
        "status",
    ]
    df = pd.DataFrame([r for chunk in res for r in chunk], columns=cols)
    df["month"] = idx[df["t"].to_numpy()]
    return df.drop(columns="t")


def score(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = df.copy()
    tested = df["status"] == "tested"
    df["p"] = np.nan
    df["upper99"] = np.nan
    df.loc[tested, "p"] = baseline.upper_p(
        df.loc[tested, "y"], df.loc[tested, "mu"], df.loc[tested, "var"]
    )
    df.loc[tested, "upper99"] = baseline.upper_quantile(
        0.99, df.loc[tested, "mu"], df.loc[tested, "var"]
    )
    df["q"] = np.nan
    df.loc[tested, "q"] = multipletests(df.loc[tested, "p"], method="fdr_bh")[1]
    df["ratio"] = df["y"] / df["mu"]
    df["alert"] = (
        tested
        & (df["q"] < C.FDR_Q)
        & (df["y"] >= C.MIN_ALERT_COUNT[metric])
        & (df["ratio"] >= C.MIN_ALERT_RATIO)
    )
    df["z"] = np.nan  # Dunn-Smyth residual, used for time-series diagnostics
    df.loc[tested, "z"] = baseline.dunn_smyth(
        df.loc[tested, "y"], df.loc[tested, "mu"], df.loc[tested, "var"]
    )
    df["period"] = np.where(
        df["month"] >= pd.Timestamp(C.HOLDOUT_START), "holdout", "retrospective"
    )
    return df


def episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive alert months (gap of at most one month) form one episode."""
    a = df[df["alert"]].sort_values(["level", "disease", "area", "month"]).copy()
    if a.empty:
        return a
    key = ["level", "disease", "area"]
    step = a.groupby(key)["month"].diff()
    new_episode = step.isna() | (step.dt.days > 62)
    a["episode"] = new_episode.groupby([a[k] for k in key]).cumsum()
    a["excess"] = a["y"] - a["mu"]
    out = a.groupby(key + ["episode"]).agg(
        start=("month", "min"),
        end=("month", "max"),
        n_alert_months=("month", "size"),
        peak_y=("y", "max"),
        peak_ratio=("ratio", "max"),
        min_q=("q", "min"),
        excess=("excess", "sum"),
    )
    return out.reset_index().drop(columns="episode")


def diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-series residual checks: bias, lag-1 autocorrelation, Ljung-Box(12), tail frequencies."""
    rows = []
    t = df[df["status"] == "tested"].sort_values("month")
    for (lvl, dis, area), d in t.groupby(["level", "disease", "area"]):
        z = d["z"].to_numpy()
        if len(z) < 36:
            continue
        lb = acorr_ljungbox(z, lags=[12], return_df=True)["lb_pvalue"].iloc[0]
        rows.append(
            {
                "level": lvl,
                "disease": dis,
                "area": area,
                "n_tested": len(z),
                "mean_z": z.mean(),
                "sd_z": z.std(),
                "acf1": pd.Series(z).autocorr(1),
                "ljung_box_p12": lb,
                "frac_p05": (d["p"] < 0.05).mean(),
                "frac_p01": (d["p"] < 0.01).mean(),
                "n_alerts": int(d["alert"].sum()),
            }
        )
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    t = df[df["status"] == "tested"]
    rows = []
    for period, d in t.groupby("period"):
        rows.append(
            {
                "metric": metric,
                "period": period,
                "n_tests": len(d),
                "frac_p<0.05": (d["p"] < 0.05).mean(),
                "frac_p<0.01": (d["p"] < 0.01).mean(),
                "mean_z": d["z"].mean(),
                "sd_z": d["z"].std(),
                "n_alerts": int(d["alert"].sum()),
                "alert_rate": d["alert"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main(metrics: list[str], models: list[str]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    for model in models:
        sfx = (
            "" if model == "baseline" else f"_{model}"
        )  # baseline keeps the original file names
        cal = []
        for metric in metrics:
            df = score(run_metric(metric, model), metric)
            df.to_parquet(C.PROCESSED / f"tests_{metric}{sfx}.parquet")
            df[df["alert"]].drop(columns=["status"]).to_csv(
                TABLES / f"alerts_{metric}{sfx}.csv", index=False
            )
            df[df["status"] != "tested"].to_csv(
                TABLES / f"emergence_{metric}{sfx}.csv", index=False
            )
            episodes(df).to_csv(TABLES / f"episodes_{metric}{sfx}.csv", index=False)
            diag = diagnostics(df)
            diag.to_csv(TABLES / f"series_diagnostics_{metric}{sfx}.csv", index=False)
            cal.append(calibration(df, metric))
            print(
                f"[{metric}/{model}] tests={int((df.status == 'tested').sum()):,} "
                f"alerts={int(df.alert.sum()):,} | series with Ljung-Box p<0.05: "
                f"{(diag.ljung_box_p12 < 0.05).mean():.0%} of {len(diag)}"
            )
        cal = pd.concat(cal)
        cal.insert(0, "model", model)
        cal.to_csv(TABLES / f"calibration{sfx}.csv", index=False)
        print(cal.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", default=["outbreaks", "events"])
    ap.add_argument(
        "--models", nargs="+", default=["baseline", "ar1"], choices=["baseline", "ar1"]
    )
    a = ap.parse_args()
    main(a.metrics, a.models)
