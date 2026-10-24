"""Long-run trends, seasonality and structural breaks in notified disease occurrence (2005-2024).

Per global / WOAH-region series:
  * robust STL decomposition of log1p monthly counts -> trend and seasonal strength, peak month;
  * Mann-Kendall (Kendall tau) + Theil-Sen slope on annual totals, FDR-adjusted across series,
    both on raw counts and on the disease's *share* of all notifications in the same area, which
    removes the general growth in reporting;
  * PELT change points on the de-seasonalised series (min segment 12 months).
The same tools are applied to the reporting-intensity series (all diseases pooled) to expose
changes in how much is reported, which is not a change in disease.

Run with `uv run python -m t6.trends`. Tables -> reports/tables, figures -> reports/figures.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.seasonal import STL

from t6 import config as C
from t6 import series

TABLES, FIGS = C.REPORTS / "tables", C.REPORTS / "figures"
MIN_ACTIVE_YEARS = 10  # years with at least one notification before a long-run trend is tested
PELT_PENALTY = 3.0     # x log(n) x variance of the STL remainder (BIC-style)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def trend_window(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.loc[:, wide.columns <= pd.Timestamp(C.END)]


def stl_parts(y: np.ndarray):
    r = STL(np.log1p(y), period=12, robust=True).fit()
    return r.trend, r.seasonal, r.resid


def strength(resid: np.ndarray, comp: np.ndarray) -> float:
    return float(max(0.0, 1 - np.var(resid) / np.var(comp + resid)))


def breakpoints(x: np.ndarray, noise_var: float) -> list[int]:
    """PELT mean-shift change points; penalty scaled by the variance of the STL remainder.

    The remainder variance is used instead of month-to-month differences because outbreak waves
    persist, and differences understate the noise (that gave ~12 breaks per series).
    """
    if noise_var <= 0:
        return []
    algo = rpt.Pelt(model="l2", min_size=12, jump=1).fit(x.reshape(-1, 1))
    return [b for b in algo.predict(pen=PELT_PENALTY * np.log(len(x)) * noise_var) if b < len(x)]


def sen_trend(years: np.ndarray, values: np.ndarray) -> dict:
    tau, p = stats.kendalltau(years, values)
    slope, _, lo, hi = stats.theilslopes(values, years, 0.95)
    scale = max(np.median(values), 1e-9)
    return {"tau": tau, "p": p, "pct_per_year": 100 * slope / scale,
            "lo_pct": 100 * lo / scale, "hi_pct": 100 * hi / scale}


def year_totals(ob: pd.DataFrame, ev: pd.DataFrame, metric: str) -> dict:
    """Disease-pooled annual notifications per area, the denominator for reporting-adjusted trends."""
    df = {"outbreaks": ob, "events": ev}[metric]
    df = df[df["start_month"] <= pd.Timestamp(C.END)]
    yr = df["start_month"].dt.year
    out = {("global", "all"): df.groupby(yr).size()}
    for region, g in df.groupby("region"):
        out[("region", region)] = g.groupby(g["start_month"].dt.year).size()
    return out


def analyse(wide: pd.DataFrame, metric: str, denoms: dict):
    w = trend_window(wide)
    years = np.arange(w.columns[0].year, w.columns[-1].year + 1)
    annual = w.T.groupby(w.columns.year).sum().T
    ann_rows, stl_rows, cp_rows, parts = [], [], [], {}
    for key, row in zip(w.index, w.to_numpy()):
        level, disease, area = key
        tot = annual.loc[key].to_numpy(float)
        active = int((tot > 0).sum())
        rec = {"level": level, "disease": disease, "area": area, "active_years": active,
               "total": int(tot.sum())}
        if active >= MIN_ACTIVE_YEARS and tot[:10].sum() > 0:  # established diseases only
            raw = sen_trend(years, tot)
            rec.update({f"raw_{k}": v for k, v in raw.items()})
            den = denoms[(level, "all" if level == "global" else area)].reindex(years, fill_value=0).to_numpy(float)
            share = sen_trend(years, tot / np.maximum(den, 1))
            rec.update({f"share_{k}": v for k, v in share.items()})
        ann_rows.append(rec)
        if row.sum() >= C.MIN_SERIES_TOTAL[metric] and active >= 5:
            trend, seas, resid = stl_parts(row)
            peak = int(np.argmax(pd.Series(seas).groupby(w.columns.month).mean().to_numpy())) + 1
            stl_rows.append({"level": level, "disease": disease, "area": area,
                             "trend_strength": strength(resid, trend),
                             "seasonal_strength": strength(resid, seas), "peak_month": peak})
            deseason = trend + resid
            bps = breakpoints(deseason, np.var(resid))
            for b in bps:
                cp_rows.append({"level": level, "disease": disease, "area": area,
                                "break": w.columns[b].strftime("%Y-%m"),
                                "before": float(np.expm1(np.mean(deseason[max(0, b - 12):b]))),
                                "after": float(np.expm1(np.mean(deseason[b:b + 12])))})
            parts[key] = (trend, seas, resid, [w.columns[b] for b in bps])
    ann = pd.DataFrame(ann_rows)
    for kind in ("raw", "share"):
        ok = ann[f"{kind}_p"].notna()
        ann.loc[ok, f"{kind}_q"] = multipletests(ann.loc[ok, f"{kind}_p"], method="fdr_bh")[1]
        ann[f"{kind}_direction"] = np.select(
            [ann[f"{kind}_q"] < C.FDR_Q, ann[f"{kind}_p"].notna()],
            [np.where(ann[f"{kind}_tau"] > 0, "increasing", "decreasing"), "no trend"], "not tested")
    return ann, pd.DataFrame(stl_rows), pd.DataFrame(cp_rows), parts, w.columns


def reporting_intensity(ob: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "events": ev.groupby("start_month").size(),
        "outbreaks": ob.groupby("start_month").size(),
        "countries_with_events": ev.groupby("start_month")["iso_code"].nunique(),
        "diseases_with_events": ev.groupby("start_month")["disease"].nunique(),
    }).reindex(series.months(), fill_value=0)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_reporting(f: pd.DataFrame, path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True, facecolor=SURFACE)
    spec = [("countries_with_events", "Countries notifying an event (per month)"),
            ("events", "New events notified (per month)"),
            ("outbreaks", "Outbreaks started (per month, log scale)")]
    for ax, (col, title) in zip(axes, spec):
        ax.plot(f.index, f[col].rolling(3, center=True, min_periods=1).mean(), color=BLUE, lw=1.6)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        style(ax)
    axes[2].set_yscale("symlog")
    fig.suptitle("Reporting volume grows several-fold: part of any trend is reporting, not disease",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "3-month moving average. Source: WAHIS INFUR extract (WOAH).", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_stl(key, parts, months, y, path):
    trend, seas, resid, cps = parts
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True, facecolor=SURFACE)
    axes[0].plot(months, np.log1p(y), color=MUTED, lw=0.9)
    axes[0].plot(months, trend, color=BLUE, lw=2)
    for cp in cps:
        for ax in axes:
            ax.axvline(cp, color=ORANGE, lw=1, ls="--")
    axes[0].set_title("Observed (grey) and trend (blue), log scale; dashed = detected level shift",
                      loc="left", fontsize=10, color=INK)
    axes[1].plot(months, seas, color=BLUE, lw=1.4)
    axes[1].set_title("Seasonal pattern", loc="left", fontsize=10, color=INK)
    axes[2].plot(months, resid, color=MUTED, lw=0.9)
    axes[2].set_title("What is left (waves and noise)", loc="left", fontsize=10, color=INK)
    for ax in axes:
        style(ax)
    where = "global" if key[0] == "global" else key[2]
    fig.suptitle(f"{key[1]} - {where}: monthly outbreaks, 2005-2024", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_slopes(ann: pd.DataFrame, path, metric: str):
    """Two panels on the same diseases: raw growth, then growth beyond overall reporting growth."""
    d = ann[(ann["level"] == "global") & ann["raw_p"].notna()].nlargest(22, "total")
    d = d.sort_values("raw_pct_per_year")
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.3 * len(d) + 2.2), sharey=True, facecolor=SURFACE)
    y = np.arange(len(d))
    for ax, kind, title in ((axes[0], "raw", "Raw annual counts"),
                            (axes[1], "share", "Share of all notifications")):
        direction = d[f"{kind}_direction"]
        colour = np.where(direction == "increasing", ORANGE, np.where(direction == "decreasing", BLUE, MUTED))
        lo, hi, mid = (d[f"{kind}_{c}"].clip(-100, 300) for c in ("lo_pct", "hi_pct", "pct_per_year"))
        ax.hlines(y, lo, hi, color=GRID, lw=3)
        ax.scatter(mid, y, c=colour, s=34, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("change per year, % of the median (bar: 95% interval)", fontsize=8, color=MUTED)
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [s[:40] for s in d["disease"]], fontsize=8, color=INK)
    fig.suptitle(f"Which diseases are changing? Annual {metric}, global, 2005-2024 "
                 "(orange: rising, blue: falling, grey: no clear trend)", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    ob, ev = series.load_tables()

    f = reporting_intensity(ob, ev)
    f.to_csv(TABLES / "reporting_intensity.csv", index_label="month")
    fig_reporting(f, FIGS / "reporting_intensity.png")
    cps = []
    for col in ("countries_with_events", "events", "outbreaks"):
        x = f[col].to_numpy(float)
        trend, seas, resid = stl_parts(x)
        cps += [(col, f.index[i].strftime("%Y-%m")) for i in breakpoints(trend + resid, np.var(resid))]
    pd.DataFrame(cps, columns=["series", "break"]).to_csv(TABLES / "reporting_changepoints.csv", index=False)

    for metric in ("outbreaks", "events"):
        wide = series.monthly_counts(ob, ev, metric)
        wide = wide[wide.index.get_level_values("level") != "country"]
        ann, stl_df, cp, parts, months = analyse(wide, metric, year_totals(ob, ev, metric))
        ann.to_csv(TABLES / f"trends_annual_{metric}.csv", index=False)
        stl_df.to_csv(TABLES / f"trends_stl_{metric}.csv", index=False)
        cp.to_csv(TABLES / f"changepoints_{metric}.csv", index=False)
        fig_slopes(ann, FIGS / f"trend_slopes_{metric}.png", metric)
        print(f"[{metric}] series={len(wide)} tested-for-trend={int(ann['raw_p'].notna().sum())}")
        print("   raw  :", ann["raw_direction"].value_counts().to_dict())
        print("   share:", ann["share_direction"].value_counts().to_dict())
        print(f"   change points per series: median={cp.groupby(['level', 'disease', 'area']).size().median():.0f}")
        if metric == "outbreaks":
            top = (ann[ann["level"] == "global"].sort_values("total", ascending=False)
                   .head(6)[["level", "disease", "area"]].itertuples(index=False, name=None))
            for key in top:
                if key in parts:
                    y = wide.loc[key].to_numpy(float)[: len(parts[key][0])]
                    name = "".join(c if c.isalnum() else "_" for c in key[1])[:30]
                    fig_stl(key, parts[key], months, y, FIGS / f"stl_{name}.png")
    print("reporting-intensity change points:", cps)


if __name__ == "__main__":
    main()
