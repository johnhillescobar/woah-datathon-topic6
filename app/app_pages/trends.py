"""Trends: raw counts versus share of all notifications, trend verdicts, seasonal peaks and STL figures."""

import numpy as np
import pandas as pd
import streamlit as st
from explorer import charts, facts
from explorer.data import (
    METRIC_LABEL,
    area_label,
    month_name,
    read_table,
    short_disease,
    stl_figure,
)
from explorer.ui import look_at, need_table, segmented, select, show_data

VERDICT = {
    "increasing": "Rising",
    "decreasing": "Falling",
    "no trend": "No clear trend",
    "not tested": "Not tested",
}
TREND_LEVELS = {
    "all": "Global and regional",
    "global": "Global",
    "region": "WOAH region",
}
SORTS = {"size": "Largest first", "raw": "Raw change", "share": "Change in share"}
SHOW = {
    "all": "All tested series",
    "raw_up": "Rising in raw counts",
    "share_up": "Rising as a share",
    "share_down": "Falling as a share",
}

st.title("Trends: what changed over 20 years")
st.markdown(
    "Raw counts of notifications grow with the amount countries report. To separate real change from "
    "reporting, every disease is shown twice: as raw annual counts and as its **share of all notifications** "
    "in the same area. A trend is called *clear* when it passes a statistical test at a 5% false discovery "
    "rate (q-value below 0.05)."
)

with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        metric = segmented(
            "Count",
            METRIC_LABEL,
            "trends_metric",
            "outbreaks",
            format_func=METRIC_LABEL.get,
            help="Outbreaks measure intensity; events (new occurrences in a country) are less "
            "affected by how a few countries report.",
        )
    with c2:
        level = segmented(
            "Level",
            TREND_LEVELS,
            "trends_level",
            "all",
            format_func=TREND_LEVELS.get,
        )

annual = read_table(f"trends_annual_{metric}")
if annual is None:
    need_table(f"trends_annual_{metric}")
levels = ["global", "region"] if level == "all" else [level]
scope = TREND_LEVELS[level].lower()
counts = facts.trend_counts(metric, None if level == "all" else level)
unit = METRIC_LABEL[metric].lower()

m = st.columns(4)
m[0].metric(
    "Established series tested",
    counts["tested"],
    border=True,
    height="stretch",
    help="Diseases active in at least 10 of the 20 years and present before 2015.",
)
m[1].metric("Rising in raw counts", counts["raw"]["up"], border=True, height="stretch")
m[2].metric(
    "Rising as a share of notifications",
    counts["share"]["up"],
    border=True,
    height="stretch",
)
m[3].metric(
    "Falling as a share of notifications",
    counts["share"]["down"],
    border=True,
    height="stretch",
)
st.markdown(
    f"Of the {counts['tested']} established {scope} {unit} series, **{counts['raw']['up']} rise in raw annual "
    f"counts, but only {counts['share']['up']} rise once reporting growth is removed** "
    f"({counts['share']['down']} fall as a share)."
)

t = annual[
    annual["level"].isin(levels) & (annual["raw_direction"] != "not tested")
].copy()
t["area_name"] = [area_label(lv, a) for lv, a in zip(t["level"], t["area"])]
t["label"] = [
    short_disease(d)
    + (f" ({a})" if lv == "region" else (" (global)" if level == "all" else ""))
    for d, lv, a in zip(t["disease"], t["level"], t["area_name"])
]

st.markdown("##### Change per year, raw versus share of all notifications")
s1, s2 = st.columns([1, 1])
with s1:
    sort = segmented("Order", SORTS, "trends_sort", "size", format_func=SORTS.get)
with s2:
    top_n = st.slider(
        "Number of series shown",
        5,
        min(40, len(t)),
        min(20, len(t)),
        key=f"trends_n_{level}_{metric}",
    )
key = {"size": "total", "raw": "raw_pct_per_year", "share": "share_pct_per_year"}[sort]
shown = (
    t.sort_values(key, ascending=False).head(top_n)
    if sort != "size"
    else t.nlargest(top_n, "total")
)
parts = []
for kind, name in (
    ("raw", "Raw annual counts"),
    ("share", "Share of all notifications"),
):
    parts.append(
        pd.DataFrame(
            {
                "label": shown["label"],
                "full": shown["disease"],
                "area_name": shown["area_name"],
                "measure": name,
                "pct": shown[f"{kind}_pct_per_year"],
                "lo": shown[f"{kind}_lo_pct"],
                "hi": shown[f"{kind}_hi_pct"],
                "q": shown[f"{kind}_q"],
                "trend": np.where(
                    shown[f"{kind}_direction"] == "no trend",
                    "No clear trend",
                    "Clear trend",
                ),
            }
        )
    )
long = pd.concat(parts, ignore_index=True)
st.altair_chart(
    charts.trend_chart(
        long,
        title=f"Change per year of {unit} notified, {scope} series",
        subtitle="Blue: raw annual counts. Orange: share of all notifications in the area. "
        "Filled dot: clear trend (q below 0.05).",
    ),
    alt=f"Dot chart of the yearly change in {unit} for the {top_n} series shown, raw and as a share of all "
    f"notifications. {counts['raw']['up']} series rise in raw counts and {counts['share']['up']} as a share.",
)
look_at(
    "Where a blue dot sits right of zero but the orange dot does not, the disease grew with reporting and not "
    "faster than it. Only filled dots are clear trends. Bars are 95% intervals and are cut at the chart edge when "
    "very long (the full interval is in the tooltip). The share measure is dominated by a few huge diseases "
    "(African swine fever, bird flu), so shares of other diseases fall partly by construction; events are the "
    "sturdier measure."
)
show_data(
    long.rename(
        columns={
            "label": "Series",
            "full": "Disease",
            "area_name": "Area",
            "measure": "Measure",
            "pct": "Change per year, %",
            "lo": "95% interval from, %",
            "hi": "95% interval to, %",
            "q": "q-value",
            "trend": "Verdict",
        }
    ),
    "trend_slopes",
    alt="Yearly percentage change with 95% interval for each series and measure",
)

st.markdown("##### All tested series")
choice = segmented("Show", SHOW, "trends_show", "all", format_func=SHOW.get)
rows = {
    "all": t,
    "raw_up": t[t["raw_direction"] == "increasing"],
    "share_up": t[t["share_direction"] == "increasing"],
    "share_down": t[t["share_direction"] == "decreasing"],
}[choice]
table = pd.DataFrame(
    {
        "Disease": rows["disease"].map(short_disease),
        "Area": rows["area_name"],
        "Years active": rows["active_years"],
        f"{METRIC_LABEL[metric]} 2005-2024": rows["total"],
        "Raw change per year, %": rows["raw_pct_per_year"],
        "Raw verdict": rows["raw_direction"].map(VERDICT),
        "Raw q-value": rows["raw_q"],
        "Share change per year, %": rows["share_pct_per_year"],
        "Share verdict": rows["share_direction"].map(VERDICT),
        "Share q-value": rows["share_q"],
    }
).sort_values(f"{METRIC_LABEL[metric]} 2005-2024", ascending=False)
st.dataframe(
    table,
    hide_index=True,
    height=min(420, 38 * (len(table) + 1) + 4),
    alt=f"Trend verdicts for {len(table)} {scope} series",
    column_config={
        f"{METRIC_LABEL[metric]} 2005-2024": st.column_config.NumberColumn(
            format="localized"
        ),
        "Raw change per year, %": st.column_config.NumberColumn(format="%+.1f"),
        "Share change per year, %": st.column_config.NumberColumn(format="%+.1f"),
        "Raw q-value": st.column_config.NumberColumn(format="%.4f"),
        "Share q-value": st.column_config.NumberColumn(format="%.4f"),
    },
)
st.caption(
    "Change per year is the Theil-Sen slope as a percentage of the series' median annual value. The q-value "
    "is the chance of seeing such a trend by luck, adjusted for testing many series; below 0.05 the "
    "trend is called clear. Annual series are short (20 values), so treat slopes as descriptive. Click a "
    "column header to sort."
)

st.divider()
st.header("Seasons and waves")
st.markdown(
    "**Trend, season and what is left (global outbreaks).** The pipeline splits each monthly series into a "
    "slow trend, a repeating yearly pattern and what is left over (waves and noise)."
)
pool = annual[
    (annual["level"] == "global") & (annual["raw_direction"] != "not tested")
].sort_values("total", ascending=False)
with_fig = [d for d in pool["disease"] if stl_figure(d)]
if with_fig:
    left, right = st.columns([3, 2])
    with right:
        disease = select(
            "Disease", with_fig, "trends_stl_disease", format_func=short_disease
        )
        st.caption(
            "Top panel: observed monthly counts (grey) and the trend (blue) on a log scale; dashed orange lines "
            "mark detected level shifts. Middle: the repeating yearly pattern. Bottom: what is left. "
            "Reporting in general changed level near 2006, 2016, 2020 and 2024, so a shift near those dates "
            "may be reporting rather than disease."
        )
    with left:
        st.image(
            stl_figure(disease),
            alt=f"Decomposition of monthly global outbreaks of {short_disease(disease)} "
            "into trend, seasonal pattern and remainder",
            width="stretch",
        )
else:
    st.caption(
        "No STL figures found in `reports/figures/`. Run `uv run python -m t6.trends` to create them."
    )

stl = read_table(f"trends_stl_{metric}")
if stl is not None:
    st.markdown(f"##### Seasonal peak month and strength ({unit}, {scope} series)")
    s = stl[stl["level"].isin(levels)].merge(
        annual[["level", "disease", "area", "total"]],
        on=["level", "disease", "area"],
        how="left",
    )
    total_col = f"{METRIC_LABEL[metric]} 2005-2024"
    seasonal = pd.DataFrame(
        {
            "Disease": s["disease"].map(short_disease),
            "Area": [area_label(lv, a) for lv, a in zip(s["level"], s["area"])],
            total_col: s["total"],
            "Peak month": s["peak_month"].astype(int).map(month_name),
            "Seasonal strength": s["seasonal_strength"],
            "Trend strength": s["trend_strength"],
        }
    ).sort_values(total_col, ascending=False)
    st.dataframe(
        seasonal,
        hide_index=True,
        height=min(420, 38 * (len(seasonal) + 1) + 4),
        alt=f"Seasonal peak month and strength for {len(seasonal)} {scope} series",
        column_config={
            total_col: st.column_config.NumberColumn(format="localized"),
            "Seasonal strength": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.2f"
            ),
            "Trend strength": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.2f"
            ),
        },
    )
    st.caption(
        "Strength is the share of the month-to-month variation explained by that component: 0 means none and "
        "1 means all of it. The peak month is only meaningful where the seasonal strength is clearly above "
        "zero (rare diseases can show a strong pattern from only a handful of notifications). Largest series first; "
        "click a header to sort differently."
    )
