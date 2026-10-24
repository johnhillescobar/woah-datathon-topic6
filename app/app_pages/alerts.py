"""Alerts: months that are unusually high for a disease and area, and the largest episodes."""

import pandas as pd
import streamlit as st
from explorer import charts
from explorer.data import (
    LEVEL_LABEL,
    LEVELS,
    METRIC_LABEL,
    VARIANT_LABEL,
    C,
    area_label,
    read_table,
    read_tests,
    series_catalog,
    series_title,
    short_disease,
    suffix,
)
from explorer.ui import (
    look_at,
    need_processed,
    need_table,
    segmented,
    select,
    show_data,
)

DEFAULT_DISEASE = "Bluetongue virus"
METRIC_HELP = {
    "outbreaks": "Outbreak records: the intensity of a wave. A few high-volume reporters dominate.",
    "events": "New epidemiological events (an event is a disease occurrence in a country, which "
    "contains many outbreaks). Sparser, and less sensitive to reporting style.",
}
VARIANT_TEXT = {
    "baseline": "**Baseline**: compares each month with a normal, seasonal level learned from the previous five "
    "years. A wave stays flagged for as long as it lasts, so it shows how big and how long a variation was.",
    "ar1": "**Autoregressive**: also takes last month's count into account, so a month is flagged only if it is a "
    "surprise given the month before. Meant for spotting the start of a burst (timing was not tested separately).",
}
DATE = st.column_config.DateColumn(format="MMM YYYY")

st.title("Alerts: unusually high months")
st.markdown(
    "Pick a disease and an area to see its notified count month by month against what is normal for it. "
    "Months far above the normal range are alerts. Below the chart you find the alert episodes of that "
    "series; further down, the largest episodes anywhere."
)

with st.container(border=True):
    r1 = st.columns([1, 1.3, 1.5])
    with r1[0]:
        metric = segmented(
            "Count",
            METRIC_LABEL,
            "alerts_metric",
            "outbreaks",
            format_func=METRIC_LABEL.get,
            help="\n\n".join(
                f"**{METRIC_LABEL[k]}**: {v}" for k, v in METRIC_HELP.items()
            ),
        )
    with r1[1]:
        variant = segmented(
            "Detector",
            VARIANT_LABEL,
            "alerts_variant",
            "baseline",
            format_func=VARIANT_LABEL.get,
        )
    with r1[2]:
        level = segmented(
            "Level", LEVELS, "alerts_level", "global", format_func=LEVEL_LABEL.get
        )
    st.caption(VARIANT_TEXT[variant])
    cat = series_catalog(metric, variant)
    if cat is not None:
        sub = cat[cat["level"] == level]
        diseases = sorted(sub["disease"].unique(), key=short_disease)
        biggest = sub.groupby("disease")["total"].sum().idxmax()
        r2 = st.columns([1.6, 1.3, 1.5])
        with r2[0]:
            disease = select(
                "Disease",
                diseases,
                "alerts_disease",
                format_func=short_disease,
                default=DEFAULT_DISEASE if DEFAULT_DISEASE in diseases else biggest,
                help="Only diseases with enough history at this level are listed. Full WAHIS names "
                "are in the table under the chart.",
            )
        areas = (
            sub[sub["disease"] == disease]
            .sort_values("total", ascending=False)["area"]
            .tolist()
        )
        with r2[1]:
            area = select(
                {"global": "Area", "region": "WOAH region", "country": "Country"}[
                    level
                ],
                areas,
                "alerts_area",
                format_func=lambda a: area_label(level, a),
                disabled=level == "global",
                help="Countries are listed with the most notifications first.",
            )

if cat is None:
    need_processed("The month-by-month chart")
else:
    tests = read_tests(metric, variant)
    full = tests[
        (tests["level"] == level)
        & (tests["disease"] == disease)
        & (tests["area"] == area)
    ]
    tested = full[full["status"] == "tested"]
    years = (int(full["month"].dt.year.min()), int(full["month"].dt.year.max()))
    with r2[2]:
        shown = (
            st.slider("Years shown", *years, years) if years[0] < years[1] else years
        )
    d = full[full["month"].dt.year.between(*shown)]
    n_alerts = int(d["alert"].sum())
    name = series_title(level, disease, area)
    unit = METRIC_LABEL[metric].lower()
    st.altair_chart(
        charts.alert_chart(
            d,
            title=f"{name}: {unit} notified per month",
            subtitle=f"{VARIANT_LABEL[variant]} detector. {n_alerts} alert month{'s' if n_alerts != 1 else ''} "
            f"in the years shown (orange).",
            y_label=f"{METRIC_LABEL[metric]} notified per month",
            holdout_start=pd.Timestamp(C.HOLDOUT_START),
        ),
        alt=f"Bar chart of monthly {unit} for {name}, {shown[0]} to {shown[1]}, with the expected count as a line, "
        f"the 99% upper limit as a band and {n_alerts} alert months marked in orange.",
    )
    look_at(
        "Orange bars with a marker above them are alert months: the count rose above the grey band, the 99% upper "
        "limit built from the previous five years only. Grey shading marks the 2025-26 holdout, when no weather "
        "data exist. Months without a bar had no notifications or could not be tested yet."
    )
    st.caption(
        ":material/info: An alert means the number of *notified* "
        f"{unit} was unusually high for this disease and area. It is not a diagnosis, and it can reflect more "
        "reporting as well as more disease."
    )
    table = d.assign(
        month=d["month"].dt.strftime("%Y-%m"),
        alert=d["alert"].map({True: "Alert", False: ""}),
    )
    table = table[
        ["month", "y", "mu", "upper99", "ratio", "p", "q", "alert", "status", "period"]
    ]
    show_data(
        table.rename(
            columns={
                "month": "Month",
                "y": "Observed",
                "mu": "Expected",
                "upper99": "99% upper limit",
                "ratio": "Observed / expected",
                "p": "Chance of a count this high (p)",
                "q": "FDR-adjusted q",
                "alert": "Alert",
                "status": "Status",
                "period": "Period",
            }
        ),
        "alerts_series",
        alt=f"Monthly observed and expected {unit} for {name}",
        column_config={
            "Expected": st.column_config.NumberColumn(format="%.1f"),
            "Observed / expected": st.column_config.NumberColumn(format="%.1f"),
            "Chance of a count this high (p)": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "FDR-adjusted q": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.markdown(f"##### Alert episodes for {name}")
    episodes = read_table(f"episodes_{metric}{suffix(variant)}")
    if episodes is not None:
        episodes = episodes.round({"excess": 0})
    if episodes is None:
        need_table(f"episodes_{metric}{suffix(variant)}")
    mine = episodes[
        (episodes["level"] == level)
        & (episodes["disease"] == disease)
        & (episodes["area"] == area)
    ]
    if mine.empty:
        st.caption("No alert episodes for this series.")
    else:
        st.dataframe(
            mine.sort_values("start")[
                [
                    "start",
                    "end",
                    "n_alert_months",
                    "peak_y",
                    "peak_ratio",
                    "excess",
                    "min_q",
                ]
            ].rename(
                columns={
                    "start": "Start",
                    "end": "End",
                    "n_alert_months": "Alert months",
                    "peak_y": "Peak monthly count",
                    "peak_ratio": "Peak / expected",
                    "excess": "Count above expected",
                    "min_q": "Smallest q",
                }
            ),
            hide_index=True,
            alt=f"Alert episodes of {name}",
            column_config={
                "Start": DATE,
                "End": DATE,
                "Peak monthly count": st.column_config.NumberColumn(format="localized"),
                "Peak / expected": st.column_config.NumberColumn(format="%.1f"),
                "Count above expected": st.column_config.NumberColumn(
                    format="localized",
                    help="Total notified count above the expected count over the alert months",
                ),
                "Smallest q": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(
            "An episode is a run of alert months with gaps of at most one month. *Peak / expected* is the "
            "largest ratio of the observed to the expected count in the episode."
        )
    notes = full[full["status"] != "tested"].sort_values("month")
    if len(notes):
        st.markdown("##### First occurrences and re-emergences")
        kinds = {
            "first_occurrence": "First notification in the data",
            "reemergence": "Re-appeared after a long quiet spell",
        }
        st.dataframe(
            notes.assign(status=notes["status"].map(kinds))[
                ["month", "status", "y"]
            ].rename(
                columns={
                    "month": "Month",
                    "status": "What happened",
                    "y": f"{METRIC_LABEL[metric]} that month",
                }
            ),
            hide_index=True,
            alt=f"First occurrences and re-emergences of {name}",
            column_config={
                "Month": DATE,
                f"{METRIC_LABEL[metric]} that month": st.column_config.NumberColumn(
                    format="localized"
                ),
            },
        )
        st.caption(
            "These months have no baseline to compare with, so they are listed here instead of being "
            "tested. They are not statistical alerts, but a new or returning disease is worth a look."
        )

st.divider()
st.header("Largest episodes")
episodes = read_table(f"episodes_{metric}{suffix(variant)}")
if episodes is not None:
    episodes = episodes.round({"excess": 0})
if episodes is None:
    need_table(f"episodes_{metric}{suffix(variant)}")
st.markdown(
    f"The 15 alert episodes with the biggest count above expected, for **{METRIC_LABEL[metric].lower()}** "
    f"and the **{VARIANT_LABEL[variant].lower()}** detector chosen above."
)
with st.container(border=True):
    f1, f2 = st.columns([1.3, 1.7])
    with f1:
        levels = st.pills(
            "Level",
            LEVELS,
            selection_mode="multi",
            default=list(LEVELS),
            format_func=LEVEL_LABEL.get,
            key="episodes_levels",
        )
    y_lo, y_hi = (
        int(episodes["start"].dt.year.min()),
        int(episodes["start"].dt.year.max()),
    )
    with f2:
        span = (
            st.slider("Episode started in", y_lo, y_hi, (y_lo, y_hi))
            if y_lo < y_hi
            else (y_lo, y_hi)
        )
top = episodes[
    episodes["level"].isin(levels) & episodes["start"].dt.year.between(*span)
].nlargest(15, "excess")
if top.empty:
    st.info(
        "No episodes match this selection. Choose at least one level and a wider range of years."
    )
else:
    top = top.assign(
        disease_full=top["disease"],
        area_name=[area_label(lv, a) for lv, a in zip(top["level"], top["area"])],
    )
    top["label"] = [
        f"{series_title(r.level, r.disease, r.area)}, {r.start:%b %Y}"
        + (f" to {r.end:%b %Y}" if r.end > r.start else "")
        for r in top.itertuples()
    ]
    st.altair_chart(
        charts.episodes_chart(
            top,
            title=f"The {len(top)} largest alert episodes",
            subtitle=f"{METRIC_LABEL[metric]}, started {span[0]}-{span[1]}",
        ),
        alt=f"Horizontal bar chart of the {len(top)} largest alert episodes by count above expected; the largest is "
        f"{top['label'].iloc[0]}.",
    )
    look_at(
        "Each bar is one episode: the total number of notified "
        f"{METRIC_LABEL[metric].lower()} above the expected count while the area was in alert. The biggest ones "
        "match documented epizootics. Large single-country bars are often one wave seen again in the region "
        "and the global series."
    )
    show_data(
        top[
            [
                "disease_full",
                "area_name",
                "level",
                "start",
                "end",
                "n_alert_months",
                "peak_y",
                "peak_ratio",
                "excess",
            ]
        ]
        .assign(
            start=top["start"].dt.strftime("%Y-%m"), end=top["end"].dt.strftime("%Y-%m")
        )
        .rename(
            columns={
                "disease_full": "Disease",
                "area_name": "Area",
                "level": "Level",
                "start": "Start",
                "end": "End",
                "n_alert_months": "Alert months",
                "peak_y": "Peak monthly count",
                "peak_ratio": "Peak / expected",
                "excess": "Count above expected",
            }
        ),
        "largest_episodes",
        alt="The largest alert episodes with disease, area, dates and excess count",
        column_config={
            "Peak / expected": st.column_config.NumberColumn(format="%.1f"),
            "Count above expected": st.column_config.NumberColumn(format="localized"),
        },
    )
