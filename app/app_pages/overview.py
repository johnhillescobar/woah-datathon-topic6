"""Overview: what the app shows, the six key findings and why raw counts mislead."""

import pandas as pd
import streamlit as st
from explorer import charts, facts
from explorer.data import month_name, read_table
from explorer.ui import fmt_int, look_at, need_table, segmented, show_data

from t6 import config as C

ri = read_table("reporting_intensity")
if ri is None:
    need_table("reporting_intensity")

st.title("Trends in animal disease occurrence")
st.subheader(
    "What is normal, what is unusual and what is changing in WAHIS outbreak notifications"
)

first, last = ri["month"].min(), facts.companion_summary()["last_month"]
extracted = pd.Timestamp(C.EXTRACT_DATE)
st.markdown(
    f"This explorer summarises a statistical analysis of the immediate notifications and follow-up reports that "
    f"countries send to WOAH through WAHIS, from {first:%B %Y} to {last:%B %Y}. For each disease and area it learns "
    f"what is *normal* (a seasonal level based on the previous five years), flags months that are far above it, "
    f"describes long-run trends, gives an expectation for next month and tests whether weather is linked to "
    f"outbreaks. All counts are *notified* outbreaks and events, so they reflect reporting as well as disease. "
    f"The data were extracted on {extracted.day} {extracted:%B %Y}; the newest {C.CENSOR_MONTHS} months are left out because "
    f"late reports are still arriving. The app only displays results that the analysis pipeline already "
    f"produced; nothing is recomputed here."
)

# Key findings, each with its headline number read from the result tables
det, trend = facts.detector(), facts.trend_counts("outbreaks")
gains, wx, comp = facts.model_gains(), facts.weather(), facts.companion_summary()
peaks = facts.peak_months()
rising = [d for d in facts.rising_global("outbreaks")]
peak_text = ", ".join(
    f"{name} in {month_name(peaks[key])}"
    for name, key in (
        ("bluetongue", "Bluetongue virus"),
        ("HPAI", "HPAI (all hosts)"),
        ("African swine fever", "African swine fever virus"),
    )
    if key in peaks
)
series_text = f" across {det['series']} series" if det["series"] else ""
only_asf = (
    " African swine fever is the only global disease clearly growing faster than reporting."
    if rising == ["African swine fever virus"]
    else ""
)
lstm, gbm = gains[("lstm", "2020-2024")][0], gains[("gbm", "2020-2024")][0]
findings = [
    (
        "1. Normal is seasonal and disease-specific",
        f"{C.WINDOW_MONTHS // 12}-year baseline",
        (
            f"Peaks differ by disease: {peak_text}. The baseline is refitted every month on "
            f"the previous {C.WINDOW_MONTHS // 12} years and ignores one-off waves."
        ),
    ),
    (
        "2. Unusually high months are flagged",
        f"{fmt_int(det['alert_months'])} alert months",
        (
            f"grouped into {fmt_int(det['episodes'])} episodes{series_text}. The largest match "
            "well-documented epizootics (bluetongue, bird flu, screwworm, sheep and goat pox)."
        ),
    ),
    (
        "3. Most upward trends are a reporting effect",
        f"{trend['raw']['up']} rising, then {trend['share']['up']}",
        (
            f"Of {trend['tested']} established disease series, {trend['raw']['up']} rise in raw "
            f"counts but only {trend['share']['up']} still rise once growth in reporting is "
            f"removed.{only_asf}"
        ),
    ),
    (
        "4. Forecasts help during waves",
        f"+{lstm:.2f} score gain",
        (
            f"LSTM forecast over the best statistical model, 2020-24 (gradient-boosted model "
            f"+{gbm:.2f}; higher is better). Almost all of the gain comes in wave months, so "
            "forecasts supplement the alerts and do not replace them."
        ),
    ),
    (
        "5. Weather adds very little",
        f"p = {wx['placebo_p_year_effects']:.2f}",
        (
            "Once shared time trends are removed, the whole weather block is not significant. "
            "One exploratory signal remains: wetter months precede more vector-borne outbreaks "
            f"a month later (rate ratio {wx['rr']:.2f} per standard deviation)."
        ),
    ),
    (
        "6. Early warning is only partly supported",
        f"{comp['horizon_months']} month ahead",
        (
            "The methods flag a surge in the month it happens and forecast one month ahead. "
            "Reporting delays in real time, and longer lead times, were not shown to work."
        ),
    ),
]
st.markdown("#### Six key findings")
for row in (findings[:3], findings[3:]):
    for col, (title, value, text) in zip(st.columns(3), row):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.metric(title, value, label_visibility="collapsed")
            st.write(text)

st.info(
    title="How to read this app",
    icon=":material/menu_book:",
    body=(
        "- **Counts are notifications.** A rise can mean more disease, more testing or more reporting, so every chart "
        "says *notified*.\n"
        "- **Colours.** Blue is what was observed. Orange marks alerts and highlights. Grey is context: the expected "
        "level, the 99% upper limit and the 2025-26 holdout period.\n"
        "- **Pages.** *Alerts* shows unusually high months, *Next month* the expected count and its range, *Trends* what "
        "changed over 20 years, *Weather* what was tested and found, *Methods and limits* how far to trust it.\n"
        "- **Every chart has a plain caption** (what to look at) and a *Show the data* table underneath."
    ),
)

st.markdown("#### Why raw counts mislead")
measures = {
    "countries_with_events": "Countries notifying an event per month",
    "events": "New events notified per month",
    "outbreaks": "Outbreaks notified per month",
}
column = segmented(
    "Measure of reporting",
    measures,
    "overview_measure",
    "countries_with_events",
    format_func=measures.get,
)
cp = read_table("reporting_changepoints")
changes = (
    [pd.Timestamp(b) for b in cp.loc[cp["series"] == column, "break"]]
    if cp is not None
    else []
)
st.altair_chart(
    charts.reporting_chart(
        ri,
        column,
        measures[column],
        changes,
        title=measures[column] + ", all diseases",
        subtitle="Orange lines: detected shifts in the level of reporting",
    ),
    alt=f"Line chart of {measures[column].lower()} from {first:%Y} to {last:%Y}, rising with level shifts at "
    + ", ".join(f"{c:%B %Y}" for c in changes),
)
growth = facts.reporting_growth()
e0, e1 = facts.EARLY_YEARS, facts.LATE_YEARS
span = (f"{e0[0]}-{str(e0[1])[-2:]}", f"{e1[0]}-{str(e1[1])[-2:]}")
look_at(
    f"More countries report more often today than in {e0[0]}: between {span[0]} and {span[1]} the average number of "
    f"countries notifying an event per month rose from {growth['countries_with_events']['early']:.1f} to "
    f"{growth['countries_with_events']['late']:.1f} ({growth['countries_with_events']['ratio']:.1f}x), new events "
    f"{growth['events']['ratio']:.1f}x and outbreaks {growth['outbreaks']['ratio']:.1f}x. Counts for almost any "
    f"disease therefore rise over time, which is why the Trends page also shows each disease as a share of all "
    f"notifications. A change in one disease that coincides with an orange line may be reporting, not disease."
)
show_data(
    ri.assign(month=ri["month"].dt.strftime("%Y-%m")).rename(
        columns={
            "month": "Month",
            "events": "Events",
            "outbreaks": "Outbreaks",
            "countries_with_events": "Countries notifying an event",
            "diseases_with_events": "Diseases notified",
        }
    ),
    "reporting_intensity",
    alt="Monthly number of countries, diseases, events and outbreaks notified",
)

st.markdown("#### Data citation")
st.markdown(f"> {C.CITATION}")
st.caption("This citation applies to all outputs derived from the WAHIS data.")
