"""Next month: the companion forecast, an expectation given recent activity (not an alert threshold)."""

import pandas as pd
import streamlit as st
from explorer import charts, facts
from explorer.data import (
    LEVEL_LABEL,
    LEVELS,
    area_label,
    read_history,
    read_table,
    series_title,
    short_disease,
)
from explorer.ui import look_at, need_table, select, show_data

forecast = read_table("companion_forecast")
if forecast is None:
    need_table("companion_forecast")

comp = facts.companion_summary()
last, nxt = comp["last_month"], comp["forecast_month"]
st.title(f"Next month: what to expect in {nxt:%B %Y}")
st.warning(
    title="An expectation, not an alert threshold",
    icon=":material/info:",
    body=(
        f"These numbers are the expected count for {nxt:%B %Y} given activity up to {last:%B %Y}. A wave that is "
        f"already under way is expected to continue, so a high expectation does not mean an alert. Intervals for the "
        f"largest series are wide, and the two models can disagree widely: among the {comp['n_large']} series with an "
        f"expected count of 100 or more, one model's forecast can be up to {comp['max_gap_large']:.1f}x the other's."
    ),
)

df = forecast.assign(
    disease_short=forecast["disease"].map(short_disease),
    area_name=[area_label(lv, a) for lv, a in zip(forecast["level"], forecast["area"])],
    key=forecast["level"] + "|" + forecast["disease"] + "|" + forecast["area"],
    gap=lambda d: (
        d[["expected", "expected_lstm"]].max(axis=1)
        / d[["expected", "expected_lstm"]].min(axis=1)
    ),
)
df["gap"] = df["gap"].where(
    df[["expected", "expected_lstm"]].min(axis=1) >= 1
)  # ratios of tiny counts mislead

with st.container(border=True):
    f1, f2 = st.columns([1.3, 1.7])
    with f1:
        levels = st.pills(
            "Level",
            LEVELS,
            selection_mode="multi",
            default=list(LEVELS),
            format_func=LEVEL_LABEL.get,
            key="next_levels",
        )
    with f2:
        query = (
            st.text_input(
                "Search disease or area",
                placeholder="For example: bluetongue, Europe or DEU",
                key="next_query",
            )
            .strip()
            .lower()
        )
view = df[df["level"].isin(levels)]
if query:
    haystack = (
        view["disease"]
        + " "
        + view["disease_short"]
        + " "
        + view["area"]
        + " "
        + view["area_name"]
    ).str.lower()
    view = view[haystack.str.contains(query, regex=False)]
view = view.sort_values("expected_lstm", ascending=False)


def rng(lo, hi):
    return [f"{a:,.0f} to {b:,.0f}" for a, b in zip(lo, hi)]


st.markdown(f"##### Expected outbreaks in {nxt:%B %Y}: {len(view)} of {len(df)} series")
if view.empty:
    st.info("No series match. Clear the search box or select more levels.")
else:
    st.dataframe(
        pd.DataFrame(
            {
                "Disease": view["disease_short"],
                "Area": view["area_name"],
                f"Observed {last:%b %Y}": view["last_observed"],
                "Expected (LSTM)": view["expected_lstm"].round(1),
                "80% range (LSTM)": rng(view["lo80_lstm"], view["hi80_lstm"]),
                "Expected (GBM)": view["expected"].round(1),
                "80% range (GBM)": rng(view["lo80"], view["hi80"]),
                "Model gap": view["gap"],
                f"Surprise p ({last:%b})": view["last_month_p_lstm"],
            }
        ),
        hide_index=True,
        height=420,
        alt=f"Expected outbreaks next month for {len(view)} disease series with intervals from two models",
        column_config={
            f"Observed {last:%b %Y}": st.column_config.NumberColumn(format="localized"),
            "Expected (LSTM)": st.column_config.NumberColumn(
                format="localized",
                help="LSTM: the forecast model adopted "
                "for this table (best score in 2020-24).",
            ),
            "Expected (GBM)": st.column_config.NumberColumn(
                format="localized",
                help="GBM: a second pooled model; its "
                "intervals are too wide for large series.",
            ),
            "Model gap": st.column_config.NumberColumn(
                format="%.1fx",
                help="Larger forecast divided by the smaller one. Shown when both are at least 1.",
            ),
            f"Surprise p ({last:%b})": st.column_config.NumberColumn(
                format="%.3f",
                help=f"Chance of a count at least as high as {last:%B %Y}'s if the model's "
                "expectation for that month had been right. Below 0.05: unusually high.",
            ),
        },
    )
    st.caption(
        f"**Last month's surprise (p-value).** How unusual was {last:%B %Y}? It is the chance of seeing a count at least "
        "that high if the model's expectation for that month had been right. Values near 0 (below 0.05) mean the month "
        "was unusually high; values near 1 mean it was ordinary or low. **Model gap** is the larger of the two "
        "expected counts divided by the smaller. The 80% range holds the count in about 8 months out of 10. "
        "Columns can be sorted by clicking their headers."
    )

    st.markdown("##### One series in detail")
    options = view["key"].tolist()
    labels = dict(
        zip(
            df["key"],
            [
                f"{r.disease_short}, {r.area_name} ({LEVEL_LABEL[r.level].lower()})"
                for r in df.itertuples()
            ],
        )
    )
    pick = select("Show the chart for", options, "next_series", format_func=labels.get)
    row = df[df["key"] == pick].iloc[0]
    history = read_history()
    if history is not None:
        h = history[
            (history["level"] == row["level"])
            & (history["disease"] == row["disease"])
            & (history["area"] == row["area"])
        ]
        h = h[h["month"] > row["last_month"] - pd.DateOffset(months=36)]
    else:
        h = pd.DataFrame({"month": [row["last_month"]], "y": [row["last_observed"]]})
    name = series_title(row["level"], row["disease"], row["area"])
    st.altair_chart(
        charts.forecast_chart(
            h,
            row,
            title=f"{name}: observed and expected outbreaks",
            subtitle=f"Bars: the last 36 months. Dots: expected count for {nxt:%b %Y}; thick line: 80% range; thin "
            "line: 95% range.",
            y_label="Outbreaks notified per month",
        ),
        alt=f"Bar chart of the last 36 months of outbreaks for {name} with the expected count for {nxt:%B %Y} from "
        f"two models: LSTM {row['expected_lstm']:,.0f}, GBM {row['expected']:,.0f}, each with 80% and 95% ranges.",
    )
    look_at(
        "Compare the dots with the recent bars: the expectation follows recent activity. Where the LSTM (orange) "
        "and GBM (grey) disagree or the ranges are very wide, treat the number as a rough order of magnitude. "
        "A range reaching zero means no outbreak is also a plausible outcome."
    )
    if history is None:
        st.caption(
            "Only the last observed month is drawn because the longer history comes from "
            "`data/processed/` (see the Alerts page for the commands that create it)."
        )
    show_data(
        pd.DataFrame(
            {
                "Model": ["LSTM", "GBM"],
                f"Expected {nxt:%b %Y}": [row["expected_lstm"], row["expected"]],
                "80% low": [row["lo80_lstm"], row["lo80"]],
                "80% high": [row["hi80_lstm"], row["hi80"]],
                "95% low": [row["lo95_lstm"], row["lo95"]],
                "95% high": [row["hi95_lstm"], row["hi95"]],
                f"Surprise p ({last:%b})": [
                    row["last_month_p_lstm"],
                    row["last_month_p"],
                ],
            }
        ),
        "next_month_series",
        label="Show the data: forecast numbers for this series",
        alt=f"Forecast numbers for {name} from the LSTM and the GBM",
    )
    full = view.drop(columns=["disease_short", "area_name", "key", "gap"]).assign(
        last_month=view["last_month"].dt.strftime("%Y-%m"),
        forecast_month=view["forecast_month"].dt.strftime("%Y-%m"),
    )
    show_data(
        full,
        "next_month_all_series",
        label="Show the data: every series in the table above, all columns",
        alt="Full forecast table for the series shown above, all columns",
    )
