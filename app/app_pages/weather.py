"""Weather: what was tested, the verdict, the fitted curves, the coefficient table and the out-of-sample check."""

import pandas as pd
import streamlit as st
from explorer import charts, facts
from explorer.data import read_table
from explorer.ui import look_at, need_table, segmented, show_data

VARIABLES = {
    "prec_z": "Rainfall",
    "tmin_anom": "Minimum temperature",
    "tmax_anom": "Maximum temperature",
}
GROUPS = {"vector": "Vector-borne diseases", "other": "Other diseases"}
ADJUSTMENTS = {
    "year_effects": "With year effects (recommended)",
    "no_year_effects": "Without year effects",
}
EXPERIMENT_LABELS = {
    "nb_pooled - ar1": (
        "Counts-only regression vs best statistical detector",
        "counts",
    ),
    "nb_pooled - gbm": ("Counts-only regression vs gradient-boosted model", "counts"),
    "nb_climate - nb_pooled": ("Add weather terms: all series", "weather"),
    "nb_climate - nb_pooled [vector-borne rows]": (
        "Add weather terms: vector-borne rows",
        "weather",
    ),
    "nb_climate_vector - nb_pooled": (
        "Weather terms for vector-borne diseases: all series",
        "weather",
    ),
    "nb_climate_vector - nb_pooled [vector-borne rows]": (
        "Weather terms for vector-borne diseases: vector-borne rows",
        "weather",
    ),
    "nb_gam - nb_pooled": ("Smooth weather curves (GAM): all series", "weather"),
    "nb_gam - nb_pooled [vector-borne rows]": (
        "Smooth weather curves (GAM): vector-borne rows",
        "weather",
    ),
    "nb_precip_vector - nb_pooled": (
        "Rainfall only, vector-borne (exploratory): all series",
        "weather",
    ),
    "nb_precip_vector - nb_pooled [vector-borne rows]": (
        "Rainfall only, vector-borne (exploratory): vector-borne rows",
        "weather",
    ),
}
Y_DOMAIN = (0.25, 4.0)

curves = read_table("gam_association_curves")
coefs = read_table("climate_association")
experiments = read_table("nb_experiments")
for name, table in (
    ("gam_association_curves", curves),
    ("climate_association", coefs),
    ("nb_experiments", experiments),
):
    if table is None:
        need_table(name)
wx = facts.weather()

st.title("Weather: what was tested and what was found")
st.info(
    title="Verdict: weather adds very little",
    icon=":material/thunderstorm:",
    body=(
        "Past counts carry almost all of the predictive signal. Adding weather does not improve forecasts of next "
        "month's count, and the overall link between weather and outbreaks disappears once shared time trends are "
        f"removed (placebo p = {wx['placebo_p_year_effects']:.2f}). One exploratory pattern remains: for "
        f"vector-borne diseases, wetter-than-usual months are followed one month later by more outbreaks "
        f"(rate ratio {wx['rr']:.2f} per standard deviation). It was found after looking at the results, so it is "
        "a hypothesis to confirm, not an early-warning tool."
    ),
)
st.markdown(
    f"**What was tested.** For {wx['n_series']} country-level series, monthly weather *anomalies* (how far a month "
    "was from the usual weather of that place and calendar month in 1991-2020: lowest temperature, highest "
    "temperature and rainfall) one to three months earlier were compared with the number of outbreaks notified. "
    "Two questions were asked separately. Does weather **help predict** next month's count out of sample? Is "
    "there an **association** in the data once shared trends (warming and reporting both rise over time) are "
    "removed? Association is tested against a *placebo*: whole years of weather are shuffled within each country, "
    "which keeps the seasons but breaks the link with disease."
)

st.header("Rate-ratio curves")
st.markdown(
    "A smooth curve was fitted for each anomaly and lag. It shows how the expected number of outbreaks changes "
    "when the weather one, two or three months earlier was unusually dry or wet (or cold or warm)."
)
with st.container(border=True):
    v1, v2 = st.columns(2)
    with v1:
        variable = segmented(
            "Weather variable",
            VARIABLES,
            "wx_variable",
            "prec_z",
            format_func=VARIABLES.get,
        )
    with v2:
        group = segmented(
            "Disease group",
            GROUPS,
            "wx_group",
            "vector",
            format_func=GROUPS.get,
            help="Vector-borne: bluetongue, West Nile fever, lumpy skin disease, Rift Valley fever, "
            "epizootic haemorrhagic disease, African horse sickness and Schmallenberg.",
        )
sel = curves[(curves["variable"] == variable) & (curves["group"] == group)]
x_title = f"{VARIABLES[variable]} anomaly (standard deviations; 0 = usual)"
for col, lag in zip(st.columns(3), (1, 2, 3)):
    part = sel[sel["lag"] == lag]
    with col:
        st.altair_chart(
            charts.curve_chart(
                part,
                x_title=x_title,
                y_domain=Y_DOMAIN,
                title=f"{lag} month{'s' if lag > 1 else ''} earlier",
            ),
            alt=f"Rate ratio of outbreak counts against {VARIABLES[variable].lower()} anomaly {lag} months earlier "
            f"for {GROUPS[group].lower()}, with a 95% band that includes 1 over most of the range.",
        )
look_at(
    "A rate ratio of 1 means no change in outbreak counts. The blue line is the estimated ratio and the shaded band "
    "the 95% range: where the band covers 1 (grey line), the data do not show an effect. Bands widen at the edges, "
    "where few months are unusually wet, dry, cold or hot, and are cut at the chart edge. The curves are smooth "
    "and close to straight, so there are no threshold effects."
)
show_data(
    sel.assign(
        group=sel["group"].map(GROUPS), variable=sel["variable"].map(VARIABLES)
    ).rename(
        columns={
            "variable": "Weather variable",
            "lag": "Months earlier",
            "group": "Disease group",
            "anomaly_sd": "Anomaly (SD)",
            "irr": "Rate ratio",
            "lo": "95% band from",
            "hi": "95% band to",
        }
    ),
    "gam_curves",
    alt="Rate ratio and 95% band by weather anomaly and lag",
)

st.header("Coefficients and placebo tests")
adjustment = segmented(
    "Adjustment",
    ADJUSTMENTS,
    "wx_adjustment",
    "year_effects",
    format_func=ADJUSTMENTS.get,
    help="Year effects remove trends shared by all countries, such as warming and rising "
    "reporting. They are the fair comparison.",
)
c = coefs[coefs["adjustment"] == adjustment].copy()
verdict = pd.Series("No clear association", index=c.index)
verdict[c["placebo_p"] < 0.05] = "Passes the single-term placebo test only"
verdict[c["placebo_p_familywise"] < 0.05] = "Survives the strict placebo test"
table = pd.DataFrame(
    {
        "Weather variable": c["variable_lag"]
        .str.rsplit("_lag", n=1)
        .str[0]
        .map(VARIABLES),
        "Months earlier": c["variable_lag"].str.rsplit("_lag", n=1).str[1].astype(int),
        "Disease group": c["disease_group"].map(
            {"vector": "Vector-borne", "other": "Other"}
        ),
        "Rate ratio per SD": c["irr_per_sd"],
        "95% interval": [f"{a:.2f} to {b:.2f}" for a, b in zip(c["ci_lo"], c["ci_hi"])],
        "Naive p": c["p"],
        "Placebo p": c["placebo_p"],
        "Placebo p (strict)": c["placebo_p_familywise"],
        "Verdict": verdict,
    }
).sort_values(["Placebo p (strict)", "Placebo p"])
st.dataframe(
    table,
    hide_index=True,
    height=420,
    alt=f"Weather coefficients with placebo p-values, {ADJUSTMENTS[adjustment].lower()}",
    column_config={
        "Rate ratio per SD": st.column_config.NumberColumn(format="%.2f"),
        "Naive p": st.column_config.NumberColumn(
            format="%.4f",
            help="Standard test; much too confident here, so it is not relied on.",
        ),
        "Placebo p": st.column_config.NumberColumn(
            format="%.4f",
            help="Share of 300 shuffled-weather runs that "
            "gave a result at least this strong.",
        ),
        "Placebo p (strict)": st.column_config.NumberColumn(
            format="%.4f",
            help="Compares each term with the strongest of all 18 terms in every shuffle, so "
            "it allows for having looked at 18 terms.",
        ),
    },
)
st.caption(
    f"**How to read it.** The rate ratio is the change in expected outbreaks for a weather anomaly one standard "
    f"deviation above usual (above 1: more outbreaks, below 1: fewer). The *naive p* is shown for completeness but "
    f"is over-confident; judge the terms by the placebo columns (p below 0.05 is the usual cut-off). With year "
    f"effects the whole weather block has placebo p = {wx['placebo_p_year_effects']:.2f}; without them it looks "
    f"significant ({wx['placebo_p_no_year_effects']:.3f}) only because weather and reporting both trend upwards. "
    f"Only the 1-month rainfall term for vector-borne diseases survives the strict test (placebo p "
    f"{wx['rr_placebo_p']:.3f}, strict {wx['rr_placebo_fw']:.3f})."
)

st.header("Does weather improve forecasts?")
g = experiments[
    (experiments["period"] == "2020-2024") & (experiments["kind"] == "log_score_gain")
]
g = g[g["comparison"].isin(EXPERIMENT_LABELS)].copy()
g["label"] = g["comparison"].map(lambda k: EXPERIMENT_LABELS[k][0])
g["panel"] = g["comparison"].map(lambda k: EXPERIMENT_LABELS[k][1])
g = g.rename(
    columns={"mean_log_score": "gain", "ci_lo": "lo", "ci_hi": "hi", "n_rows": "rows"}
)
g["verdict"] = ((g["lo"] > 0) | (g["hi"] < 0)).map(
    {True: "Interval excludes zero", False: "Interval includes zero"}
)
st.markdown(
    "Each model predicted the next month for 2020-2024 using only earlier data. The **gain** is how much better "
    "one model's average log score is than the comparison (positive is better). Log score rewards putting "
    "probability on the count that actually happened."
)
a = g[g["panel"] == "counts"]
b = g[g["panel"] == "weather"]
st.altair_chart(
    charts.gain_chart(
        a,
        x_title="Gain in mean log score per month (higher is better)",
        title="Counts alone compared with other models",
        subtitle="Large differences: past counts matter.",
    ),
    alt="Forest chart: the counts-only regression gains about one log-score point over the best statistical "
    "detector and is slightly below the gradient-boosted model.",
)
st.altair_chart(
    charts.gain_chart(
        b,
        x_title="Gain in mean log score per month (higher is better; note the much smaller scale)",
        title="What adding weather changes",
        subtitle="Tiny differences: all intervals sit within a few thousandths of zero.",
    ),
    alt="Forest chart: adding weather terms changes the log score by less than 0.01 per month in every variant; "
    "only the exploratory rainfall-only model has an interval above zero.",
)
look_at(
    "Dots show the gain and bars the 95% interval. In the lower panel every dot sits within about 0.01 of zero: "
    "adding weather does not help (the smooth-curve model is slightly worse). Only the exploratory rainfall-only "
    f"model for vector-borne diseases has an interval above zero, and its gain is about +{wx['precip_gain']:.3f} "
    "per month, far too small to matter in practice."
)
show_data(
    g[["label", "gain", "lo", "hi", "rows", "verdict"]].rename(
        columns={
            "label": "Comparison",
            "gain": "Gain in mean log score",
            "lo": "95% interval from",
            "hi": "95% interval to",
            "rows": "Rows scored",
            "verdict": "Interval",
        }
    ),
    "nb_experiments",
    alt="Gain in log score with 95% interval for each model comparison, 2020-2024",
    column_config={
        c_: st.column_config.NumberColumn(format="%+.4f")
        for c_ in ("Gain in mean log score", "95% interval from", "95% interval to")
    },
)
