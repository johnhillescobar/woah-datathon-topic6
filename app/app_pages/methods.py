"""Methods and limits: how the results were produced, how they were checked and how far to trust them."""

import pandas as pd
import streamlit as st
from explorer import facts
from explorer.data import C, read_table

REPORT_FILES = [
    (
        "technical_report.html",
        "Technical report (self-contained web page)",
        "text/html",
    ),
    ("technical_report.pdf", "Technical report (PDF)", "application/pdf"),
]

st.title("Methods and limits")
st.markdown(
    "How the numbers were produced, how they were checked, and what they can and cannot support. The full "
    "technical report has the detail; this page is the short version."
)

st.header("Method in short")
st.markdown(
    f"""
1. **Series.** For every disease, the number of outbreaks (or new events) notified each month, for the world, each
   WOAH region and each country. Series with too few notifications are left out.
2. **What is normal.** For each month, a seasonal level is learned from the previous {C.WINDOW_MONTHS // 12} years
   only, ignoring one-off waves so that a past wave does not count as normal. It gives a full range of plausible
   counts, not one number.
3. **What is unusual.** A month is an *alert* when it is far above that range (false discovery rate
   {C.FDR_Q:.0%}), has at least {C.MIN_ALERT_COUNT["outbreaks"]} outbreaks
   ({C.MIN_ALERT_COUNT["events"]} events) and is at least {C.MIN_ALERT_RATIO:.0f} times the expected count.
   Consecutive alert months form one *episode*. Diseases appearing for the first time, or returning after
   a long absence, are listed separately because there is nothing to compare them with.
4. **Trends.** Raw annual counts and each disease's share of all notifications are tested for direction; the
   monthly series is split into trend, season and what is left over.
5. **Next month.** Two pooled forecasting models (a gradient-boosted model and an LSTM neural network) give an
   expected count and range for the next month. They supplement the alerts.
6. **Weather.** Monthly temperature and rainfall anomalies one to three months earlier are compared with outbreak
   counts, with placebo tests that shuffle years of weather.
"""
)

st.header("How it was checked")
st.markdown(
    "**Simulation.** The detectors were run on simulated 20-year seasonal series with known, injected outbreaks "
    "(average over three baseline levels). The *season screen* is the step that keeps a one-off wave out of "
    "the baseline."
)
sim = facts.simulation_summary()
st.dataframe(
    sim,
    alt="Share of injected outbreaks detected, by size, and false alerts for three detectors",
    column_config={
        c: st.column_config.NumberColumn(
            format="percent" if c.startswith("Detected") else "%.2f"
        )
        for c in sim.columns
    },
)
false_rate = sim.loc[
    "Baseline (season screen)", "False alerts per 100 outbreak-free months"
]
st.caption(
    "Detected: the share of simulated outbreaks of that size (a multiple of the normal level, lasting three months) "
    "that raised an alert. False alerts: months flagged in series with no outbreak, per 100 months; the baseline's "
    f"{false_rate:.2f} per 100 months is about one false alert per {100 / false_rate / 12:.0f} years per series."
)

tests = facts.unit_test_count()
st.markdown(
    (f"**Unit tests ({tests}).** " if tests else "**Unit tests.** ")
    + "They check that the fitting code equals a reference implementation, that false alarms on simulated normal "
    "data stay below 2% at a nominal 1%, that an injected outbreak is detected, that one wave does not distort the "
    "following year's expectation, that sparse data no longer produce exploding fits, that the forecast models "
    "cannot see the future, and that the weather placebo preserves each country's data. Run them with "
    "`uv run pytest`."
)
st.markdown(
    "**Rolling-origin scoring.** Every forecast uses only data from before the month it predicts, refitted year by "
    "year from 2018, and all models are scored on the same months. The decision rule for adopting a model was "
    "written down before scoring."
)
cal = read_table("calibration")
cal_ar1 = read_table("calibration_ar1")
if cal is not None and cal_ar1 is not None:
    both = pd.concat([cal, cal_ar1])
    both = both[both["metric"] == "outbreaks"].assign(
        model=lambda d: d["model"].map(
            {"baseline": "Baseline", "ar1": "Autoregressive"}
        ),
        period=lambda d: d["period"].map(
            {"retrospective": "2007-2024", "holdout": "2025-26 holdout"}
        ),
    )
    st.dataframe(
        both[["model", "period", "n_tests", "n_alerts", "alert_rate", "sd_z"]].rename(
            columns={
                "model": "Detector",
                "period": "Period",
                "n_tests": "Months tested",
                "n_alerts": "Alert months",
                "alert_rate": "Alert rate",
                "sd_z": "Residual spread (1 = calibrated)",
            }
        ),
        hide_index=True,
        alt="Calibration of the outbreak detectors: tests, alerts and residual spread",
        column_config={
            "Months tested": st.column_config.NumberColumn(format="%d"),
            "Alert rate": st.column_config.NumberColumn(format="percent"),
            "Residual spread (1 = calibrated)": st.column_config.NumberColumn(
                format="%.2f"
            ),
        },
    )
    st.caption(
        "A residual spread above 1 means real counts vary more than the model allows, so some alerts are "
        "partly over-dispersion. Real-data alert rates are not false-alarm rates (genuine anomalies are in "
        "them); the simulation is the evidence on error rates."
    )

st.header("Limitations")
delay, gap = facts.REPORT_DELAY_DAYS, facts.companion_summary()
st.markdown(
    f"""
- **Notified events only.** Reporting effort, surveillance and trade or political context are mixed with disease
  occurrence. Europe is about 70% of the data, so "global" results are largely European, and three diseases
  (African swine fever, bird flu, bluetongue) drive pooled results.
- **No true absences.** A month without a notification is a zero, not proof that the disease was absent.
- **Recent months are incomplete.** The median report arrives {delay["median"]} days after an outbreak starts, but one in ten
  takes more than {delay["p90"]} days, so the newest {C.CENSOR_MONTHS} months are left out and every result uses
  final counts, not what would have been visible in real time.
- **Waves that recur can count as normal.** After a multi-year wave the baseline can still expect too much.
- **Choices were fixed, not tuned.** The five-year window, 5% false discovery rate and minimum alert size are
  sensible but not unique.
- **Weather is coarse.** Country-footprint averages of half-degree anomalies, monthly, ending December 2024; only
  temperature and rainfall are available, so local conditions and humidity are out of reach.
- **Forecasts are one month ahead only.** For the largest series ({gap["n_large"]} series expected to exceed 100
  outbreaks) the two models differ by up to {gap["max_gap_large"]:.1f} times, and the gradient-boosted model's
  ranges are too wide there.
- **The rainfall pattern was found after looking.** It should be confirmed on new data before anyone relies on it.
"""
)

st.header("What is, and is not, supported as early warning")
ok, no = st.columns(2)
with ok:
    st.success(
        title="Supported",
        icon=":material/check_circle:",
        body=(
            "- Flagging a surge **in the month it happens**, on final counts.\n"
            "- An expectation and range for **next month**, so one month of lead time for waves already under way.\n"
            "- A list of first and returning occurrences to review next to the alerts.\n"
            "- Disease-specific seasons for planning surveillance (for example bluetongue in autumn, bird flu in winter)."
        ),
    )
with no:
    st.warning(
        title="Not supported",
        icon=":material/block:",
        body=(
            "- **Pre-emptive warning** before a wave begins, or lead times longer than one month.\n"
            "- **Real-time use** with reporting delays: the latest months are incomplete and this was not tested.\n"
            "- **Weather as a predictor**: at most one exploratory rainfall signal for vector-borne diseases.\n"
            "- **Spread from neighbouring countries**, the most plausible source of longer lead times, was not tested."
        ),
    )

st.header("Reproduce it")
st.markdown(
    "From the project folder, with [uv](https://docs.astral.sh/uv/) installed. The downloaded data go to `data/`, "
    "which is not stored in the repository; everything in it is re-created by these commands."
)
st.code(
    """uv sync                          # Python 3.13 environment (core, no torch)
uv run python -m t6.fetch        # download WAHIS INFUR extract + WorldClim rasters (about 2 GB)
uv run python -m t6.ingest       # Excel -> typed outbreak / event tables
uv run python -m t6.climate      # sample WorldClim at outbreak locations, monthly anomalies
uv run python -m t6.exposure     # country-level climate exposure
uv run python -m t6.detect       # anomaly detection, two variants
uv run python -m t6.trends       # STL, Mann-Kendall / Theil-Sen, change points
uv run python -m t6.simulate     # simulation benchmark of the detectors
uv run python -m t6.deep gbm     # pooled GBM benchmark (add the LSTM with: uv sync --extra deep)
uv run python -m t6.compare      # score every model on the same rows
uv run python -m t6.companion    # expected-next-month forecasts
uv run python -m t6.nbmodels     # negative-binomial regressions, climate association
uv run python -m t6.gam          # smooth-curve (GAM) version of the climate test
uv run pytest                    # unit tests and simulated benchmarks
uv run streamlit run app/streamlit_app.py   # this explorer""",
    language="bash",
)

st.header("Report files")
st.markdown(
    "- `reports/technical_report.html` and `.pdf`: the full report (method, results, validation, limitations)\n"
    "- `reports/technical_report.md`: the same text in Markdown\n"
    "- `reports/dl_evaluation.md`: the forecast-model benchmark, protocol and adoption rule\n"
    "- `reports/tables/`: every result table behind this app\n"
    "- `reports/figures/`: the static figures\n"
    "- `README.md`: the quick start and the decision log"
)
buttons = st.container(horizontal=True)
for filename, label, mime in REPORT_FILES:
    path = C.REPORTS / filename
    if path.exists():
        buttons.download_button(
            label,
            data=path.read_bytes,
            file_name=filename,
            mime=mime,
            icon=":material/download:",
            key=f"report_{filename}",
            on_click="ignore",
        )
