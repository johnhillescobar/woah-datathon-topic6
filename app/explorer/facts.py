"""Headline numbers quoted in the text, read from the result tables so the app never drifts from them."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from explorer import ROOT
from explorer.data import read_table, series_catalog, tests_available

# Periods compared in the technical report (section 6.3).
EARLY_YEARS = (2005, 2009)
LATE_YEARS = (2020, 2024)
# Reporting delay quoted in the technical report (section 3); it is not stored in any result table.
REPORT_DELAY_DAYS = {"median": 15, "p90": 104}
SIM_LABELS = {
    "legacy": "No season screen",
    "baseline": "Baseline (season screen)",
    "ar1": "Autoregressive",
}


@st.cache_data(show_spinner=False)
def reporting_growth() -> dict:
    """Mean countries / events / outbreaks per month in the early and late periods, and the ratio."""
    r = read_table("reporting_intensity")
    out = {}
    for col in ("countries_with_events", "events", "outbreaks"):
        early = r.loc[r["month"].dt.year.between(*EARLY_YEARS), col].mean()
        late = r.loc[r["month"].dt.year.between(*LATE_YEARS), col].mean()
        out[col] = {"early": early, "late": late, "ratio": late / early}
    return out


@st.cache_data(show_spinner=False)
def trend_counts(metric: str = "outbreaks", level: str | None = None) -> dict:
    """How many tested series rise or fall in raw counts and as a share of all notifications."""
    t = read_table(f"trends_annual_{metric}")
    t = t[t["raw_direction"] != "not tested"]
    if level:
        t = t[t["level"] == level]
    rows = {}
    for kind in ("raw", "share"):
        d = t[f"{kind}_direction"]
        rows[kind] = {
            "up": int((d == "increasing").sum()),
            "down": int((d == "decreasing").sum()),
            "flat": int((d == "no trend").sum()),
        }
    return {"tested": len(t), **rows}


@st.cache_data(show_spinner=False)
def rising_global(metric: str = "outbreaks") -> list[str]:
    """Global diseases that rise as a share of all notifications."""
    t = read_table(f"trends_annual_{metric}")
    return t.loc[
        (t["level"] == "global") & (t["share_direction"] == "increasing"), "disease"
    ].tolist()


@st.cache_data(show_spinner=False)
def peak_months() -> dict[str, int]:
    """Seasonal peak month (1-12) of every global outbreak series."""
    t = read_table("trends_stl_outbreaks")
    t = t[t["level"] == "global"]
    return dict(zip(t["disease"], t["peak_month"].astype(int)))


@st.cache_data(show_spinner=False)
def detector() -> dict:
    """Totals for the baseline detector on outbreaks."""
    alerts = read_table("alerts_outbreaks")
    episodes = read_table("episodes_outbreaks")
    cat = series_catalog("outbreaks", "baseline") if tests_available() else None
    return {
        "alert_months": len(alerts),
        "episodes": len(episodes),
        "holdout_alerts": int((alerts["period"] == "holdout").sum()),
        "series": None if cat is None else len(cat),
    }


@st.cache_data(show_spinner=False)
def model_gains() -> dict:
    """Log-score gain of the GBM and LSTM over the best statistical model (ar1), by period."""
    m = read_table("model_comparison")
    m = m[
        (m["metric"] == "dlogscore")
        & (m["stratum"] == "all")
        & (m["ref_model"] == "ar1")
    ]
    return {(r.model, r.period): (r.value, r.lo, r.hi) for r in m.itertuples()}


@st.cache_data(show_spinner=False)
def weather() -> dict:
    """Numbers behind the weather verdict."""
    tests = read_table("climate_association_tests").set_index("Unnamed: 0")
    coef = read_table("climate_association")
    row = coef[
        (coef["term"] == "prec_z_lag1_vector") & (coef["adjustment"] == "year_effects")
    ].iloc[0]
    exp = read_table("nb_experiments")
    gain = exp[
        (exp["period"] == "2020-2024")
        & (exp["comparison"] == "nb_precip_vector - nb_pooled")
    ].iloc[0]
    return {
        "n_series": int(tests.loc["n_series", "year_effects"]),
        "n_terms": int((coef["adjustment"] == "year_effects").sum()),
        "placebo_p_year_effects": tests.loc["placebo_p", "year_effects"],
        "placebo_p_no_year_effects": tests.loc["placebo_p", "no_year_effects"],
        "rr": row["irr_per_sd"],
        "rr_lo": row["ci_lo"],
        "rr_hi": row["ci_hi"],
        "rr_placebo_p": row["placebo_p"],
        "rr_placebo_fw": row["placebo_p_familywise"],
        "precip_gain": gain["mean_log_score"],
    }


@st.cache_data(show_spinner=False)
def companion_summary() -> dict:
    """Facts about the next-month forecast table."""
    f = read_table("companion_forecast")
    horizon = (
        f["forecast_month"].dt.year * 12
        + f["forecast_month"].dt.month
        - f["last_month"].dt.year * 12
        - f["last_month"].dt.month
    ).iloc[0]
    big = f[(f["expected"] >= 100) | (f["expected_lstm"] >= 100)]
    both = big[["expected", "expected_lstm"]]
    gap = both.max(axis=1) / both.min(axis=1)
    return {
        "series": len(f),
        "last_month": f["last_month"].iloc[0],
        "forecast_month": f["forecast_month"].iloc[0],
        "horizon_months": int(horizon),
        "n_large": len(big),
        "max_gap_large": gap.max(),
    }


@st.cache_data(show_spinner=False)
def simulation_summary() -> pd.DataFrame:
    """Detection probability by outbreak size and false alerts, averaged over the baseline levels."""
    s = read_table("simulation_detection").assign(
        multiplier=lambda d: d["multiplier"].astype(str)
    )
    det = (
        s[s["multiplier"] != "none"]
        .groupby(["model", "multiplier"])["detection_prob"]
        .mean()
        .unstack()
    )
    det = det[sorted(det.columns, key=float)]
    det.columns = [f"Detected: {c}x outbreak" for c in det.columns]
    det["False alerts per 100 outbreak-free months"] = (
        s[s["multiplier"] == "none"]
        .groupby("model")["false_alerts_per_100_months"]
        .mean()
    )
    det = det.loc[[m for m in SIM_LABELS if m in det.index]]
    det.index = [SIM_LABELS[m] for m in det.index]
    return det


@st.cache_data(show_spinner=False)
def unit_test_count() -> int | None:
    """Number of unit tests of the pipeline (this app's own tests are not counted)."""
    files = [p for p in (ROOT / "tests").glob("test_*.py") if p.name != "test_app.py"]
    if not files:
        return None
    return sum(
        len(
            re.findall(
                r"^\s*def test_", p.read_text(encoding="utf8"), flags=re.MULTILINE
            )
        )
        for p in files
    )
