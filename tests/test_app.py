"""Smoke tests for the Streamlit explorer: every page renders with default inputs and key elements appear."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from t6 import config as C

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
sys.path.insert(0, str(ROOT / "app"))

PAGES = ["overview", "alerts", "next_month", "trends", "weather", "methods"]
needs_data = pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "tests_outbreaks.parquet").exists(),
    reason="detector output missing: run `uv run python -m t6.detect`",
)


def open_page(page: str) -> AppTest:
    """Run the app and switch to one page (the overview is the default page)."""
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    if page != "overview":
        at.switch_page(f"app_pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def charts(at: AppTest) -> list:
    return at.get("vega_lite_chart")


def texts(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


@needs_data
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_default_inputs(page):
    at = open_page(page)
    assert len(at.title) == 1
    assert not at.error
    assert charts(at) or page in {"methods"}  # the methods page is text and tables


@needs_data
def test_overview_has_findings_citation_and_exhibit():
    at = open_page("overview")
    assert len(at.metric) == 6
    assert C.CITATION in texts(at)
    assert "Data extracted by Latin Team" in C.CITATION
    assert len(charts(at)) == 1
    values = {m.label: m.value for m in at.metric}
    assert any(
        "2,210" in v for v in values.values()
    )  # alert months, read from reports/tables


@needs_data
def test_alerts_defaults_and_dependent_selects():
    at = open_page("alerts")
    assert at.selectbox(key="alerts_disease").value == "Bluetongue virus"
    assert at.selectbox(key="alerts_area").value == "all"
    assert len(charts(at)) == 2  # the series chart and the largest-episodes chart
    assert (
        len(at.dataframe) >= 2
    )  # episodes of the series and the first-occurrence list
    at.segmented_control(key="alerts_level").set_value("country").run()
    assert not at.exception
    area = at.selectbox(key="alerts_area")
    assert (
        area.value != "all" and len(area.options) > 1
    )  # a country, not the global series
    at.segmented_control(key="alerts_level").set_value("global").run()
    assert not at.exception
    assert at.selectbox(key="alerts_area").value == "all"


@needs_data
def test_alerts_other_count_and_detector_variants():
    at = open_page("alerts")
    at.segmented_control(key="alerts_variant").set_value("ar1").run()
    assert not at.exception
    at.segmented_control(key="alerts_metric").set_value("events").run()
    assert not at.exception
    disease = at.selectbox(key="alerts_disease")
    assert (
        disease.value in disease.options
    )  # Bluetongue may have no events series: falls back safely


@needs_data
def test_next_month_banner_table_and_search():
    at = open_page("next_month")
    assert "not an alert threshold" in at.warning[0].proto.title
    assert len(at.dataframe[0].value) == 481
    at.text_input(key="next_query").set_value("bluetongue").run()
    assert not at.exception
    shown = at.dataframe[0].value
    assert 0 < len(shown) < 481
    assert shown["Disease"].str.lower().str.contains("bluetongue").all()
    at.text_input(key="next_query").set_value("no such disease").run()
    assert not at.exception
    assert at.info  # friendly message instead of an empty table


@needs_data
def test_trends_headline_numbers_come_from_the_tables():
    at = open_page("trends")
    values = {m.label: m.value for m in at.metric}
    assert values["Established series tested"] == "98"
    assert values["Rising in raw counts"] == "23"
    assert values["Rising as a share of notifications"] == "4"
    at.segmented_control(key="trends_level").set_value("region").run()
    assert not at.exception


@needs_data
def test_weather_charts_and_verdict():
    at = open_page("weather")
    assert at.info[0].proto.title.startswith("Verdict")
    assert len(charts(at)) == 5  # three lags and two forest panels
    at.segmented_control(key="wx_group").set_value("other").run()
    at.segmented_control(key="wx_variable").set_value("tmax_anom").run()
    assert not at.exception


@needs_data
def test_methods_page_lists_validation_and_reproduction():
    at = open_page("methods")
    assert len(at.dataframe[0].value) == 3  # three detectors in the simulation table
    assert "uv run python -m t6.detect" in " ".join(c.value for c in at.code)


@pytest.fixture
def no_processed_folder(monkeypatch, tmp_path):
    st.cache_data.clear()
    monkeypatch.setattr(C, "PROCESSED", tmp_path)
    yield
    st.cache_data.clear()


def test_missing_processed_data_shows_commands_and_other_pages_still_work(
    no_processed_folder,
):
    at = open_page("alerts")
    assert at.info[0].proto.title == "Detector results not found"
    assert "t6.detect" in " ".join(c.value for c in at.code)
    assert len(charts(at)) == 1  # the largest-episodes chart only needs reports/tables
    for page in ("overview", "next_month", "trends", "weather", "methods"):
        open_page(page)


def test_short_disease_labels_are_unique_and_shorter():
    from explorer.data import short_disease

    names = pd.read_csv(ROOT / "reports" / "tables" / "companion_forecast.csv")[
        "disease"
    ].unique()
    labels = [short_disease(n) for n in names]
    assert len(set(labels)) == len(labels)
    assert (
        short_disease("Aethina tumida(Small hive beetle)(2006-)")
        == "Small hive beetle (Aethina tumida)"
    )
    assert short_disease("Tilapia lake virus (2023-)") == "Tilapia lake virus"
    assert max(len(x) for x in labels) <= 55


def test_chart_builders_and_figure_lookup():
    from explorer import charts as ch
    from explorer.data import stl_figure

    months = pd.date_range("2023-01-01", periods=6, freq="MS")
    d = pd.DataFrame(
        {
            "month": months,
            "y": [1.0, 2, 0, 9, 3, 1],
            "mu": [1.5] * 6,
            "upper99": [6.0] * 6,
            "ratio": [0.7, 1.3, 0, 6, 2, 0.7],
            "p": [0.5, 0.3, 1, 0.001, 0.1, 0.5],
            "alert": [False, False, False, True, False, False],
            "status": ["tested"] * 6,
        }
    )
    spec = ch.alert_chart(
        d,
        title="t",
        subtitle="s",
        y_label="y",
        holdout_start=pd.Timestamp(C.HOLDOUT_START),
    ).to_dict()
    assert spec["title"]["text"] == "t" and len(spec["layer"]) >= 5
    assert stl_figure("HPAI (all hosts)") is not None
    assert stl_figure("Disease that does not exist") is None
