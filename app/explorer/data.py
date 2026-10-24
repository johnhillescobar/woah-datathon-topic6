"""Cached loaders and display labels. Everything here reads files the pipeline already produced."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from t6 import config as C

TABLES = C.REPORTS / "tables"
FIGURES = C.REPORTS / "figures"

LEVELS = ("global", "region", "country")
LEVEL_LABEL = {"global": "Global", "region": "WOAH region", "country": "Country"}
METRIC_LABEL = {"outbreaks": "Outbreaks", "events": "Events"}
VARIANT_LABEL = {"baseline": "Baseline", "ar1": "Autoregressive"}
MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
DATE_COLUMNS = ("month", "start", "end", "last_month", "forecast_month")

# The detector files are written by `python -m t6.detect`; the app shows a hint when they are missing.
GENERATE_COMMANDS = (
    "uv run python -m t6.fetch\nuv run python -m t6.ingest\nuv run python -m t6.detect"
)

# Long WAHIS disease names that need a hand-made short label (full names stay in tooltips and tables).
SHORT_NAMES = {
    "Aethina tumida(Small hive beetle)(2006-)": "Small hive beetle (Aethina tumida)",
    "Cochliomyia hominivorax (New world screwworm)": "New world screwworm",
    "Eastern equine encephalitis virus (Eastern equine encephalomyelitis) (2006-)": "Eastern equine encephalitis virus",
    "Western equine encephalitis virus (Western equine encephalomyelitis) (2006-)": "Western equine encephalitis virus",
    "Infectious salmon anaemia virus (HPR-deleted or HPR0 genotypes)": "Infectious salmon anaemia virus",
    "Melissococcus plutonius (European foulbrood)": "European foulbrood",
    "Paenibacillus larvae (American foulbrood)": "American foulbrood",
    "Middle East respiratory syndrome coronavirus (inf. with) (2022-)": "MERS coronavirus",
    "Mycoplasma gallisepticum (Avian mycoplasmosis)": "Avian mycoplasmosis (M. gallisepticum)",
    "Mycoplasma synoviae (Avian mycoplasmosis) (2006-)": "Avian mycoplasmosis (M. synoviae)",
    "Mycoplasma mycoides subsp. mycoides (Contagious bovine pleuropneumonia)": "Contagious bovine pleuropneumonia",
    "Pathogenic rabbit lagoviruses (Rabbit haemorrhagic disease)": "Rabbit haemorrhagic disease",
    "Taylorella equigenitalis (Contagious equine metritis)": "Contagious equine metritis",
    "Theileria equi and Babesia caballi (Equine piroplasmosis)": "Equine piroplasmosis",
    "Burkholderia mallei (Glanders)": "Glanders",
    "Varroa spp. (Varroosis)": "Varroosis (Varroa spp.)",
}
_YEARS = re.compile(
    r"\s*\((?:\d{4}-\d{0,4}|-\d{4})\)\s*$"
)  # WAHIS list-version suffix such as "(2006-)"


def short_disease(name: str) -> str:
    """Readable label for a WAHIS disease name."""
    if name in SHORT_NAMES:
        return SHORT_NAMES[name]
    return re.sub(r"(?<=\S)\(", " (", _YEARS.sub("", name)).strip()


def _read_csv(path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, parse_dates=[c for c in DATE_COLUMNS if c in header])


@st.cache_data(show_spinner=False, max_entries=64)
def read_table(name: str) -> pd.DataFrame | None:
    """A committed result table from reports/tables, or None when the file is missing."""
    path = TABLES / f"{name}.csv"
    return _read_csv(path) if path.exists() else None


def suffix(variant: str) -> str:
    return "" if variant == "baseline" else f"_{variant}"


def tests_path(metric: str, variant: str = "baseline"):
    return C.PROCESSED / f"tests_{metric}{suffix(variant)}.parquet"


def tests_available() -> bool:
    return tests_path("outbreaks").exists()


@st.cache_data(show_spinner=False, max_entries=8)
def read_tests(metric: str, variant: str) -> pd.DataFrame | None:
    """One row per series and tested month (detector output), or None before `t6.detect` has run."""
    path = tests_path(metric, variant)
    if not path.exists():
        return None
    cols = [
        "level",
        "disease",
        "area",
        "month",
        "y",
        "mu",
        "upper99",
        "p",
        "q",
        "ratio",
        "alert",
        "status",
        "period",
    ]
    return pd.read_parquet(path, columns=cols)


@st.cache_data(show_spinner=False, max_entries=8)
def series_catalog(metric: str, variant: str) -> pd.DataFrame | None:
    """Series that have at least one tested month, with their alert counts and totals."""
    df = read_tests(metric, variant)
    if df is None:
        return None
    df = df.assign(tested=df["status"] == "tested")
    cat = (
        df.groupby(["level", "disease", "area"])
        .agg(alerts=("alert", "sum"), total=("y", "sum"), tested=("tested", "sum"))
        .reset_index()
    )
    return cat[cat["tested"] > 0].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def read_history() -> pd.DataFrame | None:
    """Observed monthly counts for the companion series (pooled-model rolling-origin file)."""
    for path in (C.PROCESSED / "tests_outbreaks_gbm.parquet", tests_path("outbreaks")):
        if path.exists():
            return pd.read_parquet(
                path, columns=["level", "disease", "area", "month", "y"]
            )
    return None


@st.cache_data(show_spinner=False)
def country_names() -> dict[str, str]:
    """ISO code -> country name from the interim table; empty (codes are shown) when it is absent."""
    path = C.INTERIM / "outbreaks.parquet"
    if not path.exists():
        return {}
    d = pd.read_parquet(path, columns=["iso_code", "country"]).drop_duplicates(
        "iso_code"
    )
    return dict(zip(d["iso_code"], d["country"]))


def area_label(level: str, area: str) -> str:
    if level == "global":
        return "All countries"
    if level == "region":
        return area
    name = country_names().get(area)
    return f"{name} ({area})" if name else area


def series_title(level: str, disease: str, area: str) -> str:
    """'Bluetongue virus, global' / 'HPAI (all hosts), Europe' / 'Rabies virus, Germany (DEU)'."""
    where = "global" if level == "global" else area_label(level, area)
    return f"{short_disease(disease)}, {where}"


def month_name(n: int) -> str:
    return MONTH_NAMES[int(n) - 1]


def stl_figure(disease: str):
    """Path of the pipeline's STL decomposition figure for a global disease, if it exists."""
    path = FIGURES / f"stl_{re.sub(r'[^A-Za-z0-9]', '_', disease)}.png"
    return path if path.exists() else None
