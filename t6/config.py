"""Paths, URLs and analysis constants for Topic 6 (trends in animal disease occurrence)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

INFUR_URL = "https://data-integration-department-woah.github.io/WOAH-Datathon/data/infur_20260702.xlsx"
INFUR_XLSX = RAW / "infur" / "infur_20260702.xlsx"
INFUR_SHEET = "infur_20260702"

# WorldClim 2.1 historical monthly weather, downscaled from CRU-TS 4.09 (tmin, tmax, prec)
WC_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/hist/cts4.09"
WC_RES = "10m"  # CRU-TS is 0.5 degrees natively; finer rasters add no anomaly information
WC_VARS = ("tmin", "tmax", "prec")
WC_PERIODS = ("1990-1999", "2000-2009", "2010-2019", "2020-2024")
WC_DIR = RAW / "worldclim"

# Analysis window. WorldClim ends 2024-12, so 2025+ is a disease-only holdout.
START = "2005-01-01"
END = "2024-12-31"
CLIM_REF = ("1991-01", "2020-12")  # WMO-style 30-year reference for monthly anomalies

PREC_Z_CLIP = 5.0  # |z| bound for monthly precipitation anomalies (see README, decision log)

# Extract date of the INFUR file; late reports mean the newest months are incomplete
EXTRACT_DATE = "2026-07-02"
CENSOR_MONTHS = 4  # drop months within 4 months of extraction (90th pct report lag = 104 days)
HOLDOUT_START = "2025-01-01"  # WorldClim ends 2024-12: later months are a disease-only holdout

# Disease names that are one surveillance target split across WAHIS list changes
DISEASE_MERGE = [
    (r"^(?:High pathogenicity avian influenza|(?:Avian i|I)nfluenza A viruses of high pathogenicity)",
     "HPAI (all hosts)"),
]

# Detector (Farrington-style quasi-Poisson baseline, refit every month on a trailing window)
WINDOW_MONTHS = 60
MIN_WINDOW = 24  # months of history needed before a series is testable
MIN_WINDOW_SUM = {"outbreaks": 10, "events": 5}  # total count inside the training window
MIN_ALERT_COUNT = {"outbreaks": 3, "events": 2}  # ignore alerts on trivial absolute counts
MIN_ALERT_RATIO = 2.0  # observed / expected
FDR_Q = 0.05
MIN_SERIES_TOTAL = {"outbreaks": 30, "events": 10}

CITATION = (
    "World Organisation for Animal Health (WOAH) (2026) - WAHIS periodical extraction of "
    "early warning outbreak data. Retrieved on 2026-10-23. Data extracted by Latin Team."
)
