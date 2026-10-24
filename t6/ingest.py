"""Turn the INFUR workbook (outbreak x species rows) into typed outbreak- and event-level tables.

Run with `uv run python -m t6.ingest`. Outputs land in data/interim/.
"""

import re

import pandas as pd

from t6 import config as C

# Fully empty or translated-duplicate columns carry no signal
DROP = [
    "strain_fr",
    "strain_esp",
    "sero_sub_genotype_fr",
    "sero_sub_genotype_esp",
    "disease_fr",
    "disease_esp",
    "water_type",
    "morbidity",
    "mortality",
]


def short_disease(name: str) -> str:
    """'Bluetongue virus (Inf. with)' -> 'Bluetongue virus'; keeps era qualifiers like '(2017-)'."""
    return re.sub(r"\s*\((?:Inf\. (?:with|of [^)]*))\)", "", name).strip()


def read_infur() -> pd.DataFrame:
    df = pd.read_excel(C.INFUR_XLSX, sheet_name=C.INFUR_SHEET)
    df = df.drop(columns=DROP)
    df["disease"] = df["disease_eng"].map(short_disease)
    return df


def build_outbreaks(df: pd.DataFrame) -> pd.DataFrame:
    """One row per Outbreak_id; species-level counts are summed.

    `disease_id` is not a clean key (e.g. bluetongue has ids 29, 32, 46), so group on `disease`.
    """
    first = (
        df.sort_values("Reporting_date")
        .groupby("Outbreak_id", as_index=False)
        .first()[
            [
                "Outbreak_id",
                "epi_event_id",
                "disease_id",
                "disease",
                "iso_code",
                "country",
                "region",
                "Terra_Aqua",
                "Outbreak_start_date",
                "Outbreak_end_date",
                "Reporting_date",
                "Latitude",
                "Longitude",
                "Location_approx",
                "Is_cluster",
                "N_outbreaks_cluster",
                "level1_unique_code",
                "level1_name",
                "Epi_unit",
                "is_wild",
            ]
        ]
    )
    # Follow-up reports repeat outbreaks; keep the latest figures per outbreak x species
    latest = df.sort_values("Reporting_date").drop_duplicates(
        ["Outbreak_id", "Species"], keep="last"
    )
    agg = (
        latest.groupby("Outbreak_id")
        .agg(
            n_species=("Species", "nunique"),
            cases=("cases", "sum"),
            dead=("dead", "sum"),
            susceptible=("susceptible", "sum"),
        )
        .reset_index()
    )
    out = first.merge(agg, on="Outbreak_id")
    out["start_month"] = out["Outbreak_start_date"].dt.to_period("M").dt.to_timestamp()
    out["report_lag_days"] = (
        out["Reporting_date"] - out["Outbreak_start_date"]
    ).dt.days
    return out


def build_events(df: pd.DataFrame, outbreaks: pd.DataFrame) -> pd.DataFrame:
    ev = df.drop_duplicates("epi_event_id")[
        [
            "epi_event_id",
            "disease_id",
            "disease",
            "iso_code",
            "country",
            "region",
            "Terra_Aqua",
            "reason_for_notification",
            "event_start_date",
            "event_confirmation_date",
            "event_closing_date",
        ]
    ]
    n = outbreaks.groupby("epi_event_id").agg(
        n_outbreaks=("Outbreak_id", "size"), n_admin1=("level1_unique_code", "nunique")
    )
    n["n_reports"] = df.groupby("epi_event_id")["Report_id"].nunique()
    ev = ev.merge(n, left_on="epi_event_id", right_index=True, how="left")
    ev["start_month"] = ev["event_start_date"].dt.to_period("M").dt.to_timestamp()
    return ev


def main() -> None:
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    df = read_infur()
    ob = build_outbreaks(df)
    ev = build_events(df, ob)
    ob.to_parquet(C.INTERIM / "outbreaks.parquet")
    ev.to_parquet(C.INTERIM / "events.parquet")
    print(f"rows {len(df):,} -> outbreaks {len(ob):,}, events {len(ev):,}")
    print(
        f"outbreaks missing coords: {ob[['Latitude', 'Longitude']].isna().any(axis=1).sum()}"
    )


if __name__ == "__main__":
    main()
