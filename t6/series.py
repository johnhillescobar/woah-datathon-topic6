"""Monthly count series per disease x area (global, WOAH region, country)."""
import pandas as pd

from t6 import config as C


def score_end() -> pd.Timestamp:
    """Last month considered complete: extraction date minus the reporting-lag margin."""
    cutoff = pd.Timestamp(C.EXTRACT_DATE) - pd.DateOffset(months=C.CENSOR_MONTHS)
    return cutoff.to_period("M").to_timestamp() - pd.DateOffset(months=1)


def months() -> pd.DatetimeIndex:
    return pd.date_range(C.START, score_end(), freq="MS")


def merge_diseases(names: pd.Series) -> pd.Series:
    out = names.copy()
    for pattern, label in C.DISEASE_MERGE:
        out = out.where(~names.str.contains(pattern, regex=True), label)
    return out


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    ob = pd.read_parquet(C.INTERIM / "outbreaks.parquet")
    ev = pd.read_parquet(C.INTERIM / "events.parquet")
    for df in (ob, ev):
        df["disease"] = merge_diseases(df["disease"])
    return ob[ob["region"] != "Other"], ev[ev["region"] != "Other"]


def monthly_counts(ob: pd.DataFrame, ev: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Wide table: index (level, disease, area), one column per month, zeros filled in."""
    df = {"outbreaks": ob, "events": ev}[metric]
    df = df.assign(ALL="all")
    idx = months()
    parts = []
    for level, area_col in (("global", "ALL"), ("region", "region"), ("country", "iso_code")):
        g = df.groupby(["disease", area_col, "start_month"]).size().unstack("start_month", fill_value=0)
        g = g.reindex(columns=idx, fill_value=0)
        g.index = pd.MultiIndex.from_tuples([(level, d, a) for d, a in g.index],
                                            names=["level", "disease", "area"])
        parts.append(g)
    wide = pd.concat(parts)
    return wide[wide.sum(axis=1) >= C.MIN_SERIES_TOTAL[metric]]
