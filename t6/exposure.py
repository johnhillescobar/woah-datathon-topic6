"""Country-level monthly climate exposure.

A country's exposure is the mean weather anomaly over the grid cells in which it has ever notified
an outbreak (any disease, any year). Using the notification footprint, not the whole territory,
keeps the signal where animals and surveillance are. The footprint is not disease- or month-
specific, so it does not leak the outcome of any single month.

Run with `uv run python -m t6.exposure` (after t6.climate). Writes data/interim/country_climate.parquet.
"""
import numpy as np
import pandas as pd

from t6 import config as C

VARS = ("tmin_anom", "tmax_anom", "prec_z")


def build() -> pd.DataFrame:
    cells = pd.read_parquet(C.INTERIM / "outbreak_cells.parquet")
    ob = pd.read_parquet(C.INTERIM / "outbreaks.parquet", columns=["Outbreak_id", "iso_code"])
    footprint = ob.merge(cells, on="Outbreak_id").drop_duplicates(["iso_code", "cell_id"])

    clim = pd.read_parquet(C.INTERIM / "cell_climate.parquet", columns=["cell_id", "month", *VARS])
    clim = clim.sort_values(["cell_id", "month"])
    months = np.sort(clim["month"].unique())
    cell_ids = clim["cell_id"].to_numpy()[:: len(months)]  # one entry per cell, in row order
    pos = pd.Series(np.arange(len(cell_ids)), index=cell_ids)
    arr = {v: clim[v].to_numpy(np.float32).reshape(len(cell_ids), len(months)) for v in VARS}

    out = []
    for iso, g in footprint.groupby("iso_code"):
        idx = pos.loc[g["cell_id"]].to_numpy()
        part = pd.DataFrame({v: arr[v][idx].mean(0) for v in VARS})
        part.insert(0, "month", months)
        part.insert(0, "iso_code", iso)
        part["n_cells"] = len(idx)
        out.append(part)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    df = build()
    df.to_parquet(C.INTERIM / "country_climate.parquet")
    print(f"{df['iso_code'].nunique()} countries x {df['month'].nunique()} months; "
          f"cells per country: median {df.groupby('iso_code')['n_cells'].first().median():.0f}")
    print(df[list(VARS)].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
