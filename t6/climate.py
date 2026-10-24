"""Link outbreaks to WorldClim monthly weather and derive monthly anomalies.

Steps: (1) map outbreak coordinates to 10-arcmin grid cells, snapping ocean/no-data cells to the
nearest valid land cell; (2) read tmin/tmax/prec for those cells for every month 1990-2024;
(3) express each month as an anomaly against the cell's 1991-2020 calendar-month climatology.

Run with `uv run python -m t6.climate`.
"""
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy.spatial import cKDTree

from t6 import config as C

MONTH_RE = re.compile(r"_(\d{4})-(\d{2})\.tif$")


def month_files(var: str) -> list[tuple[pd.Timestamp, str]]:
    files = []
    for f in sorted((C.WC_DIR / var).glob("*.tif")):
        y, m = MONTH_RE.search(f.name).groups()
        files.append((pd.Timestamp(int(y), int(m), 1), str(f)))
    return files


def assign_cells(lon: np.ndarray, lat: np.ndarray) -> pd.DataFrame:
    """Grid cell per point; points on no-data cells are snapped to the nearest valid cell."""
    ref_file = month_files("tmin")[0][1]
    with rasterio.open(ref_file) as src:
        tr, shape, nodata = src.transform, src.shape, src.nodata
        band = src.read(1)
    valid = np.isfinite(band) & (band != nodata) & (band > -1e10)
    r, c = rowcol(tr, lon, lat)
    r = np.clip(np.asarray(r), 0, shape[0] - 1)
    c = np.clip(np.asarray(c), 0, shape[1] - 1)
    snapped = ~valid[r, c]
    if snapped.any():
        vr, vc = np.nonzero(valid)
        tree = cKDTree(np.column_stack([vr, vc]))
        dist, idx = tree.query(np.column_stack([r[snapped], c[snapped]]))
        r[snapped], c[snapped] = vr[idx], vc[idx]
        print(f"snapped {snapped.sum():,} points to nearest land cell (max {dist.max():.1f} cells)")
    return pd.DataFrame({"row": r, "col": c, "snapped": snapped,
                         "cell_id": r * shape[1] + c})


def extract(cells: pd.DataFrame) -> pd.DataFrame:
    """Long table (cell_id, month, tmin, tmax, prec) for each unique cell."""
    u = cells.drop_duplicates("cell_id")[["cell_id", "row", "col"]].reset_index(drop=True)
    out = {}
    for var in C.WC_VARS:
        series = []
        for month, f in month_files(var):
            with rasterio.open(f) as src:
                band = src.read(1)
            series.append(band[u["row"].to_numpy(), u["col"].to_numpy()])
        out[var] = np.stack(series, axis=1).astype("float32")  # cells x months
        months = [m for m, _ in month_files(var)]
        print(f"{var}: {out[var].shape}")
    n_cells = len(u)
    long = pd.DataFrame({
        "cell_id": np.repeat(u["cell_id"].to_numpy(), len(months)),
        "month": np.tile(months, n_cells),
        **{v: out[v].ravel() for v in C.WC_VARS},
    })
    return long


def add_anomalies(long: pd.DataFrame) -> pd.DataFrame:
    """Anomaly vs the cell's calendar-month climatology over C.CLIM_REF.

    tmin/tmax: difference in degC. prec: z-score of monthly total (gamma-free, robust enough at
    monthly scale; the ratio is unstable in arid cells where the climatological mean is ~0).
    """
    long = long.copy()
    long["cal_month"] = long["month"].dt.month
    ref = long[(long["month"] >= C.CLIM_REF[0]) & (long["month"] <= C.CLIM_REF[1])]
    stats = ref.groupby(["cell_id", "cal_month"])[list(C.WC_VARS)].agg(["mean", "std"])
    stats.columns = [f"{v}_{s}" for v, s in stats.columns]
    long = long.merge(stats, left_on=["cell_id", "cal_month"], right_index=True, how="left")
    for v in ("tmin", "tmax"):
        long[f"{v}_anom"] = long[v] - long[f"{v}_mean"]
    z = (long["prec"] - long["prec_mean"]) / long["prec_std"].clip(lower=1.0)
    clipped = z.abs() > C.PREC_Z_CLIP
    print(f"prec_z clipped to +/-{C.PREC_Z_CLIP}: {clipped.sum():,} cell-months "
          f"({clipped.mean():.4%}) in {long.loc[clipped, 'cell_id'].nunique():,} cells")
    long["prec_z"] = z.clip(-C.PREC_Z_CLIP, C.PREC_Z_CLIP)
    keep = ["cell_id", "month", "tmin", "tmax", "prec", "tmin_anom", "tmax_anom", "prec_z"]
    return long[keep]


def main() -> None:
    ob = pd.read_parquet(C.INTERIM / "outbreaks.parquet", columns=["Outbreak_id", "Longitude", "Latitude"])
    cells = assign_cells(ob["Longitude"].to_numpy(), ob["Latitude"].to_numpy())
    ob = pd.concat([ob[["Outbreak_id"]], cells[["cell_id", "snapped"]]], axis=1)
    ob.to_parquet(C.INTERIM / "outbreak_cells.parquet")
    print(f"{len(ob):,} outbreaks -> {ob['cell_id'].nunique():,} unique cells")
    long = add_anomalies(extract(cells))
    long.to_parquet(C.INTERIM / "cell_climate.parquet")
    print(long.describe().T[["count", "mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
