"""Download raw inputs: WAHIS INFUR extract and WorldClim monthly weather.

Idempotent: existing files are skipped. Run with `uv run python -m t6.fetch`.
"""

import sys
import zipfile
from pathlib import Path

import requests

from t6 import config as C


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"skip   {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            f.writelines(r.iter_content(1 << 20))
    tmp.rename(dest)
    print(f"fetched {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def fetch_worldclim() -> None:
    for var in C.WC_VARS:
        out_dir = C.WC_DIR / var
        for period in C.WC_PERIODS:
            name = f"wc2.1_cruts4.09_{C.WC_RES}_{var}_{period}"
            if len(list(out_dir.glob(f"{name}*.tif"))) >= 1:
                print(f"skip   {name} (extracted)")
                continue
            zpath = download(f"{C.WC_BASE}/{name}.zip", out_dir / f"{name}.zip")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(out_dir)
            zpath.unlink()  # keep only the GeoTIFFs


def main() -> None:
    download(C.INFUR_URL, C.INFUR_XLSX)
    fetch_worldclim()
    print("done")


if __name__ == "__main__":
    sys.exit(main())
