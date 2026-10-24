"""Helpers for the Streamlit explorer: file loading, labels, charts and small widgets."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(
        str(ROOT)
    )  # t6 is a plain folder next to app/, not an installed package
