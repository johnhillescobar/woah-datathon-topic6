"""Small shared widgets: dependent selects, 'Show the data' expanders and hints for missing files."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from explorer.data import GENERATE_COMMANDS


def select(label: str, options, key: str, *, default=None, format_func=str, **kwargs):
    """Selectbox whose options depend on other widgets: a stale choice falls back to the default."""
    options = list(options)
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]
    return st.selectbox(label, options, key=key, format_func=format_func, **kwargs)


def segmented(label: str, options, key: str, default, *, format_func=str, **kwargs):
    """Single-choice segmented control that cannot be left empty."""
    value = st.segmented_control(
        label,
        list(options),
        default=default,
        required=True,
        key=key,
        format_func=format_func,
        **kwargs,
    )
    return default if value is None else value


def show_data(
    df: pd.DataFrame,
    name: str,
    *,
    alt: str,
    column_config=None,
    label: str = "Show the data",
) -> None:
    """Expander with the table behind a chart (the accessible twin of the chart) and a CSV download."""
    with st.expander(label, icon=":material/table:"):
        st.dataframe(df, hide_index=True, column_config=column_config, alt=alt)
        st.download_button(
            "Download as CSV",
            data=lambda: df.to_csv(index=False),
            file_name=f"{name}.csv",
            mime="text/csv",
            icon=":material/download:",
            key=f"download_{name}",
            on_click="ignore",
        )


def look_at(text: str) -> None:
    """The one or two sentence 'what to look at' caption under a chart."""
    st.caption(f":material/visibility: **What to look at.** {text}")


def need_processed(what: str) -> None:
    """Friendly message when the detector output in data/processed is missing."""
    st.info(
        f"{what} needs the detector output in `data/processed/`, which is not on this computer "
        "(the folder is not stored in the repository). Create it from the project folder with these "
        "commands, then reload this page. The other pages still work.",
        title="Detector results not found",
        icon=":material/folder_off:",
    )
    st.code(GENERATE_COMMANDS, language="bash")


def need_table(name: str) -> None:
    """A committed result table is missing: say so and stop the page."""
    st.error(
        f"`reports/tables/{name}.csv` is missing. It is stored in the repository; restore it or rerun "
        "the pipeline step that writes it (see the Methods and limits page).",
        title="Result table not found",
        icon=":material/error:",
    )
    st.stop()


def fmt_int(x) -> str:
    return f"{x:,.0f}"
