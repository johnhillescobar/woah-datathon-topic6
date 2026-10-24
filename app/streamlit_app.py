"""Explorer for WOAH Datathon Topic 6 (trends in animal disease occurrence), Latin Team.

Run from the project root: uv run streamlit run app/streamlit_app.py
The app only displays results written by the t6 pipeline (data/processed and reports/tables).
"""

import explorer  # noqa: F401  (puts the project root on sys.path so that t6 can be imported)
import streamlit as st

from t6 import config as C

st.set_page_config(
    page_title="WAHIS disease trends explorer",
    page_icon=":material/vaccines:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(
            "app_pages/overview.py",
            title="Overview",
            icon=":material/home:",
            default=True,
        ),
        st.Page(
            "app_pages/alerts.py",
            title="Alerts",
            icon=":material/notifications_active:",
        ),
        st.Page(
            "app_pages/next_month.py",
            title="Next month",
            icon=":material/event_upcoming:",
        ),
        st.Page("app_pages/trends.py", title="Trends", icon=":material/trending_up:"),
        st.Page("app_pages/weather.py", title="Weather", icon=":material/cloud:"),
        st.Page(
            "app_pages/methods.py",
            title="Methods and limits",
            icon=":material/fact_check:",
        ),
    ],
    position="top",
)
page.run()

if page.title != "Overview":  # the Overview shows the citation in full
    st.divider()
    st.caption(f"Data citation: {C.CITATION}")
