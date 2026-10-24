"""Altair chart builders. Palette and mark rules follow the dataviz skill (thin marks, one blue, orange for
highlights, grey for context, hairline recessive grid, legends, tooltips, no dual axes)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"
INK, INK2, SURFACE, GRID, AXIS = "#0b0b0b", "#52514e", "#fcfcfb", "#e1e0d9", "#c3c2b7"
HALF_MONTH = pd.Timedelta(days=16)
# Compact tick labels: 0, 500, 1K, 2.5K (Vega expression)
COUNT_LABELS = "datum.value >= 1000 ? format(datum.value / 1000, ',.2~f') + 'K' : format(datum.value, ',.2~f')"


def finish(
    chart, title: str | None = None, subtitle: str | None = None, height: int = 360
):
    """Apply the shared chart chrome to a finished chart."""
    props = {"height": height}
    if title:
        props["title"] = alt.TitleParams(
            text=title, subtitle=subtitle or "", anchor="start", offset=12
        )
    return (
        chart.properties(**props)
        .configure(background=SURFACE)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=GRID,
            gridWidth=1,
            domainColor=AXIS,
            tickColor=AXIS,
            labelColor=INK2,
            titleColor=INK2,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
            titlePadding=10,
            labelPadding=6,
        )
        .configure_legend(
            labelColor=INK2,
            labelFontSize=12,
            symbolSize=110,
            symbolOpacity=1,
            labelLimit=420,
            orient="top",
            direction="horizontal",
            title=None,
            padding=0,
            offset=8,
            columnPadding=18,
        )
        .configure_title(
            color=INK,
            fontSize=16,
            fontWeight=600,
            subtitleColor=INK2,
            subtitleFontSize=12,
            subtitlePadding=6,
        )
    )


def _color(domain: list[str], colors: list[str], legend: bool = True) -> alt.Color:
    return alt.Color(
        "series:N",
        scale=alt.Scale(domain=domain, range=colors),
        legend=alt.Legend(title=None, symbolType="square") if legend else None,
    )


def _months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + b.month - a.month + 1


def alert_chart(
    d: pd.DataFrame,
    *,
    title: str,
    subtitle: str,
    y_label: str,
    holdout_start: pd.Timestamp,
    height: int = 460,
    width_px: int = 950,
) -> alt.LayerChart:
    """Observed bars, expected line, 99% upper band, alert months and the shaded holdout period."""
    d = d.sort_values("month").reset_index(drop=True)
    first, last = d["month"].min(), d["month"].max()
    n = _months_between(first, last)
    size = float(np.clip(0.8 * width_px / max(n, 1), 2, 18))
    tested = d[d["mu"].notna()]
    top = max(d["y"].max(), 1.0)
    cap = tested["upper99"].clip(upper=1.3 * top) if len(tested) else tested["upper99"]
    ymax = max(top, cap.max() if len(cap) else 0.0, 3.0) * 1.06

    kinds = ["Observed count", "Alert month", "Expected count", "99% upper limit"]
    color = _color(kinds, [BLUE, ORANGE, INK2, GREY])
    xscale = alt.Scale(domain=[first - HALF_MONTH, last + HALF_MONTH])
    x = alt.X(
        "month:T",
        title="Month",
        scale=xscale,
        axis=alt.Axis(
            format="%Y", grid=False, labelAngle=0, tickCount="year", labelOverlap=True
        ),
    )
    x_plain = alt.X("month:T", scale=xscale)
    yscale = alt.Scale(domain=[0, ymax], nice=True, zero=True)
    y = alt.Y(
        "y:Q",
        title=y_label,
        scale=yscale,
        axis=alt.Axis(labelExpr=COUNT_LABELS, tickCount=6),
    )

    bars = d.assign(series=np.where(d["alert"], "Alert month", "Observed count"))
    tip = [
        alt.Tooltip("month:T", title="Month", format="%b %Y"),
        alt.Tooltip("y:Q", title="Observed", format=",.0f"),
        alt.Tooltip("mu:Q", title="Expected", format=",.1f"),
        alt.Tooltip("upper99:Q", title="99% upper limit", format=",.0f"),
        alt.Tooltip("ratio:Q", title="Observed / expected", format=",.1f"),
        alt.Tooltip("p:Q", title="Chance of a count this high", format=".3g"),
        alt.Tooltip("status_label:N", title="Status"),
    ]
    hover = alt.selection_point(
        name="hover",
        fields=["month"],
        on="pointerover",
        clear="pointerout",
        empty=False,
    )
    hit = (
        alt.Chart(
            d.assign(
                status_label=np.where(
                    d["alert"],
                    "Alert month",
                    d["status"]
                    .map(
                        {
                            "tested": "Tested, no alert",
                            "first_occurrence": "First occurrence",
                            "reemergence": "Re-emergence",
                        }
                    )
                    .fillna(""),
                )
            )
        )
        .mark_rule(strokeWidth=max(2.0, width_px / max(n, 1)), color=GREY)
        .encode(
            x=x_plain,
            opacity=alt.condition(hover, alt.value(0.28), alt.value(0)),
            tooltip=tip,
        )
        .add_params(hover)
    )

    layers = []
    if last >= holdout_start >= first - HALF_MONTH:
        hold = pd.DataFrame(
            {
                "start": [holdout_start - HALF_MONTH],
                "end": [last + HALF_MONTH],
                "label": [f"Holdout (from {holdout_start:%b %Y})"],
            }
        )
        layers.append(
            alt.Chart(hold)
            .mark_rect(color=GREY, opacity=0.13)
            .encode(x=alt.X("start:T", scale=xscale), x2="end:T")
        )
        layers.append(
            alt.Chart(hold)
            .mark_text(
                align="right", baseline="top", dx=-6, dy=-2, color=INK2, fontSize=12
            )
            .encode(
                x=alt.X("end:T", scale=xscale),
                y=alt.value(4),
                text="label:N",
            )
        )
    if len(tested):
        band = tested.assign(series="99% upper limit")
        layers.append(
            alt.Chart(band)
            .mark_area(opacity=0.22, clip=True)
            .encode(
                x=x_plain,
                y=alt.Y("mu:Q", scale=yscale),
                y2="upper99:Q",
                color=color,
            )
        )
        layers.append(
            alt.Chart(band)
            .mark_line(strokeWidth=1, clip=True)
            .encode(x=x_plain, y=alt.Y("upper99:Q", scale=yscale), color=color)
        )
    layers.append(
        alt.Chart(bars)
        .mark_bar(
            size=size,
            cornerRadiusTopLeft=min(3, size / 2),
            cornerRadiusTopRight=min(3, size / 2),
        )
        .encode(x=x, y=y, color=color)
    )
    if len(tested):
        layers.append(
            alt.Chart(tested.assign(series="Expected count"))
            .mark_line(strokeWidth=2, clip=True)
            .encode(x=x_plain, y=alt.Y("mu:Q", scale=yscale), color=color)
        )
    alerts = bars[bars["alert"]]
    if len(alerts):
        layers.append(
            alt.Chart(alerts.assign(series="Alert month"))
            .mark_point(
                shape="triangle-down",
                filled=True,
                size=120,
                opacity=1,
                stroke=SURFACE,
                strokeWidth=2,
                dy=-10,
            )
            .encode(x=x_plain, y=alt.Y("y:Q", scale=yscale), color=color)
        )
    layers.append(hit)
    return finish(alt.layer(*layers), title, subtitle, height)


def forecast_chart(
    hist: pd.DataFrame | None,
    row: pd.Series,
    *,
    title: str,
    subtitle: str,
    y_label: str,
    height: int = 460,
) -> alt.LayerChart:
    """Last observed months plus the next-month expectation and its 80/95% intervals for both models."""
    fm = pd.Timestamp(row["forecast_month"])
    models = [
        (
            "LSTM",
            "expected_lstm",
            "lo80_lstm",
            "hi80_lstm",
            "lo95_lstm",
            "hi95_lstm",
            ORANGE,
            -12,
            "circle",
        ),
        ("GBM", "expected", "lo80", "hi80", "lo95", "hi95", GREY, 12, "diamond"),
    ]
    fc = pd.DataFrame(
        [
            {
                "series": f"{name} forecast",
                "x": fm + pd.Timedelta(days=off),
                "expected": row[e],
                "lo80": row[l8],
                "hi80": row[h8],
                "lo95": row[l95],
                "hi95": row[h95],
                "label": f"{name} {row[e]:,.0f}"
                if row[e] >= 10
                else f"{name} {row[e]:,.1f}",
            }
            for name, e, l8, h8, l95, h95, _c, off, _s in models
        ]
    )
    hist = hist if hist is not None else pd.DataFrame({"month": [], "y": []})
    start = (
        (pd.Timestamp(row["last_month"]) - pd.DateOffset(months=35))
        if len(hist) == 0
        else hist["month"].min()
    )
    domain = [start - HALF_MONTH, fm + pd.DateOffset(months=3)]
    ymax = (
        max(float(hist["y"].max()) if len(hist) else 0.0, float(fc["hi95"].max()), 3.0)
        * 1.06
    )
    yscale = alt.Scale(domain=[0, ymax], nice=True, zero=True)
    names = ["Observed count", "LSTM forecast", "GBM forecast"]
    color = _color(names, [BLUE, ORANGE, GREY])
    x = alt.X(
        "month:T",
        title="Month",
        scale=alt.Scale(domain=domain),
        axis=alt.Axis(
            format="%b %Y", grid=False, labelAngle=0, tickCount=8, labelOverlap=True
        ),
    )
    xf = alt.X("x:T", scale=alt.Scale(domain=domain))
    y = alt.Y(
        "y:Q",
        title=y_label,
        scale=yscale,
        axis=alt.Axis(labelExpr=COUNT_LABELS, tickCount=6),
    )

    shade = pd.DataFrame({"start": [fm - HALF_MONTH], "end": [fm + HALF_MONTH]})
    layers = [
        alt.Chart(shade)
        .mark_rect(color=GREY, opacity=0.13)
        .encode(x=alt.X("start:T", scale=alt.Scale(domain=domain)), x2="end:T")
    ]
    if len(hist):
        layers.append(
            alt.Chart(hist.assign(series="Observed count"))
            .mark_bar(size=11, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=x,
                y=y,
                color=color,
                tooltip=[
                    alt.Tooltip("month:T", title="Month", format="%b %Y"),
                    alt.Tooltip("y:Q", title="Observed", format=",.0f"),
                ],
            )
        )
    tip = [
        alt.Tooltip("series:N", title="Model"),
        alt.Tooltip("expected:Q", title="Expected next month", format=",.1f"),
        alt.Tooltip("lo80:Q", title="80% range from", format=",.0f"),
        alt.Tooltip("hi80:Q", title="to", format=",.0f"),
        alt.Tooltip("lo95:Q", title="95% range from", format=",.0f"),
        alt.Tooltip("hi95:Q", title="to", format=",.0f"),
    ]
    yf = lambda f: alt.Y(f, scale=yscale)
    layers += [
        alt.Chart(fc)
        .mark_rule(strokeWidth=1.5)
        .encode(x=xf, y=yf("lo95:Q"), y2="hi95:Q", color=color, tooltip=tip),
        alt.Chart(fc)
        .mark_rule(strokeWidth=6)
        .encode(x=xf, y=yf("lo80:Q"), y2="hi80:Q", color=color, tooltip=tip),
        alt.Chart(fc)
        .mark_point(filled=True, size=130, opacity=1, stroke=SURFACE, strokeWidth=2)
        .encode(
            x=xf,
            y=yf("expected:Q"),
            color=color,
            shape=alt.Shape(
                "series:N",
                legend=None,
                scale=alt.Scale(
                    domain=["LSTM forecast", "GBM forecast"],
                    range=["circle", "diamond"],
                ),
            ),
            tooltip=tip,
        ),
    ]
    xl = alt.X("xl:T", scale=alt.Scale(domain=domain))
    for model, dy in (("LSTM forecast", 9), ("GBM forecast", -9)):
        one = fc[fc["series"] == model].assign(xl=fm + pd.Timedelta(days=12))
        for halo in (
            True,
            False,
        ):  # a surface-coloured halo keeps the label legible over bars
            layers.append(
                alt.Chart(one)
                .mark_text(
                    align="left",
                    dx=16,
                    dy=dy,
                    fontSize=12,
                    color=SURFACE if halo else INK2,
                    stroke=SURFACE if halo else None,
                    strokeWidth=4 if halo else 0,
                )
                .encode(x=xl, y=yf("expected:Q"), text="label:N")
            )
    return finish(alt.layer(*layers), title, subtitle, height)


def reporting_chart(
    r: pd.DataFrame,
    column: str,
    label: str,
    changes: list[pd.Timestamp],
    *,
    title: str,
    subtitle: str = "",
    height: int = 360,
) -> alt.LayerChart:
    """Monthly reporting intensity with the detected level shifts as thin vertical rules."""
    d = r[["month", column]].rename(columns={column: "value"}).assign(series=label)
    x = alt.X(
        "month:T",
        title="Month",
        axis=alt.Axis(
            format="%Y", grid=False, labelAngle=0, tickCount="year", labelOverlap=True
        ),
    )
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=[label, "Level shift"], range=[BLUE, ORANGE]),
        legend=alt.Legend(title=None, symbolType="square"),
    )
    line = (
        alt.Chart(d)
        .mark_line(strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y(
                "value:Q",
                title=label,
                scale=alt.Scale(domain=[0, float(d["value"].max()) * 1.15], nice=True),
                axis=alt.Axis(labelExpr=COUNT_LABELS, tickCount=6),
            ),
            color=color,
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%b %Y"),
                alt.Tooltip("value:Q", title=label, format=",.0f"),
            ],
        )
    )
    ch = pd.DataFrame(
        {
            "month": changes,
            "series": "Level shift",
            "text": [f"{c:%b %Y}" for c in changes],
        }
    )
    rules = (
        alt.Chart(ch)
        .mark_rule(strokeWidth=1.5, opacity=0.9)
        .encode(
            x="month:T",
            color=color,
            tooltip=[alt.Tooltip("month:T", title="Level shift", format="%b %Y")],
        )
    )
    text = (
        alt.Chart(ch)
        .mark_text(align="left", baseline="top", dx=4, dy=2, color=INK2, fontSize=12)
        .encode(x="month:T", y=alt.value(2), text="text:N")
    )
    return finish(alt.layer(rules, line, text), title, subtitle, height)


def trend_chart(d: pd.DataFrame, *, title: str, subtitle: str = "") -> alt.LayerChart:
    """Per-series change per year (% of the median), raw counts versus share of all notifications."""
    order = d.drop_duplicates("label")["label"].tolist()
    measures = ["Raw annual counts", "Share of all notifications"]
    color = alt.Color(
        "measure:N",
        scale=alt.Scale(domain=measures, range=[BLUE, ORANGE]),
        legend=alt.Legend(title=None, symbolType="circle"),
    )
    fillo = alt.FillOpacity(
        "trend:N",
        scale=alt.Scale(domain=["Clear trend", "No clear trend"], range=[1, 0]),
        legend=alt.Legend(
            title=None,
            symbolType="circle",
            symbolStrokeColor=INK2,
            symbolFillColor=INK2,
        ),
    )
    y = alt.Y(
        "label:N",
        sort=order,
        title=None,
        axis=alt.Axis(labelLimit=330, grid=True, domain=False),
    )
    yo = alt.YOffset("measure:N", scale=alt.Scale(domain=measures, paddingOuter=0.2))
    xt = "Change per year (% of the series' median)"
    tip = [
        alt.Tooltip("full:N", title="Disease"),
        alt.Tooltip("area_name:N", title="Area"),
        alt.Tooltip("measure:N", title="Measure"),
        alt.Tooltip("pct:Q", title="Change per year, %", format="+.1f"),
        alt.Tooltip("lo:Q", title="95% interval from, %", format="+.1f"),
        alt.Tooltip("hi:Q", title="to, %", format="+.1f"),
        alt.Tooltip("q:Q", title="q-value", format=".3f"),
        alt.Tooltip("trend:N", title="Verdict"),
    ]
    zero = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_rule(color=INK2, strokeWidth=1)
        .encode(x="x:Q")
    )
    lo_lim = min(d["pct"].min(), d["lo"].quantile(0.25), 0.0)
    hi_lim = max(d["pct"].max(), d["hi"].quantile(0.75), 0.0)
    pad = 0.04 * (hi_lim - lo_lim)  # very long interval bars are cut at the edge
    xs = alt.Scale(domain=[lo_lim - pad, hi_lim + pad], nice=True)
    rules = (
        alt.Chart(d)
        .mark_rule(strokeWidth=2, opacity=0.45, clip=True)
        .encode(
            x=alt.X("lo:Q", title=xt, scale=xs, axis=alt.Axis(grid=True)),
            x2="hi:Q",
            y=y,
            yOffset=yo,
            color=color,
            tooltip=tip,
        )
    )
    dots = (
        alt.Chart(d)
        .mark_point(filled=True, size=90, strokeWidth=2, opacity=1)
        .encode(
            x=alt.X("pct:Q", scale=xs),
            y=y,
            yOffset=yo,
            color=color,
            fillOpacity=fillo,
            stroke=color,
            tooltip=tip,
        )
    )
    return finish(alt.layer(zero, rules, dots), title, subtitle, 30 * len(order) + 170)


def curve_chart(
    d: pd.DataFrame,
    *,
    x_title: str,
    y_domain: tuple[float, float],
    title: str,
    height: int = 300,
) -> alt.LayerChart:
    """Rate ratio against weather anomaly with a 95% band (one lag)."""
    x = alt.X(
        "anomaly_sd:Q",
        title=x_title,
        scale=alt.Scale(domain=[-3, 3], nice=False),
        axis=alt.Axis(values=list(range(-3, 4)), grid=False),
    )
    ys = alt.Scale(type="log", domain=list(y_domain))
    y = alt.Y(
        "irr:Q",
        title="Rate ratio (1 = no change)",
        scale=ys,
        axis=alt.Axis(values=[0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4], format="~g"),
    )
    one = (
        alt.Chart(pd.DataFrame({"y": [1.0]}))
        .mark_rule(color=GREY, strokeWidth=1.5)
        .encode(y=alt.Y("y:Q", scale=ys))
    )
    band = (
        alt.Chart(d)
        .mark_area(color=BLUE, opacity=0.16, clip=True)
        .encode(x=x, y=alt.Y("lo:Q", scale=ys), y2="hi:Q")
    )
    line = alt.Chart(d).mark_line(color=BLUE, strokeWidth=2, clip=True).encode(x=x, y=y)
    nearest = alt.selection_point(
        name="near",
        fields=["anomaly_sd"],
        nearest=True,
        on="pointerover",
        clear="pointerout",
        empty=False,
    )
    tip = [
        alt.Tooltip("anomaly_sd:Q", title="Anomaly (SD)", format="+.1f"),
        alt.Tooltip("irr:Q", title="Rate ratio", format=".2f"),
        alt.Tooltip("lo:Q", title="95% band from", format=".2f"),
        alt.Tooltip("hi:Q", title="to", format=".2f"),
    ]
    pick = (
        alt.Chart(d)
        .mark_point(color=BLUE, size=80, filled=True, stroke=SURFACE, strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("irr:Q", scale=ys),
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=tip,
        )
        .add_params(nearest)
    )
    return finish(alt.layer(band, one, line, pick), title, None, height)


def gain_chart(
    d: pd.DataFrame, *, x_title: str, title: str, subtitle: str = ""
) -> alt.LayerChart:
    """Forest-style chart: gain in mean log score with its 95% interval, one row per comparison."""
    order = d["label"].tolist()
    states = ["Interval includes zero", "Interval excludes zero"]
    color = alt.Color(
        "verdict:N",
        scale=alt.Scale(domain=states, range=[BLUE, ORANGE]),
        legend=alt.Legend(title=None, symbolType="circle"),
    )
    y = alt.Y(
        "label:N",
        sort=order,
        title=None,
        axis=alt.Axis(labelLimit=420, grid=True, domain=False),
    )
    tip = [
        alt.Tooltip("label:N", title="Comparison"),
        alt.Tooltip("gain:Q", title="Gain", format="+.4f"),
        alt.Tooltip("lo:Q", title="95% interval from", format="+.4f"),
        alt.Tooltip("hi:Q", title="to", format="+.4f"),
        alt.Tooltip("rows:Q", title="Rows scored", format=",.0f"),
    ]
    zero = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_rule(color=INK2, strokeWidth=1)
        .encode(x="x:Q")
    )
    rules = (
        alt.Chart(d)
        .mark_rule(strokeWidth=2)
        .encode(
            x=alt.X("lo:Q", title=x_title, axis=alt.Axis(grid=True)),
            x2="hi:Q",
            y=y,
            color=color,
            tooltip=tip,
        )
    )
    dots = (
        alt.Chart(d)
        .mark_point(filled=True, size=90, opacity=1, stroke=SURFACE, strokeWidth=2)
        .encode(x="gain:Q", y=y, color=color, tooltip=tip)
    )
    return finish(alt.layer(zero, rules, dots), title, subtitle, 46 * len(order) + 150)


def episodes_chart(
    d: pd.DataFrame, *, title: str, subtitle: str = ""
) -> alt.LayerChart:
    """Horizontal bars of the excess count of the largest episodes."""
    order = d["label"].tolist()
    y = alt.Y(
        "label:N",
        sort=order,
        title=None,
        axis=alt.Axis(labelLimit=460, grid=False, domain=False, ticks=False),
    )
    tip = [
        alt.Tooltip("disease_full:N", title="Disease"),
        alt.Tooltip("area_name:N", title="Area"),
        alt.Tooltip("start:T", title="Start", format="%b %Y"),
        alt.Tooltip("end:T", title="End", format="%b %Y"),
        alt.Tooltip("n_alert_months:Q", title="Alert months"),
        alt.Tooltip("peak_y:Q", title="Peak monthly count", format=",.0f"),
        alt.Tooltip("excess:Q", title="Above expected (total)", format=",.0f"),
    ]
    bars = (
        alt.Chart(d)
        .mark_bar(
            size=14, color=BLUE, cornerRadiusTopRight=3, cornerRadiusBottomRight=3
        )
        .encode(
            x=alt.X(
                "excess:Q",
                title="Count above expected during the episode",
                axis=alt.Axis(labelExpr=COUNT_LABELS),
            ),
            y=y,
            tooltip=tip,
        )
    )
    text = (
        alt.Chart(d)
        .mark_text(align="left", dx=6, color=INK2, fontSize=12)
        .encode(x="excess:Q", y=y, text=alt.Text("excess:Q", format=",.0f"))
    )
    return finish(alt.layer(bars, text), title, subtitle, 30 * len(order) + 90)
