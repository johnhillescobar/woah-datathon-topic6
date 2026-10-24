# WOAH Datathon - Topic 6: Trends in animal disease occurrence

Retrospective statistical analysis of WAHIS outbreak notifications (2005 - late 2024) that
(1) defines what is *normal* and *expected* for a disease, (2) flags *significant variations*,
(3) describes long-run *trends*, and (4) tests whether weather anomalies (WorldClim) are associated
with disease events, as a basis for early warning.

> Status: work in progress. Sections marked **(planned)** are not implemented yet.

## Quick start

```bash
uv sync                          # Python 3.13 environment from uv.lock (core, no torch)
uv sync --extra deep             # optional: adds PyTorch (~500 MB), needed only for the LSTM
uv run python -m t6.fetch        # download WAHIS INFUR extract + WorldClim rasters (~2 GB) into data/raw
uv run python -m t6.ingest       # Excel -> typed outbreak / event tables (data/interim)
uv run python -m t6.climate      # sample WorldClim at outbreak locations, compute monthly anomalies
uv run python -m t6.exposure     # country-level climate exposure
uv run python -m t6.detect       # anomaly detection, two variants -> data/processed, reports/tables
uv run python -m t6.trends       # STL, Mann-Kendall / Theil-Sen, change points -> reports/
uv run python -m t6.simulate     # simulation benchmark of the detectors -> reports/tables
uv run python -m t6.deep gbm     # pooled GBM benchmark (seconds); `t6.deep` alone adds the LSTM (~10 min, needs torch)
uv run python -m t6.compare      # score every model on the same rows -> reports/tables
uv run python -m t6.companion    # "expected next month" forecasts (GBM; add --model lstm)
uv run python -m t6.nbmodels     # pooled negative-binomial regressions + climate association
uv run python -m t6.gam          # penalised-spline (GAM) version of the climate test
uv run pytest                    # unit tests and simulated benchmarks
uv run python -m t6.build_report # technical report -> reports/technical_report.html and .pdf (PDF needs Edge or Chrome)
```

`data/` is git-ignored: everything in it is re-created by the commands above. If the organisers'
download URL (`t6/config.py: INFUR_URL`) ever changes, place the INFUR workbook at
`data/raw/infur/infur_20260702.xlsx` and run the remaining steps.

## Data

| Source | Use | Notes |
| --- | --- | --- |
| WAHIS INFUR extract (immediate notifications + follow-up reports), 2005-01-01 to 2026-07-02 | outbreak / event counts, locations, dates | 223,773 rows = report x outbreak x species; 188,464 outbreaks, 6,625 events, 147 disease names (248 WAHIS ids), 190 countries |
| WorldClim 2.1 historical monthly weather (CRU-TS 4.09 downscaled), 1990-2024, 10 arc-min | tmin, tmax, precipitation at outbreak locations | 1991-2020 is the climatology; the series ends 2024-12 |

**Required citation (WOAH data licence):** "World Organisation for Animal Health (WOAH) (2026) -
WAHIS periodical extraction of early warning outbreak data. Retrieved on 2026-10-23. Data extracted
by Latin Team."

What the INFUR data is *not*: it contains immediate notifications and follow-up reports of
exceptional events, not endemic six-monthly reporting. "Trend" therefore means trend in *notified
events*, which mixes real disease dynamics with reporting behaviour (see Trends below).

## Pipeline and layout

```text
t6/config.py     paths, URLs, analysis constants (single place to change a parameter)
t6/fetch.py      download raw data (idempotent)
t6/ingest.py     clean + aggregate: outbreaks.parquet, events.parquet
t6/climate.py    outbreak -> grid cell, monthly tmin/tmax/prec and anomalies
t6/exposure.py   country-level climate exposure (mean anomaly over the notification footprint)
t6/series.py     monthly count series per disease x (global | WOAH region | country)
t6/baseline.py   Farrington-style quasi-Poisson baseline, negative-binomial predictive distribution
t6/detect.py     run the baseline on every eligible series, FDR control, episodes, diagnostics
t6/simulate.py   simulation benchmark: false alarms and detection probability, known truth
t6/trends.py     STL, Mann-Kendall / Theil-Sen, change points, reporting-adjusted trends
t6/deep.py       pooled one-step-ahead models: LightGBM and LSTM (rolling origin)
t6/compare.py    common-support scoring of all models, paired bootstrap by series
t6/companion.py  "expected next month" layer (GBM; LSTM as adopted add-on)
t6/nbmodels.py   pooled negative-binomial regressions, climate association, placebo test
t6/gam.py        penalised-spline negative-binomial regression (nonlinear weather effects)
t6/build_report.py  technical report: Markdown -> self-contained HTML and PDF
tests/           unit tests and simulated benchmarks
reports/         technical_report.md/.html/.pdf, tables/ (results), figures/, dl_evaluation.md (model benchmark)
charts/          the exploratory charts saved by notebooks/01_eda.ipynb
notebooks/       01_eda.ipynb / .html: describe() and charts for every dataset
```

## Anomaly detection ("what is normal, what is a significant variation")

For each series (a disease in one area) and each month *t*, using only data before *t*:

1. Fit a log-linear quasi-Poisson model on the trailing 60 months: intercept, optional linear
   trend, 0-2 annual harmonics. Complexity follows the information in the window, counting
   non-zero months as well as the total.
2. **Season screen:** compare every past month with the median of the same season (calendar month
   +/- 1) across the window and down-weight months above 3x that median. Recurring seasonal peaks
   stay in the baseline; a one-off wave does not. Then Farrington reweighting (standardised
   residual > 2.58) and a refit. A fit that looks unstable is simplified step by step (drop lag,
   trend, harmonics).
3. Predictive distribution for month *t*: negative binomial with variance
   `phi * mu + mu^2 * se(eta)^2` (overdispersion plus parameter uncertainty).
4. p-value = P(Y >= observed). Benjamini-Hochberg FDR (q = 0.05) over all tests of a metric.
5. An **alert** needs FDR-significance, an absolute count >= 3 (outbreaks) / 2 (events), and
   observed / expected >= 2, to suppress statistically significant but trivial deviations.
6. Consecutive alert months (gap <= 1 month) form one **episode**.
7. First-ever and re-appearing occurrences (no baseline to compare with) are listed separately as
   `first_occurrence` / `reemergence`; they are not statistical alerts.

Two detector variants, run side by side (`--models baseline ar1`):

| Variant | What a flagged month means | Use |
| --- | --- | --- |
| `baseline` | the month is high versus the endemic, seasonal norm; a whole wave stays flagged | how big and how long was the variation |
| `ar1` | the baseline plus last month's count (log-linear autoregression): the month is a *surprise given last month*; the lag term is only fitted where the window has >= 12 non-zero months and 36 months of history, so in 51 % of tested months it equals `baseline` | intended for onset detection (timing not evaluated separately) |

Two count metrics are analysed: `outbreaks` (intensity, dominated by a few high-volume reporters) and
`events` (new epidemiological events, sparser but less sensitive to reporting style).

### Validation

**Unit tests (`pytest`, 27):** the IRLS fit equals a reference statsmodels GLM; false-alarm rate on a
simulated seasonal negative-binomial null is below 2% at a nominal 1%; a 6x injected outbreak is
detected; a one-off wave inflates next year's expected count 391-fold without the season screen
and 1.07-fold with it; a peak that recurs every year stays in the baseline; a window with two
non-zero months in 60 no longer produces an exploding fit; the pooled models cannot see the
future; NB fits recover known coefficients and the placebo shuffle preserves each country's data.

**Simulation benchmark (`t6.simulate`, in the spirit of Noufaily et al. 2013):** 720 simulated
20-year seasonal overdispersed series with two historical waves each; one outbreak of known size
(3 months) injected into the test period. Mean over three baseline levels:

| Detector | Detected, 2.5x outbreak | 4x | 8x | False alerts per 100 outbreak-free months |
| --- | --- | --- | --- | --- |
| legacy (no season screen) | 0.33 | 0.61 | 0.89 | 0.46 |
| baseline (screened) | 0.48 | 0.79 | 1.00 | 0.57 |
| ar1 | 0.46 | 0.77 | 1.00 | 0.56 |

The season screen raises detection by 15 points at 2.5x and 18 at 4x (11 at 8x) for 0.11 more false
alerts per 100 outbreak-free months (the total rate, 0.57 per 100 months, is about one per 15 years per series). The simulated outbreak-free series have no
persistence, so they do not show the benefit of the autoregressive term; see the real-data
diagnostics below.

**Real data, outbreaks** (465 of 486 eligible series have tested months; 45,242 tests, scoring 2007-01 to 2026-02):

| | baseline | ar1 |
| --- | --- | --- |
| Alerts (retrospective / 2025-26 holdout) | 1,926 / 284 | 1,507 / 233 |
| Episodes | 924 | 949 |
| Alert months per episode (mean, 90th pct) | 2.4, 5 | 1.8, 4 |
| Series failing Ljung-Box (12 lags, p < 0.05) on residuals | 50 % | 43 % |
| Sd of Dunn-Smyth residuals (1 = calibrated) | 1.50 | 1.40 |

Real-data exceedance frequencies are not false-alarm rates (genuine anomalies are in them);
the simulation above is the calibration evidence. Persistence is real, which is why alerts are
grouped into episodes and why the autoregressive variant exists. Top global episodes match known
events: bluetongue in Europe (2007, 2014, 2024), HPAI (2015, 2016-17, 2020-22, 2025), lumpy skin
disease (2016), new world screwworm (2024-25), sheep pox and goat pox (2025-26), foot-and-mouth
disease (2025). Events (291 series, 27,600 tests) are sparse: 26 alerts (baseline), 15 (ar1).

**Known weakness.** After a multi-year wave the baseline can still over-predict: a wave that
recurs in several years counts as "normal". Europe HPAI in December 2023: about 1,288 expected,
208 observed. Among rows with an expected count above 20, 12 % had an observed count below
10 % of expected (29 % when the season screen is switched off in the current code).

## Trends ("what has changed over 20 years")

`t6.trends` works on global and WOAH-region series, 2005-2024:

* Robust STL decomposition of log1p monthly counts: trend, seasonal strength, peak month.
* Mann-Kendall test and Theil-Sen slope on annual totals, FDR-adjusted over series, for diseases
  active in at least 10 of the 20 years (and present before 2015).
* The same tests on each disease's **share of all notifications** in its area, which removes the
  general growth in reporting. Because the denominator is dominated by a few diseases (ASF is 39 %
  of outbreaks), shares of other diseases fall mechanically; the `events` metric is the sturdier one.
* PELT change points on the de-seasonalised series.

Findings (outbreaks, 98 established series: 40 global, 58 regional): **23 rise and 1 falls in raw counts, but only
4 rise and 6 fall once reporting growth is removed.** African swine fever is the only global
disease growing clearly faster than overall reporting (+26 % of its median share per year);
for events, HPAI and ASF rise, foot-and-mouth disease, Newcastle disease and classical swine fever
fall in share (regionally, ASF in Europe, classical swine fever in Asia-Pacific and rabies in Africa also rise). Countries notifying per month change level at 2006-02, 2016-06, 2020-09 and 2024-07 (events series: 2006-02, 2016-06, 2020-10, 2024-08)
(`reports/tables/reporting_changepoints.csv`): between 2005-09 and 2020-24 the average number of
countries notifying an event per month rose 2.2x (11.8 to 25.8), new events per month 3.1x (13.2 to
41.7) and outbreaks per month 5.9x (218 to 1,289) (`reports/figures/reporting_intensity.png`), so any change point in a disease series that
coincides with one of these dates should be read as a possible reporting artefact.

Caveats: annual series are short (n = 20) and autocorrelated, so Mann-Kendall p-values are
approximate; slopes are descriptive.

## Forecast models: does LSTM or GBM help? (`reports/dl_evaluation.md`)

A pooled LightGBM and a pooled LSTM (negative-binomial output) were scored against the
statistical models under a rolling-origin, one-step-ahead protocol and a decision rule that was
written down in advance (full protocol, strata and caveats in `reports/dl_evaluation.md`).
Gain in mean negative-binomial log score over the best statistical model (`ar1`), 95 % CI from a
bootstrap over series:

| Period | GBM | LSTM | Residual sd (1 = calibrated) |
| --- | --- | --- | --- |
| 2020-2024 | +0.73 [+0.38, +1.22] | +0.87 [+0.51, +1.36] | 1.03 (statistical models: 1.45-1.60) |
| 2025-2026 holdout | +0.97 [+0.49, +1.66] | +1.11 [+0.63, +1.80] | 1.03-1.07 |

* **Almost all of the gain is in wave months** (94 % of the LSTM's gain over the baseline sits in the
  10 % of months with a count >= 3x expected). On ordinary months the LSTM gains +0.02 and the GBM
  loses 0.07. HPAI and bluetongue supply about 70 % of the gain.
* The statistical baseline is a deliberate *normal* reference that excludes one-off waves, so log
  score favours a forecaster during waves by construction. Both pooled models therefore *supplement*
  the alerts and do not replace them (they cannot explain why a month is flagged).
* **Companion** (`t6.companion`): a forecast of the next month (2026-03) for 481 series with 80 / 95 %
  intervals, plus companion alerts using the project's own alert rule. The LSTM beats the GBM by
  +0.14 [+0.11, +0.16] (2020-24) and passes the adoption rule; its forward forecasts are added as
  `*_lstm` columns. Only about 10 % of baseline alerts coincide with companion alerts: the rest are
  waves already under way, which the companion expects to continue. The GBM's high-volume
  intervals are too wide, and for the largest series the two models differ by up to 2x.

## Weather and disease: negative-binomial regressions and a GAM

`t6.nbmodels` fits pooled negative-binomial regressions on the 319 country-level series with the
same protocol and scoring (own ridge-stabilised IRLS; cluster-robust errors by country). Weather is
the monthly tmin, tmax and precipitation anomaly averaged over the cells where the country has
notified outbreaks, at lags 1-3 months (lag >= 1, so usable for early warning). Gain in mean log
score (2020-2024, country rows), 95 % CI from a bootstrap over series:

| Comparison | Gain |
| --- | --- |
| pooled NB regression (counts only) vs `ar1` | +0.99 [+0.40, +1.89] |
| pooled NB regression vs GBM | -0.049 [-0.081, -0.018] |
| + linear weather terms vs counts only | -0.002 [-0.004, 0.000] |
| + weather, separate for vector-borne diseases | -0.001 [-0.005, +0.002] |
| + GAM (smooth curves per anomaly and lag) | -0.005 [-0.011, -0.000] |
| precipitation lags 1-3, vector-borne diseases only (exploratory) | +0.003 [+0.001, +0.005] |

* **Counts explain far more than weather.** The transparent pooled NB regression is within 0.05 of the
  GBM and is well calibrated (residual sd 0.97); lagged counts carry almost all the signal.
* **In-sample association depends on time effects.** Without year effects the weather block looks
  significant (placebo test p = 0.007); with year fixed effects, which remove the shared trend
  (warming and reporting both rise), it does not (placebo p = 0.20). The placebo shuffles whole
  years of climate within each country, which keeps seasonality and within-year autocorrelation. The
  cluster-robust Wald test (p = 2e-16) is far more confident than the placebo and is not relied on.
* **One pattern survives, exploratory.** For vector-borne diseases wetter-than-usual months precede
  more outbreaks one month later: rate ratio per SD 1.48 (nominal interval 1.20 to 1.82, probably too
  narrow). A placebo test specific to the term gives p = 0.003, and p = 0.017 against the largest of all
  18 terms in each shuffle. The 3-month lag (rate ratio 1.28) does not survive (family-wise p = 0.64),
  nor does any temperature term. It was found after looking at the results, the GAM curve is flatter,
  and its out-of-sample gain is real in sign but tiny (+0.003 log score per month; +0.006 [-0.003,
  +0.015] on vector-borne rows).
* **The GAM finds no threshold effects.** The fitted curves are smooth and nearly linear (about 1.9
  effective degrees of freedom per curve, where 1 is a straight line), the penalty chosen on
  validation data was mostly the strongest, and the model is no better out of sample
  (`reports/figures/gam_climate_effects.png`).

Conclusion so far: at this resolution (country footprint, monthly, 0.5-degree source weather) climate
anomalies add at most a very small amount of predictive value for outbreak counts. A modest
rainfall signal for vector-borne diseases is plausible and worth reporting as a hypothesis, not as an
early-warning tool.

## Time-series treatment

* Seasonality: annual harmonics inside the baseline (not a pre-processing step); STL for description.
* Non-stationarity (waves, reporting growth): trailing window, local trend, season screen and
  reweighting, instead of one global model for 20 years; change points reported, not hidden.
* Autocorrelation: checked, not assumed away. Per series we report the lag-1 autocorrelation and a
  Ljung-Box test of Dunn-Smyth residuals (`reports/tables/series_diagnostics_*.csv`); the `ar1`
  variant and the pooled models model persistence explicitly.
* Validation uses rolling-origin, one-step-ahead prediction (no information from the future enters
  a prediction). 2025-01 onwards is a disease-only holdout (no WorldClim data after 2024-12).
* Weather inference uses year-block placebo tests and cluster-robust errors because both outcomes
  and weather trend over time.

## Decision log

Choices that change results, and why. Change a constant in `t6/config.py` and rerun to test
sensitivity.

1. **`disease_id` is not used as the disease key.** The same disease has several ids in WAHIS
   (bluetongue appears under 20 ids, e.g. 29/32/46; foot-and-mouth and HPAI poultry have two each). Series are keyed by the
   English disease name; HPAI poultry, non-poultry and bovine are merged into `HPAI (all hosts)`
   (`DISEASE_MERGE`) so the 2017 list change does not look like a new disease.
2. **Follow-up reports repeat outbreaks.** 16,234 outbreaks appear in more than one report. Outbreak
   counts use unique `Outbreak_id`; case/death counts use each outbreak x species once, from the
   latest report.
3. **Recent months are excluded from scoring.** 90 % of outbreaks are reported within 104 days
   (95 %: 198 days), so the newest months are incomplete. Months within 4 months of the
   extraction date are dropped (`CENSOR_MONTHS`): scoring ends 2026-02.
4. **Precipitation anomalies are clipped to +/-5 (`PREC_Z_CLIP`).**
   `prec_z = (prec - mean) / max(std, 1 mm)`, with mean and standard deviation of the same calendar
   month over 1991-2020 in that grid cell. 2,163 of 13,915,020 cell-months (0.016 %), in 1,646 of
   33,131 cells, had |z| > 5 (minimum -100.7 before clipping); 43 % of them are in 1990, outside the analysis window, and within
   2005-2024 84 % are in 2021-2024. That is
   consistent with cells whose 1991-2020 precipitation barely varies, which makes any later
   deviation look enormous. Such values reflect a near-zero denominator rather than extreme
   weather and would dominate any regression. Clipping keeps the direction of the anomaly,
   bounds its leverage, and touches almost nothing. Temperature anomalies need no clipping
   (range -13 to +15 degC).
5. **WorldClim resolution 10 arc-min.** CRU-TS, from which the monthly series is downscaled, is
   0.5 degrees natively, so finer rasters add file size, not anomaly information.
   546 outbreaks on no-data (ocean) cells were moved to the nearest land cell (at most 70 cells).
   About 20 % of outbreak coordinates are flagged approximate by WOAH; climate links are
   noisy for those.
6. **Window.** Analysis period 2005-01 to 2024-12 (WorldClim coverage). 2025-01 to 2026-02 is a
   disease-only holdout.
7. **`data/` is git-ignored**; the pipeline recreates it. The raw folder is also excluded from Dropbox
   sync.
8. **pandas, not Polars/PySpark.** The largest table (monthly weather at outbreak cells) is 13.9 M
   rows / 0.5 GB in memory and the whole detector runs in under two minutes.
9. **Season screen in the baseline.** The first version used only residual reweighting. After a
   large wave (global bluetongue, 2014) it predicted ~4,050 outbreaks for August 2015 when 2 were
   observed, because the harmonics fit the one-off wave as a seasonal peak and the dispersion
   exploded. The screen is validated by a unit test and the simulation above.
10. **Two detector variants instead of one.** `baseline` keeps the "how long did the abnormal period
    last" reading, `ar1` models the persistence found in real waves. Results of both are reported.
11. **Reporting-adjusted trends.** Raw counts grow with the amount reported; shares of all
    notifications are reported next to raw trends.
12. **Change-point penalty from the STL remainder variance.** Month-to-month differences gave about
    12 change points per series; the remainder variance gives a median of 2.
13. **Sparse windows.** A window with two non-zero months passed the old "enough data" test (its total
    was 21) and a 4-parameter model fitted to two points produced expected counts of e^20 (about
    0.5 % of tests, silencing real alerts). The complexity rule now counts non-zero months, and an
    unstable fit falls back to a simpler model.
14. **GBM and LSTM are supplements, not replacements** (decision rule fixed before scoring; the "can it
    explain a flag" criterion is a judgement). The LSTM is adopted into the companion, torch is only
    needed for it (`t6.companion` default does not import torch).
15. **Weather inference needs year effects and a placebo.** Climate and outbreaks both trend; naive
    likelihood-ratio and cluster-robust Wald tests are far too optimistic (the placebo statistic has a
    median of 96-136 on 18 degrees of freedom). Whole-year shuffles within each country calibrate it.
16. **Own NB fitting code** (ridge-stabilised IRLS) instead of statsmodels' GLM: the latter did not
    converge on the model with vector-specific weather terms and was 20-90x slower; log-likelihoods
    agree where both converge.
17. **Worker processes run single-threaded BLAS.** Twelve workers with twelve BLAS threads each ran
    several times slower.

## Limitations

* Notified events only; reporting intensity, surveillance effort and trade/political context are
  confounded with disease occurrence. Europe is about 70 % of all rows.
* No true absences: a month without a report is a zero, not proof of no disease.
* Weather is tested at monthly resolution on a country-footprint average of 0.5-degree-resolution
  anomalies, so local microclimate effects are out of reach; weather is only available to 2024-12.
* Animal counts (`cases`, `dead`, `susceptible`) are unreliable (e.g. 51 % of rows have
  no `susceptible` value at all); analyses use counts of outbreaks and events.
* The pooled forecast models lean on a few large diseases (HPAI, bluetongue) and the GBM's intervals
  are too wide for high-volume series.
