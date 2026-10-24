# Does a deep sequence model improve the monthly outbreak-count forecast?

## Question
Does a pooled LSTM (and, as a stronger tabular comparator, a pooled LightGBM) improve the one-step-ahead predictive
distribution of monthly outbreak counts over the project's per-series statistical models, and should it replace them,
supplement them, or be dropped? Statistical models: the Farrington-style `baseline` and its `ar1` variant. A negative result was
acceptable; nothing was tuned after seeing results. All numbers below use the current (bug-fixed) statistical models.

## Protocol (fixed in advance)
- Rolling origin, yearly refit, expanding window, test years 2018-2026 (2026 = Jan-Feb). For test year Y the model is fitted on targets
  up to Dec Y-1; the last 12 of those months are a validation block (early stopping, dispersion) and are not trained on. Each month of
  Y is predicted one step ahead from observations up to the month before. Series-level statistics (mean count, scalers) use training
  months only; a unit test checks that scrambling all counts from month t on leaves the inputs for month t unchanged.
- One pooled model per refit over all 486 series (global, region, country). Samples: every (series, month) at least 12 months after the
  series' first nonzero count, zeros included.
- LSTM: 24 months of log1p counts + calendar sin/cos, static features (level, log1p training mean, months since first count); 1 layer,
  32 units, dropout 0.1, negative-binomial head (log mean, log theta), Adam 1e-3, batch 512, max 40 epochs, patience 5; mean of 3 seeds.
  GBM: Poisson objective, 300 trees, lr 0.05, 15 leaves, min 50 per leaf, lags 1/2/3/6/12, rolling means 3/6/12, same month last year
  (smoothed over +-1 month), calendar month, level, training mean, series age; one NB theta per level by maximum likelihood on the
  validation block.
- Scored rows: the 24,635 months from 2018 on that every model scores and the baseline marks `tested`. Metric: mean negative-binomial log
  score (higher is better). Differences use a paired bootstrap that resamples whole series (2000 draws).
- Decision rule, written down before scoring. **Supplement** only if (a) the CI of the log-score gain over the best statistical model excludes 0 in 2020-2024,
  (b) the gain is not negative in 2025-2026, and (c) the residual sd stays within 0.9-1.3. **Alternative** (replacement) only if
  additionally the gain is not negative on surge rows, in the sparse strata (baseline mean < 0.5 and 0.5-2) and on short histories
  (< 36 months), and the model is not worse on interpretability-critical needs: (i) explaining why a month is flagged, (ii) behaviour on
  short series, (iii) reproducibility. "Not negative" means a point estimate >= 0. The best statistical model is `ar1` in every period.

## Read this first: what the baseline is for
The statistical baseline is deliberately a "normal / endemic reference" that excludes one-off waves from its reference, so that waves
stand out as alerts. Log score rewards a pure forecaster that lets last month's wave inform this month's expectation, so it favours LSTM
and GBM during waves by construction. The surge and sparse-stratum views below keep that trade-off visible; they are not evidence that
the baseline is a worse detector.

## Results (mean log score per month-series; gain = model minus reference)
| 2020-2024 (14,958 rows, 358 series) | log score | gain vs baseline [95% CI] | gain vs best stat (ar1) [95% CI] | win rate vs ar1 | residual sd | share p<0.01 |
|---|---|---|---|---|---|---|
| baseline | -2.73 | - | - | - | 1.60 | 7.5% |
| ar1 | -2.31 | +0.43 [+0.24, +0.66] | - | - | 1.45 | 6.2% |
| GBM | -1.57 | +1.16 [+0.71, +1.72] | +0.73 [+0.38, +1.22] | 59% | 1.03 | 1.9% |
| LSTM | -1.44 | +1.30 [+0.84, +1.85] | +0.87 [+0.51, +1.36] | 75% | 1.03 | 1.7% |

| 2025-2026 holdout (4,044 rows, 317 series) | log score | gain vs baseline [95% CI] | gain vs best stat (ar1) [95% CI] | win rate vs ar1 | residual sd | share p<0.01 |
|---|---|---|---|---|---|---|
| baseline | -3.21 | - | - | - | 1.71 | 9.7% |
| ar1 | -2.90 | +0.31 [+0.07, +0.55] | - | - | 1.60 | 8.5% |
| GBM | -1.93 | +1.28 [+0.75, +2.00] | +0.97 [+0.49, +1.66] | 50% | 1.03 | 1.7% |
| LSTM | -1.80 | +1.42 [+0.88, +2.13] | +1.11 [+0.63, +1.80] | 62% | 1.07 | 2.0% |

2018-2019 (early, little training data): LSTM +0.28 [+0.19, +0.38] and GBM +0.14 [+0.04, +0.24] over ar1. LSTM beats GBM by +0.14 in
every period (2020-2024: +0.136 [+0.113, +0.161]). A nominal 1% tail should hold 1% of months; the statistical models put 4-10% there.
Point accuracy is closer: 2020-2024 MAE is 10.4 baseline, 7.5 ar1, 7.9 GBM, 7.0 LSTM.

## Where the gain is (2020-2024, gain over ar1)
- **Almost all of it is in wave months.** The 9.9% of rows where the count is at least 3x the baseline mean (and >= 3), 1,487 months, give
  +8.6 [+5.2, +14.4] per row for LSTM and +8.0 [+4.6, +13.7] for GBM, and carry 94% of the LSTM's gain over the baseline. On the other
  90% of rows the LSTM gains +0.02 [+0.00, +0.04] and GBM *loses* 0.07 [-0.10, -0.04] (2025-2026: LSTM -0.03, GBM -0.14). Model-free
  surge definitions (count >= 3x the previous 12-month mean, 895 rows; >= 3x last month, 822 rows) give the same picture (+11 and
  +7 per row for LSTM). HPAI supplies 46% and bluetongue 24% of the LSTM's gain over the baseline; the top 1% of rows carry 65%.
- **On onset months** (>= 3x last month) the learned models still gain, where persistence gives no head start: they expect more
  (median mean 1.75 vs 1.06 for ar1) and allow more spread (variance/mean 6.1 vs 2.3). Giving the LSTM ar1's spread keeps about five
  sixths of the gain, so the higher mean matters most.
- **Sparse series:** positive for both (mean < 0.5: LSTM +1.09, GBM +0.99; 0.5-2: +0.59, +0.51; CIs exclude 0). **High volume (> 10):**
  LSTM +0.24 [+0.10, +0.40]; GBM -0.23 [-0.43, -0.01], i.e. worse than ar1 (one theta per level gives it overly wide tails there).
- **Level:** LSTM +0.44 global, +0.42 region, +1.18 country; GBM +0.31, +0.30, +1.04; all CIs exclude 0.
- **Short histories (< 36 months, 711 rows):** LSTM +0.56 [+0.34, +0.82], GBM +0.34 [+0.15, +0.58]; pooling helps rather than hurts.
  In 2025-2026 (196 rows) LSTM +0.30 [+0.05, +0.63], GBM +0.18 [-0.03, +0.48].
- **Not just wider tails:** refitting the statistical models' dispersion (one NB theta per level from the previous year) changes their
  mean log score little (baseline -2.73 to -2.72, ar1 -2.31 to -2.57); the LSTM still beats those versions by +1.28 [+0.88, +1.82] and
  +1.13 [+0.73, +1.69], GBM by +1.15 and +1.00.
- **Sensitivity:** log score floored at -20 per row: LSTM +0.34 [+0.29, +0.41], GBM +0.22 [+0.16, +0.28]. Bootstrapping by disease instead
  of series: LSTM +0.87 [+0.28, +1.42], GBM +0.73 [+0.15, +1.28]. Single LSTM seeds span -1.42 to -1.46 against the 3-seed -1.44.
- **Baseline blow-ups are gone:** no row has a statistical mean above 10x the series' past maximum (82 rows before the fix) and the largest
  baseline mean is 1.8e3, so MAE/RMSE of the statistical models are now sensible.

## Runtime, reproducibility
`python -m t6.deep` takes about 10 minutes of wall time on 12 cores: 27 LSTM fits (9 years x 3 seeds) as 11 single-thread processes
(about 97 CPU-minutes) and 9 GBM fits (about 1 CPU-minute). Fits are cached in `data/processed/deep_cache/`; `python -m t6.compare` takes about
15 s. Seeds are fixed and threads pinned: re-running an LSTM fit and the GBM fits gave bit-identical predictions on this machine. PyTorch
takes 507 MB of the environment (LightGBM 5 MB); the baseline needs only numpy/scipy.

## Caveats
No climate or neighbouring-country covariates. One-step-ahead only. Series are short (about 250 months, at most 98 test months) and pooled
models lean on a few large diseases (HPAI, bluetongue). WAHIS counts contain reporting artefacts (batching, late reports; the last 4
months are dropped) that a forecaster can partly learn and a detector should not. The models do not train on the most recent 12 targets
(validation block), a small handicap to them. Series were selected on full-period totals (>= 30 outbreaks), the same for every model.
Hierarchical series (global/region/country) are dependent; bootstrapping by disease widens the CIs but keeps the conclusion. Counts are
final-vintage, not as first reported.

## Verdict (decision rule applied as written)
- **LSTM: supplement. GBM: supplement. Neither is an alternative, so neither should replace the statistical models.** Both meet all
  three supplement conditions: the gain CI over ar1 excludes 0 in 2020-2024, the holdout gain is positive, and residual sd is 1.03-1.07.
  The replacement conditions on surge rows, sparse strata and short histories are met, but the "explains why a month is flagged" need
  fails (a judgement, not a measurement): the baseline gives a seasonal-trend expectation and an observed/expected ratio; the pooled
  models do not (GBM offers feature attributions at best), and they add a seed-, thread- and library-dependent training pipeline.
- **What changed after the statistical-model fixes:** the verdict is the same. The gains over ar1 are larger (LSTM +0.87, was +0.53; GBM
  +0.73, was +0.40) and more concentrated in wave months (94% of the LSTM's gain over the baseline, was 86%). Outside waves the GBM is now
  clearly no better than ar1 (-0.07 [-0.10, -0.04]) and worse at high volume; the LSTM is level with it (+0.02). Baseline blow-ups no longer
  exist.
- What would change it: (1) climate and neighbouring-country case covariates, which could help predict wave onsets; (2) a longer or less
  wave-dominated sample, where non-wave gains might grow; (3) attributions or reference bands that a reviewer accepts as an explanation,
  which would reopen the "alternative" question.

## Companion: an "expected next month" layer beside the alerts
`python -m t6.companion` (default GBM, no torch) and `python -m t6.companion --model lstm|ensemble` (stage 2) deliver the following.

**What is delivered.**
- `reports/tables/companion_forecast.csv`: for each of 481 series the forecast for 2026-03 from a final fit on all targets through 2026-02
  (same fixed hyperparameters; the GBM then refits on the validation block with the same dispersion, which in the rolling benchmark is
  slightly better, +0.013 [+0.005, +0.023] in 2020-2024). Columns: expected count, 80% and 95% NB intervals, theta, the last observed month
  and its one-step tail p-value, plus `*_lstm` columns for the adopted LSTM.
- `companion_alerts.csv`, `companion_agreement.csv`, `companion_alert_profile.csv`, `companion_coverage.csv`, `companion_stage2.csv`,
  `companion_adoption.csv`, `companion_refit_check.csv`, and figures `reports/figures/companion_series.png` (bluetongue global, HPAI Europe)
  and `companion_agreement.png`.

**How to read "expected next month".** It is the mean of the predictive distribution given counts through February 2026 and the calendar,
not an alert threshold; a wave under way is expected to continue. The intervals are count quantiles (at least 80% / 95% coverage). In the
rolling benchmark the GBM's 80% interval covers 94% of months overall (92% for expected counts above 10), so its high-volume intervals
are too wide; the LSTM's are closer to nominal there (82%). The two models can disagree by up to 2x on the biggest series (global HPAI:
GBM 1,211, LSTM 688, last month 1,315), which is itself a warning to read those intervals loosely. `last_month_p` shows how surprising the
last month was one step earlier (2 series below 0.01 for the GBM).

**Retrospective alerts and agreement with the statistical alerts.** The project's own rule (Benjamini-Hochberg q < 0.05, count >= 3,
observed/expected >= 2; `detect.score`) is applied to the GBM predictions, so the comparison is like for like. That gives 145 companion
alerts from 2018 against 1,336 (baseline) and 1,020 (ar1) statistical alerts.

| Months flagged (all levels) | both | only companion | only statistical |
|---|---|---|---|
| vs baseline, 2018-2019 | 16 | 6 | 190 |
| vs baseline, 2020-2024 | 88 | 7 | 758 |
| vs baseline, 2025-2026 | 26 | 2 | 258 |
| vs ar1, 2018-2019 | 18 | 4 | 151 |
| vs ar1, 2020-2024 | 86 | 9 | 532 |
| vs ar1, 2025-2026 | 27 | 1 | 206 |

The companion is a much stricter filter: about 10% of baseline alerts (13% of ar1's) are also companion alerts, and it adds very few of
its own (the same pattern in every level; per-level counts are in `companion_agreement.csv`). "Only statistical" months are mostly waves
under way: median previous-month count 15, 79% already at 3 or more, 60% already in statistical alert the month before. The companion
expected about a third of the observed count but not by enough to pass its stricter tail test. Months flagged by both are mostly
onsets (median previous month 0, observed about 36x the companion's expectation). The 15 "only companion" months (vs baseline) are of two
kinds. Five are African swine fever months of 2019 (global and Europe) where the statistical model, refit on the last 60 months, already
expected 466-729 outbreaks and 633-832 was normal, while the companion expected about 37: it had not yet seen that regime (its fit
ended in 2017 plus a held-out 2018), so these are companion false alarms. The other ten are sparse-series onsets (observed 4-85
against companion expectations of 0.06-2.8) that the statistical model also found unusual (p 0.003-0.08) but that missed its FDR cut.

**Stage 2: LSTM and ensemble against the GBM companion (rule decided in advance before the ensemble was scored).** Adopt the LSTM or the
ensemble (geometric mean of the two expected counts, NB dispersion from the average of 1/theta) only if, against the GBM on the rolling
predictions, (a) the 95% CI of the mean log-score gain is above 0 in 2020-2024 (paired bootstrap over series), (b) the 2025-2026 point
estimate is not negative, and (c) the residual sd is within 0.9-1.3; if both pass, the ensemble is adopted only if its gain over the LSTM
also has a 95% CI above 0 in 2020-2024, otherwise the LSTM.

| Gain [95% CI] | 2018-2019 | 2020-2024 | 2025-2026 | residual sd (2020-24 / 25-26) |
|---|---|---|---|---|
| LSTM vs GBM | +0.141 [+0.107, +0.181] | +0.136 [+0.113, +0.161] | +0.138 [+0.103, +0.176] | 1.03 / 1.07 |
| ensemble vs GBM | +0.089 [+0.070, +0.111] | +0.097 [+0.084, +0.111] | +0.095 [+0.080, +0.112] | 0.99 / 1.00 |
| ensemble vs LSTM | -0.053 [-0.072, -0.036] | -0.038 [-0.053, -0.025] | -0.043 [-0.070, -0.016] | |

Both pass against the GBM, but the ensemble is worse than the LSTM, so **the LSTM is adopted into the companion**; the ensemble is not.
Forward LSTM forecasts come from three final fits (epochs chosen by early stopping on the last 12 months, then refit on all targets for that
many epochs; about 10 minutes for the 3 seeds in parallel). That refit is not validated in the rolling benchmark for the LSTM (it is for the
GBM); it moves the forecasts little (median log ratio to the evaluated-protocol fit -0.02).

**If torch is not installed.** The default `python -m t6.companion` (GBM forecast, intervals, alerts, agreement, figures) and `t6.compare`
run without it, and `t6.deep gbm` too; torch is imported only inside the LSTM fits. Without torch the LSTM cannot be retrained: the rolling
LSTM predictions (`tests_outbreaks_lstm.parquet`) and the forward fits cached in `data/processed/deep_cache/` can still be read, but
without them the `*_lstm` columns, the narrower and better-calibrated high-volume intervals, and about +0.14 log score per month-series
over the GBM are lost. The GBM companion alone keeps most of the gain over the statistical models (+0.73 vs +0.87 over ar1 in 2020-2024).

Files: `t6/deep.py`, `t6/compare.py`, `t6/companion.py`, `tests/test_compare.py`, `tests/test_companion.py`; tables in `reports/tables/`
(`model_comparison.csv`, `model_decision.csv`, `deep_runs_{lstm,gbm}.csv`, `companion_*.csv`); predictions
`data/processed/tests_outbreaks_{lstm,gbm}.parquet`.
