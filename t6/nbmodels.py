"""Pooled negative-binomial regressions on country-level monthly outbreak counts, with and without
lagged climate anomalies.

Same rolling-origin protocol and scoring as the deep models: for test year Y the model is fitted on
all targets up to December Y-1 and each month of Y is predicted one step ahead. Models (NB2 GLM, log
link, one dispersion fitted by maximum likelihood):

  nb_pooled          counts only: two lags, last year's level, series history, reporting intensity,
                     disease x season
  nb_climate         + tmin / tmax / precipitation anomalies of the country's footprint at lags 1-3
                     (lag >= 1, so usable for early warning)
  nb_climate_vector  the same anomalies, with separate coefficients for vector-borne diseases
  nb_precip_vector   hypothesis-focused: only precipitation at lags 1-3, only for vector-borne diseases
                     (suggested by the association analysis, so exploratory)

Then an association analysis on 2005-2024: cluster-robust (by country) Wald test of the climate block,
incidence-rate ratios per +1 SD anomaly, and a placebo test that shuffles whole years of climate
within each country (keeps seasonality and within-year autocorrelation, breaks the alignment with
disease).

Run with `uv run python -m t6.nbmodels [--skip-placebo] [--association-only] [--skip-association]
[--only MODEL ...]`. Needs t6.exposure.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import special, stats
from statsmodels.stats.multitest import multipletests

from t6 import baseline, compare, deep
from t6 import config as C
from t6.exposure import VARS

OUT = C.PROCESSED / "nb"
TABLES = C.REPORTS / "tables"
LAGS = (1, 2, 3)
LAST_YEAR = 2024  # WorldClim ends here
FIRST_TEST = 2018
N_GROUPS = 12  # the 12 largest diseases get their own seasonality, the rest share one
VECTOR = (
    "Bluetongue",
    "West Nile",
    "Lumpy skin",
    "Rift Valley",
    "Epizootic hemorrhagic",
    "African horse sickness",
    "Schmallenberg",
)
MODELS = {
    "nb_pooled": "none",
    "nb_climate": "main",
    "nb_climate_vector": "split",
    "nb_precip_vector": "precip_vector",
}
PLACEBOS = 300
SEED = 0
RIDGE = (
    1e-2  # tiny L2 on all but the constant: stops near-separated dummies from diverging
)


def load() -> dict:
    keys, Y, idx = deep.load_wide()
    keep = (keys["level"] == "country").to_numpy()
    keys, Y = keys[keep].reset_index(drop=True), Y[keep]
    n_t = int((idx <= pd.Timestamp(f"{LAST_YEAR}-12-01")).sum())
    Y, idx = Y[:, :n_t], idx[:n_t]

    cc = pd.read_parquet(C.INTERIM / "country_climate.parquet")
    fit_end = pd.Timestamp(f"{FIRST_TEST - 1}-12-01")
    clim, scale = {}, {}
    for v in VARS:
        wide = cc.pivot(index="iso_code", columns="month", values=v)
        scale[v] = float(
            wide.loc[:, wide.columns <= fit_end].stack().std()
        )  # before the first test year
        clim[v] = (
            wide.reindex(keys["area"]).reindex(columns=idx).to_numpy() / scale[v]
        ).astype(np.float64)

    pre = Y[:, : (FIRST_TEST - deep.START_YEAR) * 12].sum(1)
    top = keys.assign(n=pre).groupby("disease")["n"].sum().nlargest(N_GROUPS).index
    grp = (
        keys["disease"]
        .map({d: i for i, d in enumerate(top)})
        .fillna(N_GROUPS)
        .astype(int)
        .to_numpy()
    )
    return {
        "keys": keys,
        "Y": Y,
        "idx": idx,
        "fn": deep.first_nonzero(Y),
        "L": deep.padded_log(Y),
        "clim": clim,
        "grp": grp,
        "top": list(top),
        "vec": keys["disease"].str.contains("|".join(VECTOR)).to_numpy().astype(float),
        "rep": np.log1p(
            (Y > 0).sum(0)
        ),  # countries notifying in each month, a reporting-intensity proxy
    }


def climate_names(split: bool) -> list[str]:
    base = [f"{v}_lag{k}" for k in LAGS for v in VARS]
    return [
        f"{n}_{g}" for g in (("other", "vector") if split else ("all",)) for n in base
    ]


def features(
    D: dict, s, t, tm, mode: str, clim: dict | None = None, year_fe: bool = False
):
    """(continuous count features, everything else) for target months t of series s.

    Count features use data up to t-1 only; climate uses months t-1 .. t-3. `year_fe` adds calendar-year
    dummies (association analysis only: they cannot be predicted out of sample).
    """
    L, off, clim = D["L"], deep.SEQ, clim or D["clim"]
    lag = lambda k: L[s, t + off - k]
    ang = 2 * np.pi * (t % 12) / 12
    oh = np.eye(N_GROUPS + 1)[D["grp"][s]]
    cont = np.column_stack(
        [
            lag(1),
            lag(2),
            np.mean([lag(k) for k in range(3, 13)], 0),
            np.mean([lag(11), lag(12), lag(13)], 0),
            np.log1p(tm[s]),
            np.log1p(t - D["fn"][s]),
            D["rep"][t - 1],
        ]
    )
    rest = [oh[:, :-1], oh * np.sin(ang)[:, None], oh * np.cos(ang)[:, None]]
    if year_fe:
        rest.append(np.eye(D["Y"].shape[1] // 12)[t // 12][:, 1:])
    if mode == "precip_vector":
        rest.append(
            np.column_stack([clim["prec_z"][s, t - k] for k in LAGS])
            * D["vec"][s][:, None]
        )
    elif mode != "none":
        c = np.column_stack([clim[v][s, t - k] for k in LAGS for v in VARS])
        if mode == "main":
            rest.append(c)
        else:
            vec = D["vec"][s][:, None]
            rest += [c * (1 - vec), c * vec]
    return cont, np.column_stack(rest)


def design(cont, rest, ref=None):
    """Standardise continuous features with training statistics; add the constant."""
    m, sd = ref if ref is not None else (cont.mean(0), cont.std(0) + 1e-9)
    return np.column_stack([np.ones(len(cont)), (cont - m) / sd, rest]), (m, sd)


def irls_nb(X, y, alpha, iters=100):
    """Ridge-stabilised NB2 (log link, fixed alpha) by IRLS; returns coefficients."""
    pen = np.full(X.shape[1], RIDGE)
    pen[0] = 0.0
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.mean() + 0.5)
    for _ in range(iters):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)
        XtW = X.T * (mu / (1 + alpha * mu))
        new = np.linalg.solve(XtW @ X + np.diag(pen), XtW @ (eta + (y - mu) / mu))
        step = np.clip(new - beta, -3, 3)  # damped steps keep early iterations finite
        beta = beta + step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def nb_llf(X, y, beta, alpha) -> float:
    mu, r = np.exp(np.clip(X @ beta, -20, 20)), 1.0 / alpha
    return float(
        (
            special.gammaln(y + r)
            - special.gammaln(r)
            - special.gammaln(y + 1)
            + r * np.log(r / (r + mu))
            + y * np.log(mu / (r + mu))
        ).sum()
    )


def fit(X, y, rounds=3):
    """Alpha (= 1/theta) by alternating IRLS and maximum likelihood; returns (beta, alpha)."""
    alpha = 1.0
    for _ in range(rounds):
        beta = irls_nb(X, y, alpha)
        alpha = 1.0 / deep.theta_mle(y, np.exp(np.clip(X @ beta, -20, 20)))
    return irls_nb(X, y, alpha), alpha


def cluster_cov(X, y, beta, alpha, groups):
    """Sandwich covariance of beta with scores summed within clusters (countries)."""
    mu = np.exp(np.clip(X @ beta, -20, 20))
    pen = np.full(X.shape[1], RIDGE)
    pen[0] = 0.0
    bread = np.linalg.inv((X.T * (mu / (1 + alpha * mu))) @ X + np.diag(pen))
    scores = X * ((y - mu) / (1 + alpha * mu))[:, None]
    g = np.bincount(groups).size
    cs = np.zeros((g, X.shape[1]))
    np.add.at(cs, groups, scores)
    return bread @ (cs.T @ cs) @ bread * g / (g - 1)


def rolling(D: dict, model: str) -> pd.DataFrame:
    mode, Y, T = MODELS[model], D["Y"], D["Y"].shape[1]
    out = []
    for year in range(FIRST_TEST, LAST_YEAR + 1):
        sp = deep.split(year, T)
        s, t = deep.samples(D["fn"], 0, sp.val_end)
        tm = deep.train_mean(Y, D["fn"], sp.val_end)
        X, ref = design(*features(D, s, t, tm, mode))
        beta, alpha = fit(X, Y[s, t])
        s2, t2 = deep.samples(D["fn"], sp.test[0], sp.test[-1])
        X2, _ = design(*features(D, s2, t2, tm, mode), ref=ref)
        mu = np.clip(np.exp(np.clip(X2 @ beta, -20, 20)), 1e-6, 1e6)
        out.append(
            pd.DataFrame(
                {
                    "level": "country",
                    "disease": D["keys"]["disease"].to_numpy()[s2],
                    "area": D["keys"]["area"].to_numpy()[s2],
                    "month": D["idx"][t2],
                    "y": Y[s2, t2],
                    "mu": mu,
                    "var": mu + alpha * mu**2,
                    "status": "tested",
                    "alpha": alpha,
                }
            )
        )
        print(f"  {model} {year}: n_train={len(s):,} alpha={alpha:.2f}")
    return pd.concat(out, ignore_index=True)


# ---- comparison ----------------------------------------------------------------------------------


def load_reference() -> dict:
    """Core models restricted to country-level rows the baseline tests."""
    keyc = ["disease", "area", "month"]
    base = pd.read_parquet(C.PROCESSED / "tests_outbreaks.parquet")
    base = base[(base["status"] == "tested") & (base["level"] == "country")]
    out = {"baseline": base}
    for name in ("ar1", "gbm", "lstm"):
        f = C.PROCESSED / f"tests_outbreaks_{name}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            out[name] = d[(d["status"] == "tested") & (d["level"] == "country")]
    support = base[keyc]
    return {
        k: v.merge(support, on=keyc)[keyc + ["y", "mu", "var"]] for k, v in out.items()
    }


def compare_models() -> pd.DataFrame:
    keyc = ["disease", "area", "month"]
    preds = load_reference()
    for m in (*MODELS, "nb_gam"):  # nb_gam is written by t6.gam, when it has been run
        if (OUT / f"tests_{m}.parquet").exists():
            preds[m] = pd.read_parquet(OUT / f"tests_{m}.parquet")[
                keyc + ["y", "mu", "var"]
            ]
    # common support: every row all compared models score, 2018-2024
    common = None
    for d in preds.values():
        k = d[keyc].drop_duplicates()
        common = k if common is None else common.merge(k, on=keyc)
    scored = {}
    for name, d in preds.items():
        d = d.merge(common, on=keyc).sort_values(keyc).reset_index(drop=True)
        d["ls"] = compare.log_score(d["y"], d["mu"], d["var"])
        d["z"] = baseline.dunn_smyth(d["y"], d["mu"], d["var"])
        d["g"] = d["disease"] + "|" + d["area"]
        scored[name] = d
    ref = next(iter(scored.values()))
    is_vec = ref["disease"].str.contains("|".join(VECTOR)).to_numpy()
    periods = {
        "2018-2024": ("2018-01-01", "2024-12-31"),
        "2020-2024": ("2020-01-01", "2024-12-31"),
    }
    pairs = [
        ("nb_pooled", "ar1"),
        ("nb_pooled", "gbm"),
        ("nb_climate", "nb_pooled"),
        ("nb_climate_vector", "nb_pooled"),
        ("nb_climate_vector", "nb_climate"),
        ("nb_gam", "nb_pooled"),
        ("nb_gam", "nb_climate"),
        ("nb_precip_vector", "nb_pooled"),
    ]
    pairs = [p for p in pairs if p[0] in scored and p[1] in scored]
    rows = []
    for pname, (lo, hi) in periods.items():
        sel = ((ref["month"] >= lo) & (ref["month"] <= hi)).to_numpy()
        for name, d in scored.items():
            rows.append(
                {
                    "period": pname,
                    "comparison": name,
                    "kind": "level",
                    "n_rows": int(sel.sum()),
                    "mean_log_score": d.loc[sel, "ls"].mean(),
                    "sd_z": d.loc[sel, "z"].std(),
                    "share_p_lt_0.01": float(
                        (
                            baseline.upper_p(
                                d.loc[sel, "y"], d.loc[sel, "mu"], d.loc[sel, "var"]
                            )
                            < 0.01
                        ).mean()
                    ),
                }
            )
        for a, b in pairs:
            full = (scored[a]["ls"] - scored[b]["ls"]).to_numpy()
            for label, mask in (("", sel), (" [vector-borne rows]", sel & is_vec)):
                if mask.sum() < 50 or (label and not b.startswith("nb_pooled")):
                    continue  # the vector-borne view is for models that add weather to nb_pooled
                m, lo_, hi_ = compare.boot_mean(full[mask], ref["g"].to_numpy()[mask])
                rows.append(
                    {
                        "period": pname,
                        "comparison": f"{a} - {b}{label}",
                        "kind": "log_score_gain",
                        "n_rows": int(mask.sum()),
                        "mean_log_score": m,
                        "ci_lo": lo_,
                        "ci_hi": hi_,
                        "win_rate": compare.win_rate(
                            full[mask], ref["g"].to_numpy()[mask]
                        ),
                    }
                )
    return pd.DataFrame(rows)


# ---- association ----------------------------------------------------------------------------------


def _permute_years(D: dict, rng) -> dict:
    """Shuffle whole calendar years of each country's climate (same permutation for all variables)."""
    T = D["Y"].shape[1]
    n_years = T // 12
    out = {v: np.empty_like(D["clim"][v]) for v in VARS}
    for i in range(len(D["keys"])):
        order = rng.permutation(n_years)
        for v in VARS:
            out[v][i] = D["clim"][v][i].reshape(n_years, 12)[order].reshape(-1)
    return out


_D = None


def single_thread_workers() -> None:
    """Call before creating a process pool: workers inherit these, so each runs single-threaded BLAS.

    Twelve workers x twelve BLAS threads oversubscribe the CPU and run several times slower.
    """
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _init(D):
    global _D
    _D = D


def _placebo_llf(args):
    seed, alpha, year_fe = args
    D, T = _D, _D["Y"].shape[1]
    clim = _permute_years(D, np.random.default_rng(seed))
    s, t = deep.samples(D["fn"], 0, T - 1)
    tm = deep.train_mean(D["Y"], D["fn"], T - 1)
    X, _ = design(*features(D, s, t, tm, "split", clim, year_fe))
    y = D["Y"][s, t]
    beta = irls_nb(X, y, alpha)
    return nb_llf(X, y, beta, alpha), beta[-len(climate_names(split=True)) :]


def association(D: dict, placebos: int = PLACEBOS, year_fe: bool = False):
    T = D["Y"].shape[1]
    s, t = deep.samples(D["fn"], 0, T - 1)
    tm = deep.train_mean(D["Y"], D["fn"], T - 1)
    y = D["Y"][s, t]
    groups = pd.factorize(D["keys"]["area"].to_numpy()[s])[0]
    X0, _ = design(*features(D, s, t, tm, "none", year_fe=year_fe))
    X1, _ = design(*features(D, s, t, tm, "split", year_fe=year_fe))
    beta1, alpha = fit(X1, y)
    beta0 = irls_nb(X0, y, alpha)
    names = climate_names(split=True)
    k = len(names)
    cov = cluster_cov(X1, y, beta1, alpha, groups)[-k:, -k:]
    b = beta1[-k:]
    wald = float(b @ np.linalg.pinv(cov) @ b)
    lr = 2 * (nb_llf(X1, y, beta1, alpha) - nb_llf(X0, y, beta0, alpha))
    se = np.sqrt(np.diag(cov))
    coef = pd.DataFrame(
        {
            "term": names,
            "irr_per_sd": np.exp(b),
            "ci_lo": np.exp(b - 1.96 * se),
            "ci_hi": np.exp(b + 1.96 * se),
            "p": 2 * stats.norm.sf(np.abs(b / se)),
        }
    )
    coef[["variable_lag", "disease_group"]] = coef["term"].str.rsplit(
        "_", n=1, expand=True
    )
    coef["q_bh"] = multipletests(coef["p"], method="fdr_bh")[1]

    tests = {
        "n_rows": len(y),
        "n_series": len(D["keys"]),
        "n_countries": int(groups.max() + 1),
        "alpha": alpha,
        "wald_chi2": wald,
        "wald_df": k,
        "wald_p_cluster_robust": float(stats.chi2.sf(wald, k)),
        "lr_stat": float(lr),
    }
    if placebos:
        single_thread_workers()
        with ProcessPoolExecutor(initializer=_init, initargs=(D,)) as ex:
            res = list(
                ex.map(
                    _placebo_llf, [(SEED + b, alpha, year_fe) for b in range(placebos)]
                )
            )
        lr_null = 2 * (np.array([r[0] for r in res]) - nb_llf(X0, y, beta0, alpha))
        null_b = np.array([r[1] for r in res])  # placebos x terms
        z_null = null_b / null_b.std(0)
        z_real = b / null_b.std(0)
        max_null = np.abs(z_null).max(1)
        # empirical two-sided p per term, and family-wise p against the largest |z| over all 18 terms
        coef["placebo_p"] = [
            (1 + (np.abs(null_b[:, j]) >= abs(b[j])).sum()) / (1 + placebos)
            for j in range(k)
        ]
        coef["placebo_p_familywise"] = [
            (1 + (max_null >= abs(z_real[j])).sum()) / (1 + placebos) for j in range(k)
        ]
        tests.update(
            {
                "placebos": placebos,
                "placebo_lr_median": float(np.median(lr_null)),
                "placebo_lr_p95": float(np.percentile(lr_null, 95)),
                "placebo_p": float((1 + (lr_null >= lr).sum()) / (1 + placebos)),
            }
        )
    return coef, pd.Series(tests)


def main(argv: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    D = load()
    print(
        f"{len(D['keys'])} country series, {D['Y'].shape[1]} months; top diseases: {D['top'][:4]} ..."
    )
    pd.set_option("display.width", 220)
    only = (
        [a for a in argv[argv.index("--only") + 1 :] if not a.startswith("--")]
        if "--only" in argv
        else list(MODELS)
    )
    if "--association-only" not in argv:
        for m in only:
            rolling(D, m).to_parquet(OUT / f"tests_{m}.parquet")
        cmp_ = compare_models()
        cmp_.to_csv(TABLES / "nb_experiments.csv", index=False)
        print(cmp_.round(3).to_string(index=False))
    if "--skip-association" in argv:
        return
    placebos = 0 if "--skip-placebo" in argv else PLACEBOS
    coefs, tests = [], {}
    for name, fe in (("no_year_effects", False), ("year_effects", True)):
        coef, t_ = association(D, placebos, fe)
        coefs.append(coef.assign(adjustment=name))
        tests[name] = t_
    pd.concat(coefs).to_csv(TABLES / "climate_association.csv", index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(TABLES / "climate_association_tests.csv")
    print(tests.to_string())
    show = ["adjustment", "term", "irr_per_sd", "ci_lo", "ci_hi", "p", "q_bh"]
    show += [
        c
        for c in ("placebo_p", "placebo_p_familywise")
        if c in pd.concat(coefs).columns
    ]
    print(pd.concat(coefs)[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
