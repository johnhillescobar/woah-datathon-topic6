"""Penalised-spline (GAM) negative-binomial regression: do nonlinear weather effects help?

Same features and rolling-origin protocol as t6.nbmodels, but each weather anomaly (tmin, tmax,
precipitation; lags 1-3) enters as a smooth function: cubic B-spline basis with a second-difference
penalty (P-splines), with separate curves for vector-borne and other diseases. This can pick up
thresholds (heat waves, drought, extreme rain) that a linear term averages away.

The smoothing strength is chosen per test year on an inner validation block (the last 12 months
before the test year), never on test data.

Outputs: data/processed/nb/tests_nb_gam.parquet (compared with the other NB models in
reports/tables/nb_experiments.csv), reports/tables/gam_association*.csv and
reports/figures/gam_climate_effects.png.

Run with `uv run python -m t6.gam [--skip-placebo]` (after t6.nbmodels).
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from t6 import compare, deep
from t6 import config as C
from t6 import nbmodels as N
from t6.exposure import VARS

DEGREE, N_INNER, X_RANGE = 3, 3, 3.0  # 7 basis functions on [-3, 3] SD
LAMBDAS = (1.0, 10.0, 100.0)
PLACEBOS = 100
NB_COLS = N_INNER + DEGREE + 1
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"


# ---- basis and penalty -----------------------------------------------------------------------------


def _knots():
    inner = np.linspace(-X_RANGE, X_RANGE, N_INNER + 2)[1:-1]
    return np.r_[[-X_RANGE] * (DEGREE + 1), inner, [X_RANGE] * (DEGREE + 1)]


def basis(x: np.ndarray) -> np.ndarray:
    """(n, NB_COLS) cubic B-spline design; anomalies beyond +/-3 SD are held at the boundary."""
    x = np.clip(np.asarray(x, float), -X_RANGE, X_RANGE - 1e-9)
    return interpolate.BSpline.design_matrix(x, _knots(), DEGREE).toarray()


def smooth_terms() -> list[str]:
    return [
        f"{v}_lag{k}_{g}" for g in ("other", "vector") for k in N.LAGS for v in VARS
    ]


def smooth_features(D, s, t, clim=None, ref=None):
    """Spline columns for every (lag, variable, disease group); centred to separate them from the constant."""
    clim = clim or D["clim"]
    vec = D["vec"][s][:, None]
    cols = {"other": [], "vector": []}
    for k in N.LAGS:
        for v in VARS:
            B = basis(clim[v][s, t - k])
            cols["other"].append(B * (1 - vec))
            cols["vector"].append(B * vec)
    X = np.column_stack(cols["other"] + cols["vector"])
    ref = X.mean(0) if ref is None else ref
    return X - ref, ref


def penalty(n_base: int, lam: float) -> np.ndarray:
    """Ridge on everything but the constant, plus lam * D2'D2 on every smooth block."""
    n_smooth = len(smooth_terms()) * NB_COLS
    P = np.zeros((n_base + n_smooth,) * 2)
    P[np.arange(1, n_base + n_smooth), np.arange(1, n_base + n_smooth)] = N.RIDGE
    D2 = np.diff(np.eye(NB_COLS), 2, axis=0)
    block = lam * D2.T @ D2
    for j in range(len(smooth_terms())):
        a = n_base + j * NB_COLS
        P[a : a + NB_COLS, a : a + NB_COLS] += block
    return P


# ---- penalised NB fit --------------------------------------------------------------------------------


def irls(X, y, alpha, P, iters=100):
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.mean() + 0.5)
    for _ in range(iters):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)
        XtW = X.T * (mu / (1 + alpha * mu))
        new = np.linalg.solve(XtW @ X + P, XtW @ (eta + (y - mu) / mu))
        step = np.clip(new - beta, -3, 3)
        beta = beta + step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def fit(X, y, P, rounds=2):
    alpha = 1.0
    for _ in range(rounds):
        beta = irls(X, y, alpha, P)
        alpha = 1.0 / deep.theta_mle(y, np.exp(np.clip(X @ beta, -20, 20)))
    return irls(X, y, alpha, P), alpha


def cluster_cov(X, y, beta, alpha, P, groups):
    mu = np.exp(np.clip(X @ beta, -20, 20))
    bread = np.linalg.inv((X.T * (mu / (1 + alpha * mu))) @ X + P)
    scores = X * ((y - mu) / (1 + alpha * mu))[:, None]
    g = np.bincount(groups).size
    cs = np.zeros((g, X.shape[1]))
    np.add.at(cs, groups, scores)
    return bread @ (cs.T @ cs) @ bread * g / (g - 1), bread


def effective_df(X, beta, alpha, P, cols) -> float:
    mu = np.exp(np.clip(X @ beta, -20, 20))
    A = (X.T * (mu / (1 + alpha * mu))) @ X
    H = np.linalg.solve(A + P, A)
    return float(np.trace(H[np.ix_(cols, cols)]))


# ---- rolling-origin experiment -----------------------------------------------------------------------


def _design(D, s, t, tm, ref=None, sref=None, clim=None, year_fe=False):
    cont, rest = N.features(D, s, t, tm, "none", clim, year_fe)
    X0, r0 = N.design(cont, rest, ref)
    S, sr = smooth_features(D, s, t, clim, sref)
    return np.column_stack([X0, S]), r0, sr


def _log_score(X, y, beta, alpha):
    mu = np.exp(np.clip(X @ beta, -20, 20))
    return compare.log_score(y, mu, mu + alpha * mu**2).mean()


def choose_lambda(D, year: int) -> float:
    """Inner validation: fit on targets before the last 12 months, score the last 12."""
    T = D["Y"].shape[1]
    sp = deep.split(year, T)
    s, t = deep.samples(D["fn"], 0, sp.val_end - 12)
    tm = deep.train_mean(D["Y"], D["fn"], sp.val_end - 12)
    X, ref, sref = _design(D, s, t, tm)
    s2, t2 = deep.samples(D["fn"], sp.val_end - 11, sp.val_end)
    X2, _, _ = _design(D, s2, t2, tm, ref, sref)
    best = None
    for lam in LAMBDAS:
        beta, alpha = fit(
            X, D["Y"][s, t], penalty(X.shape[1] - len(smooth_terms()) * NB_COLS, lam)
        )
        sc = _log_score(X2, D["Y"][s2, t2], beta, alpha)
        best = (sc, lam) if best is None or sc > best[0] else best
    return best[1]


def rolling(D) -> pd.DataFrame:
    T, out = D["Y"].shape[1], []
    for year in range(N.FIRST_TEST, N.LAST_YEAR + 1):
        lam = choose_lambda(D, year)
        sp = deep.split(year, T)
        s, t = deep.samples(D["fn"], 0, sp.val_end)
        tm = deep.train_mean(D["Y"], D["fn"], sp.val_end)
        X, ref, sref = _design(D, s, t, tm)
        n_base = X.shape[1] - len(smooth_terms()) * NB_COLS
        beta, alpha = fit(X, D["Y"][s, t], penalty(n_base, lam))
        s2, t2 = deep.samples(D["fn"], sp.test[0], sp.test[-1])
        X2, _, _ = _design(D, s2, t2, tm, ref, sref)
        mu = np.clip(np.exp(np.clip(X2 @ beta, -20, 20)), 1e-6, 1e6)
        out.append(
            pd.DataFrame(
                {
                    "level": "country",
                    "disease": D["keys"]["disease"].to_numpy()[s2],
                    "area": D["keys"]["area"].to_numpy()[s2],
                    "month": D["idx"][t2],
                    "y": D["Y"][s2, t2],
                    "mu": mu,
                    "var": mu + alpha * mu**2,
                    "status": "tested",
                    "alpha": alpha,
                    "lambda": lam,
                }
            )
        )
        print(f"  nb_gam {year}: lambda={lam:g} alpha={alpha:.2f} n_train={len(s):,}")
    return pd.concat(out, ignore_index=True)


# ---- association (in sample, year effects) -------------------------------------------------------------

_D = None


def _init(D):
    global _D
    _D = D


def _placebo(args):
    seed, alpha, lam = args
    D, T = _D, _D["Y"].shape[1]
    clim = N._permute_years(D, np.random.default_rng(seed))
    s, t = deep.samples(D["fn"], 0, T - 1)
    tm = deep.train_mean(D["Y"], D["fn"], T - 1)
    X, _, _ = _design(D, s, t, tm, clim=clim, year_fe=True)
    y = D["Y"][s, t]
    return N.nb_llf(
        X,
        y,
        irls(X, y, alpha, penalty(X.shape[1] - len(smooth_terms()) * NB_COLS, lam)),
        alpha,
    )


def association(D, lam: float, placebos: int):
    T = D["Y"].shape[1]
    s, t = deep.samples(D["fn"], 0, T - 1)
    tm = deep.train_mean(D["Y"], D["fn"], T - 1)
    y = D["Y"][s, t]
    groups = pd.factorize(D["keys"]["area"].to_numpy()[s])[0]
    X1, _ref, sref = _design(D, s, t, tm, year_fe=True)
    n_base = X1.shape[1] - len(smooth_terms()) * NB_COLS
    P = penalty(n_base, lam)
    beta1, alpha = fit(X1, y, P)
    X0 = X1[:, :n_base]
    beta0 = N.irls_nb(X0, y, alpha)
    lr = 2 * (N.nb_llf(X1, y, beta1, alpha) - N.nb_llf(X0, y, beta0, alpha))
    cov, _ = cluster_cov(X1, y, beta1, alpha, P, groups)
    smooth_cols = np.arange(n_base, X1.shape[1])
    edf = effective_df(X1, beta1, alpha, P, smooth_cols)

    # linear climate model for reference on the same rows (same year effects)
    Xl, _ = N.design(*N.features(D, s, t, tm, "split", year_fe=True))
    bl = N.irls_nb(Xl, y, alpha)
    lr_lin = 2 * (N.nb_llf(Xl, y, bl, alpha) - N.nb_llf(X0, y, beta0, alpha))

    tests = {
        "n_rows": len(y),
        "lambda": lam,
        "alpha": alpha,
        "smooth_edf": edf,
        "smooth_columns": len(smooth_cols),
        "lr_gam": float(lr),
        "lr_linear": float(lr_lin),
    }
    if placebos:
        N.single_thread_workers()
        with ProcessPoolExecutor(initializer=_init, initargs=(D,)) as ex:
            llfs = list(
                ex.map(_placebo, [(N.SEED + b, alpha, lam) for b in range(placebos)])
            )
        null = 2 * (np.array(llfs) - N.nb_llf(X0, y, beta0, alpha))
        tests.update(
            {
                "placebos": placebos,
                "placebo_lr_median": float(np.median(null)),
                "placebo_lr_p95": float(np.percentile(null, 95)),
                "placebo_p": float((1 + (null >= lr).sum()) / (1 + placebos)),
            }
        )

    # partial effect curves (incidence-rate ratio at each anomaly value, vs the sample average)
    grid = np.linspace(-X_RANGE, X_RANGE, 61)
    Bg = basis(grid)
    rows = []
    for j, term in enumerate(smooth_terms()):
        a = n_base + j * NB_COLS
        b, V = beta1[a : a + NB_COLS], cov[a : a + NB_COLS, a : a + NB_COLS]
        f = Bg @ b - (sref[j * NB_COLS : (j + 1) * NB_COLS] @ b)
        se = np.sqrt(np.einsum("ij,jk,ik->i", Bg, V, Bg))
        var, lag, grp = term.rsplit("_", 2)
        rows.append(
            pd.DataFrame(
                {
                    "variable": var,
                    "lag": int(lag[3:]),
                    "group": grp,
                    "anomaly_sd": grid,
                    "irr": np.exp(f),
                    "lo": np.exp(f - 1.96 * se),
                    "hi": np.exp(f + 1.96 * se),
                }
            )
        )
    return pd.Series(tests), pd.concat(rows, ignore_index=True)


def figure(curves: pd.DataFrame, path):
    titles = {
        "tmin_anom": "Minimum temperature",
        "tmax_anom": "Maximum temperature",
        "prec_z": "Precipitation",
    }
    cols = {1: BLUE, 2: ORANGE, 3: GREY}
    fig, axes = plt.subplots(
        2, 3, figsize=(10, 6), sharex=True, sharey=True, facecolor=SURFACE
    )
    for r, grp in enumerate(("vector", "other")):
        for c, var in enumerate(VARS):
            ax = axes[r, c]
            for lag in (1, 2, 3):
                d = curves[
                    (curves["variable"] == var)
                    & (curves["group"] == grp)
                    & (curves["lag"] == lag)
                ]
                ax.fill_between(
                    d["anomaly_sd"], d["lo"], d["hi"], color=cols[lag], alpha=0.12, lw=0
                )
                ax.plot(
                    d["anomaly_sd"],
                    d["irr"],
                    color=cols[lag],
                    lw=1.6,
                    label=f"{lag} month{'s' if lag > 1 else ''} before",
                )
            ax.axhline(1, color=MUTED, lw=0.8)
            ax.set_facecolor(SURFACE)
            ax.set_yscale("log")
            ax.grid(color=GRID, lw=0.6)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
            if r == 0:
                ax.set_title(titles[var], fontsize=10, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(
                    ("Vector-borne diseases" if grp == "vector" else "Other diseases")
                    + "\nrate ratio",
                    fontsize=9,
                    color=INK,
                )
            if r == 1:
                ax.set_xlabel(
                    "anomaly (SD from the 1991-2020 norm)", fontsize=8, color=MUTED
                )
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Outbreak counts versus weather anomalies, by months before (bands: 95% interval)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str]) -> None:
    N.OUT.mkdir(parents=True, exist_ok=True)
    D = N.load()
    if "--association-only" not in argv:
        res = rolling(D)
        res.to_parquet(N.OUT / "tests_nb_gam.parquet")
        cmp_ = N.compare_models()
        cmp_.to_csv(N.TABLES / "nb_experiments.csv", index=False)
        pd.set_option("display.width", 220)
        print(
            cmp_[cmp_["comparison"].str.contains("gam")].round(4).to_string(index=False)
        )
        lam = float(res["lambda"].mode().iloc[0])
    else:
        lam = 10.0
    tests, curves = association(D, lam, 0 if "--skip-placebo" in argv else PLACEBOS)
    tests.to_csv(N.TABLES / "gam_association_tests.csv", header=["value"])
    curves.to_csv(N.TABLES / "gam_association_curves.csv", index=False)
    figure(curves, C.REPORTS / "figures" / "gam_climate_effects.png")
    print(tests.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
