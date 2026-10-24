"""Farrington-style baseline for monthly count series.

For each month we fit an overdispersed (quasi-)Poisson log-linear model on a trailing window that
ends the month before, then predict the month itself. Past outbreak peaks are down-weighted so
earlier waves do not inflate the baseline. Uncertainty combines overdispersion and parameter
error, and is turned into a negative-binomial predictive distribution for p-values and bands.

Only past data is used for each prediction, so the p-values are what a live system would have
produced (no look-ahead).
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats

RIDGE = 1e-6
HUBER_C = 2.58  # standardised Pearson residuals above this are down-weighted
SCREEN_K = 3.0  # a month above 3x its seasonal median counts as a wave, not baseline
SANE_FACTOR = 10.0  # an expected count above 10x the window maximum marks an unstable fit
SANE_SE = 5.0       # so does a standard error of the log-mean above this


@dataclass
class Prediction:
    mu: float      # expected count
    phi: float     # dispersion (>= 1)
    se_eta: float  # std. error of the log-mean prediction
    var: float     # predictive variance of the count


def design(t_years: np.ndarray, cal_month: np.ndarray, n_harm: int, trend: bool,
           lag: np.ndarray | None = None) -> np.ndarray:
    """Columns: intercept, [linear trend], sin/cos pairs of the annual cycle, [log1p lagged count]."""
    cols = [np.ones(len(t_years))]
    if trend:
        cols.append(t_years)
    ang = 2 * np.pi * (cal_month - 1) / 12
    for k in range(1, n_harm + 1):
        cols += [np.sin(k * ang), np.cos(k * ang)]
    if lag is not None:
        cols.append(lag)
    return np.column_stack(cols)


def irls_poisson(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, iters: int = 40):
    """Poisson GLM (log link) by IRLS. Returns beta, mu, (X'WX)^-1."""
    n, p = X.shape
    w = np.ones(n) if w is None else w
    beta = np.zeros(p)
    beta[0] = np.log(np.average(y, weights=w) + 0.5)
    eye = RIDGE * np.eye(p)
    for _ in range(iters):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        XtW = X.T * (w * mu)
        A = XtW @ X + eye
        new = np.linalg.solve(A, XtW @ z)
        done = np.max(np.abs(new - beta)) < 1e-8
        beta = new
        if done:
            break
    eta = np.clip(X @ beta, -20, 20)
    mu = np.exp(eta)
    A = (X.T * (w * mu)) @ X + eye
    return beta, mu, np.linalg.inv(A)


def screen_weights(y: np.ndarray, cal: np.ndarray, k: float = SCREEN_K) -> np.ndarray:
    """Down-weight one-off waves before fitting.

    Each month is compared with the median of the same season (calendar month +/- 1) across the
    window. Recurring seasonal peaks stay (the median is high); a wave seen in one year is far
    above its seasonal median and is excluded. Without this the harmonics fit the wave as a
    seasonal peak and the baseline predicts it again next year with a huge dispersion.
    """
    q = np.empty(len(y))
    for m in range(1, 13):
        d = (cal - m) % 12
        near = (d <= 1) | (d == 11)
        q[cal == m] = np.median(y[near])
    cap = k * np.maximum(q, 1.0)
    return np.where(y > cap, (cap / np.maximum(y, 1e-9)) ** 2, 1.0)


def _dispersion(y, mu, w, p):
    r2 = (y - mu) ** 2 / mu
    return max(1.0, float(np.sum(w * r2) / max(np.sum(w) - p, 1.0)))


def _fit_once(y, t, cal, t_next, cal_next, n_harm, trend, lag, lag_next, screen) -> Prediction:
    if lag is not None:
        centre = lag.mean()  # keeps the intercept interpretable and the fit well-conditioned
        lag, lag_next = lag - centre, np.array([lag_next - centre])
    else:
        lag_next = None
    X = design(t, cal, n_harm, trend, lag)
    x_next = design(np.array([t_next]), np.array([cal_next]), n_harm, trend, lag_next)[0]
    w0 = screen_weights(y, cal) if screen else np.ones(len(y))
    try:
        beta, mu, cov = irls_poisson(X, y, w0)
        phi = _dispersion(y, mu, w0, X.shape[1])
        # Farrington reweighting on top of the season screen: refit once
        h = w0 * mu * np.einsum("ij,jk,ik->i", X, cov, X)
        s = (y - mu) / np.sqrt(phi * mu * np.clip(1 - h, 1e-3, None))
        w = w0 * np.where(s > HUBER_C, (HUBER_C / np.clip(s, 1e-9, None)) ** 2, 1.0)
        beta, mu, cov = irls_poisson(X, y, w)
        phi = _dispersion(y, mu, w, X.shape[1])
    except np.linalg.LinAlgError:
        beta, mu, cov = irls_poisson(X[:, :1], y, w0)
        phi = _dispersion(y, mu, w0, 1)
        x_next = x_next[:1]
    eta = float(np.clip(x_next @ beta, -20, 20))
    se2 = float(phi * x_next @ cov @ x_next)
    m = np.exp(eta)
    return Prediction(mu=m, phi=phi, se_eta=np.sqrt(se2), var=phi * m + m * m * se2)


def _sane(p: Prediction, y: np.ndarray) -> bool:
    """A fit on a sparse window can separate and explode (coefficients in the hundreds)."""
    return bool(np.isfinite(p.mu) and p.mu <= SANE_FACTOR * max(y.max(), 1.0) and p.se_eta <= SANE_SE)


def fit_predict(y, t, cal, t_next, cal_next, n_harm, trend, lag=None, lag_next=0.0,
                screen=True) -> Prediction:
    """Fit on the window (y, t in years, cal = calendar month) and predict the next month.

    With `lag` (log1p of the previous month's count for each window row) the model is a
    log-linear autoregression: it learns persistence, so a month is flagged only if it is a
    surprise given last month. `lag_next` is last month's value for the month to predict.

    If the fit is unstable (expected count far above anything seen in the window, or a huge
    standard error) the model is simplified step by step: drop the lag, then the trend, then
    harmonics, down to a constant.
    """
    use_lag = lag is not None
    while True:
        p = _fit_once(y, t, cal, t_next, cal_next, n_harm, trend, lag if use_lag else None, lag_next, screen)
        if _sane(p, y) or not (use_lag or trend or n_harm):
            return p
        if use_lag:
            use_lag = False
        elif trend:
            trend = False
        else:
            n_harm -= 1


def nb_params(mu, var):
    """Negative-binomial (size, prob) matching mean and variance; Poisson limit when var <= mu."""
    mu, var = np.asarray(mu, float), np.asarray(var, float)
    over = np.maximum(var - mu, 1e-9)
    size = np.where(var > mu * (1 + 1e-6), mu**2 / over, 1e9)
    return size, size / (size + mu)


def upper_p(y, mu, var):
    """P(Y >= y) under the predictive distribution."""
    size, prob = nb_params(mu, var)
    return stats.nbinom.sf(np.asarray(y) - 1, size, prob)


def dunn_smyth(y, mu, var, seed: int = 0):
    """Randomised quantile residuals: ~N(0,1) if the predictive distribution is calibrated.

    Well-behaved for counts near zero, unlike (y - mu) / sd, which explodes when mu is tiny.
    """
    y = np.asarray(y)
    size, prob = nb_params(mu, var)
    lo, hi = stats.nbinom.cdf(y - 1, size, prob), stats.nbinom.cdf(y, size, prob)
    u = lo + np.random.default_rng(seed).uniform(size=len(y)) * (hi - lo)
    return stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))


def upper_quantile(q, mu, var):
    size, prob = nb_params(mu, var)
    return stats.nbinom.ppf(q, size, prob)
