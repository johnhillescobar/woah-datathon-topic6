"""Pooled one-step-ahead models for monthly outbreak counts: LSTM and LightGBM.

Benchmark against the per-series baseline under a fixed rolling-origin protocol. For each test year Y
one model is fitted on targets up to December Y-1 (the last 12 months are a validation block for early
stopping / dispersion), then every month of Y is predicted one step ahead from data up to the month
before. All series are pooled in one model; all series-level statistics use training months only.

Run with `uv run python -m t6.deep [lstm|gbm] [--force]` (default: both). Writes
data/processed/tests_outbreaks_{lstm,gbm}.parquet; each fit is cached in data/processed/deep_cache/.
"""
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

from t6 import series
from t6 import config as C

TABLES = C.REPORTS / "tables"
CACHE = C.PROCESSED / "deep_cache"
LEVELS = ("global", "region", "country")
START_YEAR = int(C.START[:4])
TEST_YEARS = range(2018, 2027)
MIN_AGE = 12    # months since the first nonzero count before a series contributes samples
VAL_MONTHS = 12
SEQ = 24
LSTM = dict(hidden=32, dropout=0.1, lr=1e-3, batch=512, epochs=40, patience=5, seeds=(0, 1, 2))
GBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50)
LOG_MU, LOG_THETA = (-12.0, 12.0), (-4.0, 9.0)
MU_FLOOR = 1e-6


# ---- data and split ---------------------------------------------------------------------------

def load_wide():
    """Series keys, count matrix (series x month) and the month index."""
    ob, ev = series.load_tables()
    wide = series.monthly_counts(ob, ev, "outbreaks")
    return wide.index.to_frame(index=False), wide.to_numpy(np.float64), series.months()


def first_nonzero(Y: np.ndarray) -> np.ndarray:
    nz = Y > 0
    return np.where(nz.any(1), nz.argmax(1), Y.shape[1])


@dataclass
class Split:
    year: int
    train_end: int  # last training target (December Y-2)
    val_end: int    # last validation target (December Y-1)
    test: np.ndarray


def split(year: int, n_months: int) -> Split:
    val_end = (year - 1 - START_YEAR) * 12 + 11
    test = np.arange(val_end + 1, min(val_end + 13, n_months))
    return Split(year, val_end - VAL_MONTHS, val_end, test)


def samples(fn: np.ndarray, t_lo: int, t_hi: int):
    """(series, month) pairs in [t_lo, t_hi] with at least MIN_AGE months since the first count."""
    s, t = np.meshgrid(np.arange(len(fn)), np.arange(t_lo, t_hi + 1), indexing="ij")
    keep = t >= fn[:, None] + MIN_AGE
    return s[keep], t[keep]


def train_mean(Y: np.ndarray, fn: np.ndarray, end: int) -> np.ndarray:
    """Mean count over each series' active months up to `end` (0 if it has not started yet)."""
    return np.array([Y[i, f:end + 1].mean() if f <= end else 0.0 for i, f in enumerate(fn)])


def padded_log(Y: np.ndarray) -> np.ndarray:
    """log1p counts with SEQ zero months in front, so month t-k sits at column t+SEQ-k."""
    return np.pad(np.log1p(Y), ((0, 0), (SEQ, 0)))


def lstm_inputs(L, s, t, fn, lvl, tm, norm):
    """Sequence (N, SEQ, 3) of [scaled log1p count, sin, cos of calendar month] for months t-SEQ..t-1,
    and static features (N, 5): level one-hot, log1p training mean, log1p months since first count."""
    x = L[s[:, None], t[:, None] + np.arange(SEQ)]
    ang = 2 * np.pi * ((t[:, None] + np.arange(SEQ)) % 12) / 12  # SEQ is a multiple of 12
    seq = np.stack([(x - norm["m"]) / norm["sd"], np.sin(ang), np.cos(ang)], -1)
    stat = np.column_stack([np.eye(3)[lvl[s]], (np.log1p(tm[s]) - norm["tm"][0]) / norm["tm"][1],
                            (np.log1p(t - fn[s]) - norm["age"][0]) / norm["age"][1]])
    return seq.astype(np.float32), stat.astype(np.float32)


def lstm_norm(L, Y, fn, tm, train_end, s, t):
    """Training-only scalers: log1p counts over active training months, static features over samples."""
    T = Y.shape[1]
    m = np.arange(T)[None, :]
    vals = L[:, SEQ:][(m >= fn[:, None]) & (m <= train_end)]
    ltm, lage = np.log1p(tm[s]), np.log1p(t - fn[s])
    return {"m": vals.mean(), "sd": vals.std() + 1e-6, "tm": (ltm.mean(), ltm.std() + 1e-6),
            "age": (lage.mean(), lage.std() + 1e-6)}


def gbm_inputs(L, s, t, fn, lvl, tm):
    """Lags 1,2,3,6,12, rolling means over 3/6/12 months, smoothed same-month-last-year, calendar month,
    level, log1p training mean, months since first count. Names in GBM_COLS."""
    lag = lambda k: L[s, t + SEQ - k]
    cols = [lag(k) for k in (1, 2, 3, 6, 12)]
    cols += [np.mean([lag(k) for k in range(1, w + 1)], 0) for w in (3, 6, 12)]
    cols += [np.mean([lag(11), lag(12), lag(13)], 0), t % 12 + 1, lvl[s], np.log1p(tm[s]), t - fn[s]]
    return np.column_stack(cols)


GBM_COLS = ["lag1", "lag2", "lag3", "lag6", "lag12", "roll3", "roll6", "roll12", "seas12", "cal_month",
            "level", "log_train_mean", "age"]


def theta_mle(y, mu) -> float:
    """Maximum-likelihood NB size for given means."""
    mu = np.clip(mu, MU_FLOOR, None)

    def nll(lt):
        th = np.exp(lt)
        return -stats.nbinom.logpmf(y, th, th / (th + mu)).sum()

    return float(np.exp(optimize.minimize_scalar(nll, bounds=LOG_THETA, method="bounded").x))


def _index_sets(fn, sp):
    return {"train": samples(fn, 0, sp.train_end), "val": samples(fn, sp.train_end + 1, sp.val_end),
            "test": samples(fn, sp.test[0], sp.test[-1])}


def lstm_sets(Y, fn, lvl, sp):
    """{train, val, test: (series, month, sequence, static)} with training-only scaling."""
    L, tm, idx = padded_log(Y), train_mean(Y, fn, sp.train_end), _index_sets(fn, sp)
    norm = lstm_norm(L, Y, fn, tm, sp.train_end, *idx["train"])
    return {k: (s, t, *lstm_inputs(L, s, t, fn, lvl, tm, norm)) for k, (s, t) in idx.items()}


def gbm_sets(Y, fn, lvl, sp):
    """{train, val, test: (series, month, features)}."""
    L, tm, idx = padded_log(Y), train_mean(Y, fn, sp.train_end), _index_sets(fn, sp)
    return {k: (s, t, gbm_inputs(L, s, t, fn, lvl, tm)) for k, (s, t) in idx.items()}


# ---- LightGBM ---------------------------------------------------------------------------------

def gbm_fit_predict(Y, fn, lvl, sp, refit=False):
    """Poisson GBM on targets <= sp.train_end with one NB theta per level from the validation block.

    refit=True then refits the same model (same trees) on every target <= sp.val_end and keeps those
    thetas; it is used only for the forward forecast in companion.py, not in the rolling benchmark.
    Returns (series, month, mu, theta) for the test months and the (train, validation) sample counts.
    """
    import lightgbm as lgb

    params = {"objective": "poisson", "learning_rate": GBM["learning_rate"], "num_leaves": GBM["num_leaves"],
              "min_data_in_leaf": GBM["min_child_samples"], "num_threads": 1, "seed": 0, "deterministic": True,
              "force_row_wise": True, "verbose": -1}

    def train(sets):
        s, t, x = sets["train"]
        return lgb.train(params, lgb.Dataset(x, Y[s, t], feature_name=GBM_COLS, categorical_feature=["level"]),
                         num_boost_round=GBM["n_estimators"])

    sets = gbm_sets(Y, fn, lvl, sp)
    model = train(sets)
    (s_va, t_va, x_va), n_train = sets["val"], len(sets["train"][0])
    mu_va = np.clip(model.predict(x_va), MU_FLOOR, None)
    theta = {l: theta_mle(Y[s_va, t_va][lvl[s_va] == l], mu_va[lvl[s_va] == l]) for l in range(3)}
    if refit:
        sets = gbm_sets(Y, fn, lvl, Split(sp.year, sp.val_end, sp.val_end, sp.test))
        model, n_train = train(sets), len(sets["train"][0])
    s_te, t_te, x_te = sets["test"]
    mu = np.clip(model.predict(x_te), MU_FLOOR, None)
    return s_te, t_te, mu, np.array([theta[l] for l in lvl[s_te]]), (n_train, len(s_va))


def fit_gbm(task):
    """One refit of the rolling benchmark: targets <= train_end, a theta per level from the validation block."""
    Y, fn, lvl, year = task
    t0 = time.time()
    s_te, t_te, mu, th, (n_train, n_val) = gbm_fit_predict(Y, fn, lvl, split(year, Y.shape[1]))
    info = {"year": year, "seed": -1, "n_train": n_train, "n_val": n_val, "epochs": GBM["n_estimators"],
            "val_nll": np.nan, "seconds": time.time() - t0}
    return s_te, t_te, mu, th, info


# ---- LSTM -------------------------------------------------------------------------------------

def nb_nll(y, log_mu, log_theta):
    import torch

    th, lden = log_theta.exp(), torch.logaddexp(log_theta, log_mu)
    ll = (torch.lgamma(y + th) - torch.lgamma(th) - torch.lgamma(y + 1)
          + th * (log_theta - lden) + y * (log_mu - lden))
    return -ll


def lstm_fit_predict(Y, fn, lvl, sp, seed, fixed_epochs=None):
    """Train on targets <= sp.train_end and predict the test months (mu, theta).

    Default: early stopping on the validation block (patience 5, best weights kept). With fixed_epochs
    the net is trained for exactly that many epochs and the validation block is ignored; companion.py
    uses this to refit on all data with the epoch count found by early stopping.
    """
    import copy

    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    sets = lstm_sets(Y, fn, lvl, sp)
    (s_tr, t_tr, x_tr, z_tr), (s_va, t_va, x_va, z_va), (s_te, t_te, x_te, z_te) = (
        sets["train"], sets["val"], sets["test"])
    x_tr, z_tr, x_va, z_va, x_te, z_te = map(torch.from_numpy, (x_tr, z_tr, x_va, z_va, x_te, z_te))
    y_tr, y_va = (torch.from_numpy(Y[s, t].astype(np.float32)) for s, t in ((s_tr, t_tr), (s_va, t_va)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(3, LSTM["hidden"], batch_first=True)
            self.drop = nn.Dropout(LSTM["dropout"])
            self.head = nn.Linear(LSTM["hidden"] + 5, 2)
            with torch.no_grad():
                self.head.bias[:] = torch.tensor([float(np.log(Y[s_tr, t_tr].mean() + 0.1)), 0.0])

        def forward(self, x, z):
            h = self.lstm(x)[1][0][-1]
            o = self.head(torch.cat([self.drop(h), z], 1))
            return o[:, 0].clamp(*LOG_MU), o[:, 1].clamp(*LOG_THETA)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=LSTM["lr"])
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, bad, ep = np.inf, None, 0, 0, 0
    for ep in range(1, (fixed_epochs or LSTM["epochs"]) + 1):
        net.train()
        perm = torch.randperm(len(y_tr), generator=gen)
        for i in range(0, len(perm), LSTM["batch"]):
            b = perm[i:i + LSTM["batch"]]
            loss = nb_nll(y_tr[b], *net(x_tr[b], z_tr[b])).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if fixed_epochs:
            continue
        net.eval()
        with torch.no_grad():
            val = float(nb_nll(y_va, *net(x_va, z_va)).mean())
        if val < best - 1e-6:
            best, best_state, best_ep, bad = val, copy.deepcopy(net.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= LSTM["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        lm, lt = net(x_te, z_te)
    mu, th = np.clip(lm.exp().numpy().astype(float), MU_FLOOR, None), lt.exp().numpy().astype(float)
    return s_te, t_te, mu, th, {"n_train": len(y_tr), "n_val": len(y_va), "epochs": ep, "best_epoch": best_ep,
                                "val_nll": best}


def fit_lstm(task):
    """One refit (year, seed) of the rolling benchmark: early stopping on the validation block."""
    Y, fn, lvl, year, seed = task
    t0 = time.time()
    s_te, t_te, mu, th, info = lstm_fit_predict(Y, fn, lvl, split(year, Y.shape[1]), seed)
    return s_te, t_te, mu, th, {"year": year, "seed": seed, **info, "seconds": time.time() - t0}


# ---- driver -----------------------------------------------------------------------------------

def _key(model: str) -> str:
    cfg = repr((LSTM if model == "lstm" else GBM, MIN_AGE, VAL_MONTHS, SEQ, LOG_MU, LOG_THETA, MU_FLOOR))
    return hashlib.md5(cfg.encode()).hexdigest()[:8]


def _cache_file(model: str, year: int, seed: int):
    return CACHE / f"{model}_{year}_s{seed}_{_key(model)}.npz"


def _run(task):
    model, job, path = task
    s, t, mu, th, info = (fit_lstm if model == "lstm" else fit_gbm)(job)
    np.savez(path, s=s, t=t, mu=mu, theta=th, **{k: np.array(v) for k, v in info.items()})
    return path


def run_model(model: str, force: bool = False) -> pd.DataFrame:
    keys, Y, idx = load_wide()
    fn = first_nonzero(Y)
    lvl = keys["level"].map({l: i for i, l in enumerate(LEVELS)}).to_numpy()
    CACHE.mkdir(parents=True, exist_ok=True)
    seeds = LSTM["seeds"] if model == "lstm" else (-1,)
    todo = []
    for year in TEST_YEARS:
        for seed in seeds:
            path = _cache_file(model, year, seed)
            if force or not path.exists():
                job = (Y, fn, lvl, year, seed) if model == "lstm" else (Y, fn, lvl, year)
                todo.append((model, job, path))
    t0 = time.time()
    if todo:
        workers = min(len(todo), max(1, (os.cpu_count() or 2) - 1))
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_run, todo))
        print(f"[{model}] fitted {len(todo)} models in {time.time() - t0:.0f}s")
    else:
        print(f"[{model}] all fits cached")
    return assemble(model, keys, Y, idx, seeds)


def assemble(model: str, keys, Y, idx, seeds) -> pd.DataFrame:
    """Combine cached fits (mean over seeds of mu and theta) into the contract table."""
    parts, runs = [], []
    for year in TEST_YEARS:
        fits = [np.load(_cache_file(model, year, sd)) for sd in seeds]
        s, t = fits[0]["s"], fits[0]["t"]
        mu = np.mean([f["mu"] for f in fits], 0)
        th = np.mean([f["theta"] for f in fits], 0)
        d = pd.DataFrame({"level": keys["level"].to_numpy()[s], "disease": keys["disease"].to_numpy()[s],
                          "area": keys["area"].to_numpy()[s], "metric": "outbreaks", "month": idx[t],
                          "y": Y[s, t], "mu": mu, "var": mu + mu**2 / th, "theta": th, "status": "tested"})
        if model == "lstm":
            for k, f in enumerate(fits):
                d[f"mu_seed{k}"], d[f"theta_seed{k}"] = f["mu"], f["theta"]
        parts.append(d)
        runs += [{"model": model, **{k: float(f[k]) for k in ("year", "seed", "n_train", "n_val", "epochs",
                                                               "val_nll", "seconds")}} for f in fits]
    out = pd.concat(parts, ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(TABLES / f"deep_runs_{model}.csv", index=False)
    return out


def main(args: list[str]) -> None:
    force = "--force" in args
    models = [a for a in args if a in ("lstm", "gbm")] or ["gbm", "lstm"]
    t0 = time.time()
    for model in models:
        df = run_model(model, force)
        df.to_parquet(C.PROCESSED / f"tests_outbreaks_{model}.parquet")
        runs = pd.read_csv(TABLES / f"deep_runs_{model}.csv")
        print(f"[{model}] {len(df):,} predictions, {df['month'].min():%Y-%m}..{df['month'].max():%Y-%m}, "
              f"fit seconds (summed over fits) {runs['seconds'].sum():.0f}, "
              f"epochs mean {runs['epochs'].mean():.1f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
