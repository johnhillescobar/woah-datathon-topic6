"""Simulation benchmark for the detectors (in the spirit of Noufaily et al. 2013).

Seasonal negative-binomial series with known truth: historical waves contaminate the training
history, and one outbreak of known size is injected into the test period. We measure, per detector,
the false-alarm rate on outbreak-free series and the probability of detecting the injected outbreak.

Run with `uv run python -m t6.simulate`. Writes reports/tables/simulation_detection.csv.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from t6 import config as C
from t6 import detect

N_MONTHS = 240
TEST_FROM = 120
LEVELS = (3, 10, 40)        # baseline mean monthly count
MULTIPLIERS = (2.5, 4, 8)   # outbreak size relative to the expected count
OUTBREAK_LEN = 3
N_PER_CELL = 60
THETA = 5.0                 # overdispersion: var = mu + mu^2 / theta
DETECTORS = ("legacy", "baseline", "ar1")


def make_series(rng, level, multiplier=None, amp=0.6):
    cal = np.arange(N_MONTHS) % 12 + 1
    mu = level * np.exp(amp * np.sin(2 * np.pi * (cal - 1) / 12))
    for _ in range(2):  # past waves that contaminate the training history
        s, d = rng.integers(12, 110), rng.integers(3, 6)
        mu[s:s + d] *= rng.uniform(15, 40)
    start = None
    if multiplier is not None:
        start = int(rng.integers(TEST_FROM + 12, N_MONTHS - OUTBREAK_LEN))
        mu[start:start + OUTBREAK_LEN] *= multiplier
    return rng.negative_binomial(THETA, THETA / (THETA + mu)).astype(float), start


def build(seed=0):
    rng = np.random.default_rng(seed)
    meta = {}
    for level in LEVELS:
        for mult in (None, *MULTIPLIERS):
            for i in range(N_PER_CELL):
                y, start = make_series(rng, level, mult)
                meta[f"L{level}_m{mult}_{i}"] = (y, level, mult, start)
    return meta


def run_detector(meta, model):
    tasks = [(("sim", sid, "-", "outbreaks"), v[0], "outbreaks", model) for sid, v in meta.items()]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        res = list(ex.map(detect.run_one, tasks, chunksize=16))
    cols = ["level", "disease", "area", "metric", "t", "y", "mu", "phi", "se_eta", "var", "status"]
    df = pd.DataFrame([r for chunk in res for r in chunk], columns=cols)
    df["month"] = pd.date_range(C.START, periods=N_MONTHS, freq="MS")[df["t"].to_numpy()]
    df = detect.score(df.drop(columns="t"), "outbreaks")
    df["t"] = df["month"].dt.year * 12 + df["month"].dt.month - (2005 * 12 + 1)
    return df[(df["status"] == "tested") & (df["t"] >= TEST_FROM)]


def evaluate(meta, model):
    df = run_detector(meta, model)
    alerts = df[df["alert"]].groupby("disease")["t"].apply(list).to_dict()
    tested = df.groupby("disease").size().to_dict()
    rows = []
    for sid, (_, level, mult, start) in meta.items():
        a = alerts.get(sid, [])
        if mult is None:
            rows.append((model, level, "none", 0, len(a), tested.get(sid, 0), np.nan))
        else:
            hit = [t for t in a if start <= t < start + OUTBREAK_LEN]
            outside = [t for t in a if not (start <= t < start + OUTBREAK_LEN)]
            rows.append((model, level, mult, int(bool(hit)), len(outside), tested.get(sid, 0),
                         (min(hit) - start) if hit else np.nan))
    return pd.DataFrame(rows, columns=["model", "level", "multiplier", "detected", "false_alerts",
                                       "tested_months", "delay"])


def main() -> None:
    meta = build()
    res = pd.concat([evaluate(meta, m) for m in DETECTORS])
    res["multiplier"] = res["multiplier"].astype(str)
    out = res.groupby(["model", "level", "multiplier"]).agg(
        n=("detected", "size"),
        detection_prob=("detected", "mean"),
        median_delay_months=("delay", "median"),
    ).reset_index()
    fa = res.groupby(["model", "level", "multiplier"]).apply(
        lambda d: 100 * d["false_alerts"].sum() / max(d["tested_months"].sum(), 1), include_groups=False)
    out["false_alerts_per_100_months"] = fa.to_numpy()
    out.loc[out["multiplier"] == "none", ["detection_prob", "median_delay_months"]] = np.nan
    C.REPORTS.joinpath("tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(C.REPORTS / "tables" / "simulation_detection.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    print("\nfalse alerts per 100 outbreak-free months, by model:")
    print(out[out.multiplier == "none"].groupby("model")["false_alerts_per_100_months"].mean().round(3))
    print("detection probability, by model x multiplier (mean over levels):")
    print(out[out.multiplier != "none"].pivot_table(index="model", columns="multiplier",
                                                    values="detection_prob").round(2))


if __name__ == "__main__":
    main()
