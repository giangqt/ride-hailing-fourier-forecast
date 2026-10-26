#!/usr/bin/env python3
"""
Select the number of weekly Fourier harmonics K using TRAINING/VALIDATION data only (reviewer 1 and 2).

Protocol (everything happens before the first test origin, hour 3048 = 2024-08-06 00:00):
  - zones: top 20 by volume in hours [0, 3048) (training-only selection; ids 264/265 excluded)
  - fit SARIMAX(2,1,2)(1,1,1,24) + Fourier(K, period 168) on hours [0, 2712) for K in {0,1,2,3,4,5}
    (K=0 is the daily-only model); same maxiter ladder (50 -> 200 -> 500, keep best log-likelihood)
  - validation week 1 = hours [2712, 2880): forecast from the fit
  - validation week 2 = hours [2880, 3048): same parameters, Kalman-filtered through week 1, then forecast
  - per K, report mean validation MAE, number of non-converged fits, fit time, mean BIC
Resumable (one CSV row per zone/K task) and parallel (WORKERS processes, 1 BLAS thread each).

Usage: python fourier_k_validation.py            run / resume
       python fourier_k_validation.py --summary  only print the summary table
       python fourier_k_validation.py --synthetic   tiny sanity run, no database
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX

SYNTH = "--synthetic" in sys.argv
T_HOURS, FIRST_ORIGIN, VAL, TOP_N = 4392, 3048, 168, 20
KS = [0, 3, 1, 2, 4, 5]
LADDER = (50, 200, 500)
EXCLUDE = {264, 265}
WORKERS = int(os.environ.get("WORKERS", "3"))
OUT = os.path.expanduser("~/ride-hailing/experiments/weather_sarimax/fourier_k_validation.csv")
if SYNTH:
    T_HOURS, FIRST_ORIGIN, TOP_N, KS, LADDER, OUT = 900, 720, 3, [0, 1, 2], (20, 50), "/tmp/fourier_k_validation_synth.csv"
N_FIT = FIRST_ORIGIN - 2 * VAL

Y = None
ZONES = None


def load():
    global Y, ZONES
    if SYNTH:
        rng = np.random.default_rng(0)
        t = np.arange(T_HOURS)
        base = (1 + .5 * np.sin(2 * np.pi * t / 24)) * (1 + .4 * ((t // 24) % 7 >= 5))
        ZONES = [11, 22, 33][:TOP_N]
        Y = np.column_stack([rng.poisson(base * s) for s in (200, 120, 60)]).astype(float)
        return
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("PGHOST", "localhost"), port=os.environ.get("PGPORT", "5432"),
                            dbname=os.environ.get("PGDATABASE", "rides"), user=os.environ.get("PGUSER", "rides"),
                            password=os.environ.get("PGPASSWORD", "rides"))
    df = pd.read_sql("SELECT time_bucket, zone_id, pickup_count FROM hourly_demand_6mo_all", conn)
    conn.close()
    P = df.pivot(index="time_bucket", columns="zone_id", values="pickup_count").sort_index()
    assert len(P) == T_HOURS
    vol = P.iloc[:FIRST_ORIGIN].sum().drop(labels=list(EXCLUDE), errors="ignore").sort_values(ascending=False, kind="stable")
    ZONES = vol.index[:TOP_N].tolist()
    Y = P[ZONES].values.astype(float)


def fourier(n, k, period=168):
    t = np.arange(n)
    return np.column_stack([f(2 * np.pi * i * t / period) for i in range(1, k + 1) for f in (np.sin, np.cos)])


def fit_ladder(y, exog):
    best, conv, used = None, False, None
    for mi in LADDER:
        m = SARIMAX(y, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24), exog=exog,
                    enforce_stationarity=False, enforce_invertibility=False)
        r = m.fit(disp=False, maxiter=mi)
        c = bool(r.mle_retvals.get("converged", True))
        if best is None or r.llf > best.llf:
            best, conv, used = r, c, mi
        if c:
            break
    return best, conv, used


def task(arg):
    zi, K = arg
    y = Y[:, zi]
    ex = fourier(T_HOURS, K) if K else None
    t0 = time.perf_counter()
    res, conv, used = fit_ladder(y[:N_FIT], ex[:N_FIT] if K else None)
    fc1 = res.get_forecast(VAL, exog=ex[N_FIT:N_FIT + VAL] if K else None).predicted_mean
    res2 = res.apply(y[:N_FIT + VAL], exog=ex[:N_FIT + VAL] if K else None, refit=False)
    fc2 = res2.get_forecast(VAL, exog=ex[N_FIT + VAL:N_FIT + 2 * VAL] if K else None).predicted_mean
    out = {"zone_id": ZONES[zi], "K": K, "seconds": time.perf_counter() - t0, "converged": conv, "maxiter_used": used,
           "llf": float(res.llf), "aic": float(res.aic), "bic": float(res.bic)}
    for w, (fc, lo) in enumerate([(fc1, N_FIT), (fc2, N_FIT + VAL)], start=1):
        p = np.clip(np.asarray(fc), 0, None)
        a = y[lo:lo + VAL]
        out[f"mae_w{w}"] = float(np.abs(a - p).mean())
        out[f"rmse_w{w}"] = float(np.sqrt(((a - p) ** 2).mean()))
    return out


def summary():
    d = pd.read_csv(OUT).drop_duplicates(["zone_id", "K"], keep="first")
    d["val_mae"] = (d["mae_w1"] + d["mae_w2"]) / 2
    d["val_rmse"] = (d["rmse_w1"] + d["rmse_w2"]) / 2
    g = d.groupby("K").agg(zones=("zone_id", "nunique"), val_MAE=("val_mae", "mean"), val_RMSE=("val_rmse", "mean"),
                           MAE_week1=("mae_w1", "mean"), MAE_week2=("mae_w2", "mean"),
                           nonconverged=("converged", lambda s: int((~s.astype(bool)).sum())),
                           mean_fit_s=("seconds", "mean"), mean_BIC=("bic", "mean"))
    pd.set_option("display.width", 200)
    print(g.round(2).to_string())
    full = g[g.zones == g.zones.max()]
    if len(full):
        best = full["val_MAE"].idxmin()
        ok = full[(full.val_MAE <= 1.01 * full.val_MAE.min()) & (full.nonconverged == 0)]
        print(f"\nK with lowest mean validation MAE: {best}")
        print("Selection rule (fixed in advance): smallest K whose validation MAE is within 1% of the best and "
              "whose fits all converged ->", int(ok.index.min()) if len(ok) else "no K satisfies the rule")
    print("\nNon-converged fits by K and zone:")
    print(d[~d.converged.astype(bool)][["K", "zone_id", "maxiter_used"]].sort_values(["K", "zone_id"]).to_string(index=False))


def main():
    if "--summary" in sys.argv:
        return summary()
    load()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.zone_id, prev.K))
    tasks = [(zi, K) for K in KS for zi in range(len(ZONES)) if (ZONES[zi], K) not in done]
    print(f"zones={ZONES}\nN_FIT={N_FIT} val weeks=[{N_FIT},{N_FIT + VAL}) and [{N_FIT + VAL},{N_FIT + 2 * VAL}) "
          f"workers={WORKERS} tasks to run={len(tasks)} (already done {len(done)})", flush=True)
    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(task, t): t for t in tasks}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:
                print(f"[FAILED] zone idx {futs[f][0]} K={futs[f][1]}: {e!r}", flush=True)
                continue
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            n += 1
            el = time.time() - t0
            print(f"[{n}/{len(tasks)}] zone {r['zone_id']} K={r['K']} conv={r['converged']} "
                  f"MAE w1/w2={r['mae_w1']:.1f}/{r['mae_w2']:.1f} fit={r['seconds']:.0f}s "
                  f"elapsed={el / 60:.0f}min eta={el / n * (len(tasks) - n) / 60:.0f}min", flush=True)
    print()
    summary()


if __name__ == "__main__":
    main()