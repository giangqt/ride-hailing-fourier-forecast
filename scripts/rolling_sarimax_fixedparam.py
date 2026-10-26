#!/usr/bin/env python3
"""
Rolling-origin evaluation of the SARIMAX family on the SAME 8 origins / zones / 168-h horizon as rolling_origin_cheap.py.

Models (all SARIMAX(2,1,2)(1,1,1,24) core, same maxiter ladder 50 -> 200 -> 500, keep best log-likelihood):
  sarimax_daily            : no exogenous terms
  sarimax_fourier3         : + weekly Fourier terms, K=3 (K chosen on validation data, see fourier_k_validation.py)
  sarimax_fourier3_weather : + standardized temperature and precipitation (observed, retrospective), on the
                             Fourier model - weather tested on the corrected seasonal specification

Fixed-parameter rolling origin: each model is FITTED ONCE on hours [0, 3048) (everything before the first origin).
For origin o > 3048 the same parameters are re-applied to hours [0, o) by Kalman filtering (no re-estimation), then a
168-h forecast is made. Re-fitting 20 zones x 3 models at 8 origins is not affordable (see profile_runtime_memory).
The paper's own single-origin refit at hour 4224 is the reference for how much stale parameters cost.

Tasks are (model, zone) pairs, run in parallel and cached one file each (resumable).
Usage: python rolling_sarimax_fixedparam.py             run / resume, then summary
       python rolling_sarimax_fixedparam.py --summary   only aggregate what exists
       python rolling_sarimax_fixedparam.py --synthetic tiny sanity run
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
T_HOURS, HORIZON, N_ORIGINS, FIRST, TOP_N = 4392, 168, 8, 3048, 20
LADDER = (50, 200, 500)
EXCLUDE = {264, 265}
WORKERS = int(os.environ.get("WORKERS", "3"))
BASE = os.path.expanduser("~/ride-hailing/experiments/weather_sarimax")
CACHE = os.path.join(BASE, "rolling_sarimax_cache")
MODELS = {"sarimax_fourier3": dict(K=3, weather=False),
          "sarimax_daily": dict(K=0, weather=False),
          "sarimax_fourier3_weather": dict(K=3, weather=True)}
if SYNTH:
    T_HOURS, FIRST, TOP_N, LADDER = 720 + 8 * 168, 720, 3, (20, 50)
    BASE = "/tmp/rolling_sarimax_synth"
    CACHE = os.path.join(BASE, "cache")
ORIGINS = [FIRST + HORIZON * j for j in range(N_ORIGINS)]
Y = ZONES = W = None


def load():
    global Y, ZONES, W
    if SYNTH:
        rng = np.random.default_rng(0)
        t = np.arange(T_HOURS)
        base = (1 + .5 * np.sin(2 * np.pi * t / 24)) * (1 + .4 * ((t // 24) % 7 >= 5))
        ZONES = [11, 22, 33][:TOP_N]
        Y = np.column_stack([rng.poisson(base * s) for s in (200, 120, 60)]).astype(float)
        W = np.column_stack([rng.normal(size=T_HOURS), (rng.random(T_HOURS) < .1) * rng.exponential(2, T_HOURS)])
        return
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("PGHOST", "localhost"), port=os.environ.get("PGPORT", "5432"),
                            dbname=os.environ.get("PGDATABASE", "rides"), user=os.environ.get("PGUSER", "rides"),
                            password=os.environ.get("PGPASSWORD", "rides"))
    df = pd.read_sql("SELECT time_bucket, zone_id, pickup_count, avg_temperature, precipitation_mm "
                     "FROM hourly_demand_6mo_all", conn)
    conn.close()
    P = df.pivot(index="time_bucket", columns="zone_id", values="pickup_count").sort_index()
    assert len(P) == T_HOURS
    vol = P.iloc[:FIRST].sum().drop(labels=list(EXCLUDE), errors="ignore").sort_values(ascending=False, kind="stable")
    ZONES = vol.index[:TOP_N].tolist()
    Y = P[ZONES].values.astype(float)
    W = df.groupby("time_bucket")[["avg_temperature", "precipitation_mm"]].mean().sort_index().values.astype(float)


def fourier(n, k, period=168):
    t = np.arange(n)
    return np.column_stack([f(2 * np.pi * i * t / period) for i in range(1, k + 1) for f in (np.sin, np.cos)])


def build_exog(spec):
    cols = []
    if spec["K"]:
        cols.append(fourier(T_HOURS, spec["K"]))
    if spec["weather"]:
        mu, sd = W[:FIRST].mean(0), W[:FIRST].std(0)
        cols.append((W - mu) / np.where(sd == 0, 1.0, sd))
    return np.column_stack(cols) if cols else None


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


def path_for(model, zone):
    return os.path.join(CACHE, f"{model}_{zone}.npz")


def task(arg):
    model, zi = arg
    spec, y, ex = MODELS[model], Y[:, arg[1]], build_exog(MODELS[model])
    t0 = time.perf_counter()
    res, conv, used = fit_ladder(y[:FIRST], ex[:FIRST] if ex is not None else None)
    fit_s = time.perf_counter() - t0
    preds = np.zeros((N_ORIGINS, HORIZON))
    for oi, o in enumerate(ORIGINS):
        r = res if oi == 0 else res.apply(y[:o], exog=ex[:o] if ex is not None else None, refit=False)
        fc = r.get_forecast(HORIZON, exog=ex[o:o + HORIZON] if ex is not None else None).predicted_mean
        preds[oi] = np.clip(np.asarray(fc), 0, None)
    meta = dict(fit_seconds=fit_s, converged=conv, maxiter_used=used, llf=float(res.llf),
                temp_coef=np.nan, temp_p=np.nan, precip_coef=np.nan, precip_p=np.nan)
    if spec["weather"]:
        names = list(res.model.param_names)
        ti, pi = names.index(f"x{ex.shape[1] - 1}"), names.index(f"x{ex.shape[1]}")
        meta.update(temp_coef=float(res.params[ti]), temp_p=float(res.pvalues[ti]),
                    precip_coef=float(res.params[pi]), precip_p=float(res.pvalues[pi]))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path_for(model, ZONES[zi]), preds=preds, **meta)
    return model, ZONES[zi], meta


def metrics(a, p):
    mae = np.abs(a - p).mean()
    rmse = np.sqrt(((a - p) ** 2).mean())
    nz = a != 0
    mape = float((np.abs(a[nz] - p[nz]) / a[nz]).mean() * 100) if nz.any() else np.nan
    return float(mae), float(rmse), mape, int((~nz).sum())


def summary():
    if ZONES is None:
        load()
    rows, errs, meta_rows, missing = [], {m: np.full((N_ORIGINS, HORIZON, len(ZONES)), np.nan, np.float32) for m in MODELS}, [], []
    for model in MODELS:
        for zi, z in enumerate(ZONES):
            p = path_for(model, z)
            if not os.path.exists(p):
                missing.append((model, z))
                continue
            d = np.load(p)
            meta_rows.append(dict(model=model, zone_id=z, **{k: float(d[k]) for k in
                                  ["fit_seconds", "converged", "maxiter_used", "llf", "temp_coef", "temp_p", "precip_coef", "precip_p"]}))
            for oi, o in enumerate(ORIGINS):
                a, pr = Y[o:o + HORIZON, zi], d["preds"][oi]
                errs[model][oi, :, zi] = np.abs(a - pr)
                mae, rmse, mape, nzero = metrics(a, pr)
                rows.append(dict(origin_idx=oi, origin_start=str(pd.Timestamp("2024-04-01") + pd.Timedelta(hours=o)),
                                 stratum="top20", zone_id=z, model=model, mae=mae, rmse=rmse, mape=mape, n_zero=nzero))
    if missing:
        print(f"[note] {len(missing)} (model, zone) results missing, summary is partial: {missing[:6]}...")
    cases = pd.DataFrame(rows)
    meta = pd.DataFrame(meta_rows)
    os.makedirs(BASE, exist_ok=True)
    cases.to_csv(os.path.join(BASE, "rolling_sarimax_cases.csv"), index=False)
    meta.to_csv(os.path.join(BASE, "rolling_sarimax_fit_meta.csv"), index=False)
    np.savez_compressed(os.path.join(BASE, "rolling_sarimax_abs_err.npz"), zones=np.array(ZONES),
                        origins=np.array(ORIGINS), **errs)
    allc = cases
    cheap = os.path.join(BASE, "rolling_cheap_top20.csv")
    if os.path.exists(cheap):
        c = pd.read_csv(cheap)
        allc = pd.concat([c[cases.columns.intersection(c.columns)], cases], ignore_index=True)
    pd.set_option("display.width", 220)
    base = allc[allc.model == "naive168"].set_index(["origin_idx", "zone_id"])["mae"]
    out = []
    for m, g in allc.groupby("model"):
        gi = g.set_index(["origin_idx", "zone_id"])
        ratio = (gi["mae"] / base).replace([np.inf, -np.inf], np.nan)
        ratio = ratio[(base.reindex(gi.index) > 0) & (gi["mae"] > 0)].dropna()
        out.append(dict(model=m, cases=len(g), MAE=g.mae.mean(), RMSE=g.rmse.mean(), MAPE=g.mape.mean(),
                        rMAE_vs_naive168=float(np.exp(np.log(ratio).mean())) if len(ratio) else np.nan))
    tab = pd.DataFrame(out).sort_values("MAE")
    print("\n=== top20, identical 8 origins x 20 zones x 168 h ===")
    print(tab.round(2).to_string(index=False))
    show = [m for m in ["naive168", "gbm_full", "sarimax_daily", "sarimax_fourier3", "sarimax_fourier3_weather"]
            if m in set(allc.model)]
    print("\n=== MAE by origin ===")
    print(allc[allc.model.isin(show)].pivot_table(index="origin_idx", columns="model", values="mae")[show].round(1).to_string())
    last = allc[(allc.origin_idx == N_ORIGINS - 1) & allc.model.isin(show)].groupby("model")[["mae", "rmse", "mape"]].mean()
    print("\n=== last origin only (test week 2024-09-24..30). Paper's REFIT at hour 4224: daily 106.94, Fourier 64.87 ===")
    print(last.round(2).to_string())
    if len(meta):
        print("\n=== fits (first-origin fit) ===")
        print(meta.groupby("model").agg(fits=("zone_id", "count"), nonconverged=("converged", lambda s: int((s == 0).sum())),
                                        mean_fit_s=("fit_seconds", "mean"), total_h=("fit_seconds", lambda s: s.sum() / 3600)).round(2).to_string())
        nc = meta[meta.converged == 0][["model", "zone_id", "maxiter_used"]]
        if len(nc):
            print("non-converged:", nc.to_dict("records"))
        w = meta[meta.model == "sarimax_fourier3_weather"]
        if len(w):
            print(f"\nweather terms in the Fourier model ({len(w)} zones): temperature p<0.05 in {(w.temp_p < .05).sum()}, "
                  f"negative sign in {(w.temp_coef < 0).sum()}; precipitation p<0.05 in {(w.precip_p < .05).sum()}, "
                  f"negative sign in {(w.precip_coef < 0).sum()}")


def main():
    if "--summary" in sys.argv:
        return summary()
    load()
    os.makedirs(CACHE, exist_ok=True)
    tasks = [(m, zi) for m in MODELS for zi in range(len(ZONES)) if not os.path.exists(path_for(m, ZONES[zi]))]
    print(f"zones={ZONES}\norigins={[(i, str(pd.Timestamp('2024-04-01') + pd.Timedelta(hours=o))) for i, o in enumerate(ORIGINS)]}\n"
          f"fit on hours [0,{FIRST}), workers={WORKERS}, tasks to run={len(tasks)}", flush=True)
    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(task, t): t for t in tasks}
        for f in as_completed(futs):
            try:
                model, z, meta = f.result()
            except Exception as e:
                print(f"[FAILED] {futs[f]}: {e!r}", flush=True)
                continue
            n += 1
            el = time.time() - t0
            print(f"[{n}/{len(tasks)}] {model} zone {z} conv={meta['converged']} fit={meta['fit_seconds']:.0f}s "
                  f"elapsed={el / 60:.0f}min eta={el / n * (len(tasks) - n) / 60:.0f}min", flush=True)
    summary()


if __name__ == "__main__":
    main()