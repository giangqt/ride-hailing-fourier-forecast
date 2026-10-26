#!/usr/bin/env python3
"""
Generalizability beyond the top-20 zones for the SARIMAX family (reviewer 2).

Same protocol as rolling_sarimax_fixedparam.py (fit once on hours [0, 3048), Kalman-update to 8 weekly origins, 168-h
horizon, K=3) - the models are daily SARIMAX and Fourier K=3 only (the weather variant is not repeated here).
Zone sets come from rolling_cheap_zones.csv (chosen from training-period volume only): from each of mid20 / low20 / tail20,
PICK zones evenly spaced over the zones whose training mean demand is >= 1 trip/hour (rule fixed in advance: a series that
is mostly zeros is not a meaningful SARIMAX target; this exclusion is reported, not hidden).
Usage: python rolling_sarimax_other_zones.py           run / resume, then summary
       python rolling_sarimax_other_zones.py --summary only aggregate
       python rolling_sarimax_other_zones.py --synthetic tiny sanity run
"""
import os
import sys

import rolling_sarimax_fixedparam as m  # sets single-thread BLAS env before numpy is imported

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

SYNTH = m.SYNTH
PICK = 1 if SYNTH else 5
STRATA = ["mid20", "low20", "tail20"]
RUN_MODELS = ["sarimax_fourier3", "sarimax_daily"]
BASE = m.BASE
ZONE_OF = {}


def setup():
    m.MODELS = {k: v for k, v in m.MODELS.items() if k in RUN_MODELS}
    m.CACHE = os.path.join(BASE, "rolling_sarimax_cache_other")
    if SYNTH:
        rng = np.random.default_rng(1)
        t = np.arange(m.T_HOURS)
        base = (1 + .5 * np.sin(2 * np.pi * t / 24)) * (1 + .4 * ((t // 24) % 7 >= 5))
        scales = [90, 70, 50, 30, 20, 10]
        m.ZONES = [101, 102, 201, 202, 301, 302]
        m.Y = np.column_stack([rng.poisson(base * s) for s in scales]).astype(float)
        m.W = np.zeros((m.T_HOURS, 2))
        for z, s in zip(m.ZONES, STRATA * 2):
            ZONE_OF[z] = s
        return
    z = pd.read_csv(os.path.join(BASE, "rolling_cheap_zones.csv"))
    chosen = []
    for s in STRATA:
        e = z[(z.stratum == s) & (z.train_mean_per_hour >= 1.0)].sort_values("train_rank")
        idx = np.unique(np.linspace(0, len(e) - 1, PICK).round().astype(int))
        print(f"[{s}] eligible zones (train mean >= 1/h): {len(e)} of 20; picked {e.iloc[idx].zone_id.tolist()}")
        for zid in e.iloc[idx].zone_id.tolist():
            ZONE_OF[zid] = s
            chosen.append(zid)
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("PGHOST", "localhost"), port=os.environ.get("PGPORT", "5432"),
                            dbname=os.environ.get("PGDATABASE", "rides"), user=os.environ.get("PGUSER", "rides"),
                            password=os.environ.get("PGPASSWORD", "rides"))
    df = pd.read_sql("SELECT time_bucket, zone_id, pickup_count FROM hourly_demand_6mo_all", conn)
    conn.close()
    P = df.pivot(index="time_bucket", columns="zone_id", values="pickup_count").sort_index()
    m.ZONES = chosen
    m.Y = P[chosen].values.astype(float)
    m.W = np.zeros((m.T_HOURS, 2))


def cheap_rows():
    out = []
    for s in STRATA:
        p = os.path.join(BASE, f"rolling_cheap_{s}.csv")
        if os.path.exists(p):
            c = pd.read_csv(p)
            c = c[c.zone_id.isin([z for z, st in ZONE_OF.items() if st == s]) & c.model.isin(["naive168", "gbm_full"])]
            out.append(c[["origin_idx", "zone_id", "model", "mae", "rmse", "mape"]].assign(stratum=s))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summary():
    rows, missing = [], []
    for model in m.MODELS:
        for zi, z in enumerate(m.ZONES):
            p = m.path_for(model, z)
            if not os.path.exists(p):
                missing.append((model, z))
                continue
            d = np.load(p)
            for oi, o in enumerate(m.ORIGINS):
                mae, rmse, mape, _ = m.metrics(m.Y[o:o + m.HORIZON, zi], d["preds"][oi])
                rows.append(dict(origin_idx=oi, zone_id=z, model=model, mae=mae, rmse=rmse, mape=mape,
                                 stratum=ZONE_OF[z], converged=bool(d["converged"])))
    if missing:
        print(f"[note] {len(missing)} results missing, summary partial")
    cases = pd.DataFrame(rows)
    allc = pd.concat([cases.drop(columns="converged"), cheap_rows()], ignore_index=True)
    allc.to_csv(os.path.join(BASE, "rolling_sarimax_other_cases.csv"), index=False)
    pd.set_option("display.width", 200)
    for s in STRATA:
        g = allc[allc.stratum == s]
        zs = sorted(g.zone_id.unique())
        base = g[g.model == "naive168"].set_index(["origin_idx", "zone_id"])["mae"]
        out = []
        for mod, gm in g.groupby("model"):
            gi = gm.set_index(["origin_idx", "zone_id"])
            r = (gi["mae"] / base).replace([np.inf, -np.inf], np.nan)
            r = r[(base.reindex(gi.index) > 0) & (gi["mae"] > 0)].dropna()
            out.append(dict(model=mod, MAE=gm.mae.mean(), RMSE=gm.rmse.mean(), rMAE_vs_naive168=float(np.exp(np.log(r).mean()))))
        print(f"\n=== {s}: zones {zs} (8 origins each) ===")
        print(pd.DataFrame(out).sort_values("MAE").round(2).to_string(index=False))
    nc = cases[~cases.converged].drop_duplicates(["model", "zone_id"])
    print("\nnon-converged first-origin fits:", nc[["model", "zone_id"]].to_dict("records"))


def main():
    setup()
    if "--summary" in sys.argv:
        return summary()
    os.makedirs(m.CACHE, exist_ok=True)
    tasks = [(mod, zi) for mod in m.MODELS for zi in range(len(m.ZONES)) if not os.path.exists(m.path_for(mod, m.ZONES[zi]))]
    print(f"zones={m.ZONES} workers={m.WORKERS} tasks to run={len(tasks)}", flush=True)
    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=m.WORKERS, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(m.task, t): t for t in tasks}
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