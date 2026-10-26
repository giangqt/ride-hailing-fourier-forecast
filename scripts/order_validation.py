#!/usr/bin/env python3
"""
Is the inherited SARIMAX order (2,1,2)(1,1,1,24) a reasonable choice? Validation-only check (reviewer 1).

The order came from the pre-existing production pipeline and was never tuned on this data. Here four alternative orders are
compared against it on the same validation protocol as fourier_k_validation.py (Fourier K=3 terms, fit on hours [0, 2712),
two validation weeks before the first test origin 2024-08-06), on 8 zones evenly spaced over the training-volume ranking of the
top-20. The current order's results are reused from fourier_k_validation.csv (K=3), not refitted.
Decision rule fixed in advance: keep the current order unless an alternative has mean validation MAE lower by more than 2%
AND lower mean BIC.
Usage: python order_validation.py            run / resume
       python order_validation.py --summary
       python order_validation.py --synthetic
"""
import fourier_k_validation as kv  # sets single-thread BLAS env before numpy is imported

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ORDERS = {"(1,1,1)(1,1,1,24)": ((1, 1, 1), (1, 1, 1, 24)),
          "(1,1,1)(0,1,1,24)": ((1, 1, 1), (0, 1, 1, 24)),
          "(2,1,2)(0,1,1,24)": ((2, 1, 2), (0, 1, 1, 24)),
          "(3,1,2)(1,1,1,24)": ((3, 1, 2), (1, 1, 1, 24))}
CURRENT = "(2,1,2)(1,1,1,24)"
K = 2 if kv.SYNTH else 3
N_ZONES = 8
OUT = kv.OUT.replace("fourier_k_validation", "order_validation")
REF = kv.OUT
SUBSET = None


def fit_ladder(y, exog, order, sorder):
    best, conv, used = None, False, None
    for mi in kv.LADDER:
        r = SARIMAX(y, order=order, seasonal_order=sorder, exog=exog, enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False, maxiter=mi)
        c = bool(r.mle_retvals.get("converged", True))
        if best is None or r.llf > best.llf:
            best, conv, used = r, c, mi
        if c:
            break
    return best, conv, used


def task(arg):
    zi, name = arg
    order, sorder = ORDERS[name]
    y, ex, nf, V = kv.Y[:, zi], kv.fourier(kv.T_HOURS, K), kv.N_FIT, kv.VAL
    t0 = time.perf_counter()
    res, conv, used = fit_ladder(y[:nf], ex[:nf], order, sorder)
    fc1 = res.get_forecast(V, exog=ex[nf:nf + V]).predicted_mean
    res2 = res.apply(y[:nf + V], exog=ex[:nf + V], refit=False)
    fc2 = res2.get_forecast(V, exog=ex[nf + V:nf + 2 * V]).predicted_mean
    out = {"zone_id": kv.ZONES[zi], "order": name, "seconds": time.perf_counter() - t0, "converged": conv,
           "maxiter_used": used, "bic": float(res.bic)}
    for w, (fc, lo) in enumerate([(fc1, nf), (fc2, nf + V)], start=1):
        out[f"mae_w{w}"] = float(np.abs(y[lo:lo + V] - np.clip(np.asarray(fc), 0, None)).mean())
    return out


def summary():
    d = pd.read_csv(OUT).drop_duplicates(["zone_id", "order"])
    ref = pd.read_csv(REF)
    ref = ref[(ref.K == K) & ref.zone_id.isin(d.zone_id.unique())].drop_duplicates("zone_id").assign(order=CURRENT)
    d = pd.concat([ref[d.columns.intersection(ref.columns)], d], ignore_index=True)
    d["val_mae"] = (d.mae_w1 + d.mae_w2) / 2
    g = d.groupby("order").agg(zones=("zone_id", "nunique"), val_MAE=("val_mae", "mean"), mean_BIC=("bic", "mean"),
                               nonconverged=("converged", lambda s: int((~s.astype(bool)).sum())),
                               mean_fit_s=("seconds", "mean"))
    cur = g.loc[CURRENT]
    g["MAE_vs_current_%"] = 100 * (g.val_MAE / cur.val_MAE - 1)
    pd.set_option("display.width", 200)
    print(g.round(2).to_string())
    better = g[(g["MAE_vs_current_%"] < -2) & (g.mean_BIC < cur.mean_BIC) & (g.index != CURRENT)]
    print("\nRule (alternative must be >2% better in validation MAE AND have lower mean BIC):",
          better.index.tolist() if len(better) else "no alternative qualifies -> keep the current order")


def main():
    if "--summary" in sys.argv:
        kv.load()
        return summary()
    kv.load()
    global SUBSET
    SUBSET = sorted(set(np.linspace(0, len(kv.ZONES) - 1, min(N_ZONES, len(kv.ZONES))).round().astype(int).tolist()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.zone_id, prev.order))
    tasks = [(zi, o) for o in ORDERS for zi in SUBSET if (kv.ZONES[zi], o) not in done]
    print(f"zones={[kv.ZONES[i] for i in SUBSET]} orders={list(ORDERS)} tasks to run={len(tasks)}", flush=True)
    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=kv.WORKERS, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(task, t): t for t in tasks}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:
                print(f"[FAILED] {futs[f]}: {e!r}", flush=True)
                continue
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            n += 1
            el = time.time() - t0
            print(f"[{n}/{len(tasks)}] zone {r['zone_id']} {r['order']} conv={r['converged']} "
                  f"MAE w1/w2={r['mae_w1']:.1f}/{r['mae_w2']:.1f} fit={r['seconds']:.0f}s "
                  f"elapsed={el / 60:.0f}min eta={el / n * (len(tasks) - n) / 60:.0f}min", flush=True)
    print()
    summary()


if __name__ == "__main__":
    main()
