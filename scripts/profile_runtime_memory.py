#!/usr/bin/env python3
"""
Measured runtime / peak-memory profile: daily SARIMAX (s=24) vs Fourier-SARIMAX (K=3) vs native weekly SARIMAX (s=168).

Answers reviewer requests for runtime/memory tables. One representative zone (default 230), training length 4224 h.
Each job runs in its own subprocess with an address-space cap, so an oversized native s=168 job raises MemoryError
instead of getting the VM OOM-killed (the failure that originally killed a run at ~13 GB).

Jobs
  daily_fit / fourier_fit      : real fit(maxiter=50) at n=4224 -> wall time, peak RSS, k_states, converged
  native_loglike_n{N}          : ONE log-likelihood evaluation of s=168 at n=N (time, peak RSS)
  native_lowmem_iter1_n{N}     : fit(low_memory=True, maxiter=1) of s=168 -> time of ONE optimizer iteration, peak RSS
  native_smooth_n{N}           : model.smooth() of s=168 at n=N (this is the final step of fit() that builds the
                                 results object and stores the state covariances), time + peak RSS
The native model is NOT fitted at n=4224: its cost is extrapolated from N=672 and 1344 and printed as an estimate.

Usage:  python profile_runtime_memory.py                 (driver; runs all jobs)
        python profile_runtime_memory.py --synthetic     (no DB, tiny sizes; sanity test of the script)
"""
import json
import os
import resource
import subprocess
import sys
import time

CAP_GB = float(os.environ.get("PROFILE_CAP_GB", "9"))
ZONE = int(os.environ.get("PROFILE_ZONE", "230"))
N_TRAIN = 4224
SMOOTH_NS = [240, 480]
LOGLIKE_NS = [1344, N_TRAIN]
LOWMEM_N = 1344
OUT = os.path.expanduser("~/ride-hailing/experiments/weather_sarimax/profile_runtime_memory.json")
THREADS = os.environ.get("PROFILE_THREADS", "4")
SYNTH = "--synthetic" in sys.argv
if SYNTH:
    N_TRAIN, SMOOTH_NS, LOGLIKE_NS, LOWMEM_N, OUT = 600, [120, 240], [240, 600], 240, "/tmp/profile_synth.json"


def series():
    import numpy as np
    if SYNTH:
        rng = np.random.default_rng(0)
        t = np.arange(N_TRAIN + 10)
        return rng.poisson(100 * (1 + .5 * np.sin(2 * np.pi * t / 24)) * (1 + .3 * (t // 24 % 7 >= 5))).astype(float)
    import pandas as pd
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("PGHOST", "localhost"), port=os.environ.get("PGPORT", "5432"),
                            dbname=os.environ.get("PGDATABASE", "rides"), user=os.environ.get("PGUSER", "rides"),
                            password=os.environ.get("PGPASSWORD", "rides"))
    df = pd.read_sql(f"SELECT pickup_count FROM hourly_demand_6mo_all WHERE zone_id={ZONE} ORDER BY time_bucket", conn)
    conn.close()
    assert len(df) == 4392
    return df["pickup_count"].values.astype(float)


def fourier(n, k=3, period=168):
    import numpy as np
    t = np.arange(n)
    return np.column_stack([f(2 * np.pi * i * t / period) for i in range(1, k + 1) for f in (np.sin, np.cos)])


def run_job(name):
    resource.setrlimit(resource.RLIMIT_AS, (int(CAP_GB * 2**30), int(CAP_GB * 2**30)))
    import warnings
    warnings.filterwarnings("ignore")
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = series()
    rec = {"job": name}
    try:
        kw = dict(order=(2, 1, 2), enforce_stationarity=False, enforce_invertibility=False)
        if name in ("daily_fit", "fourier_fit"):
            n = N_TRAIN
            exog = fourier(n) if name == "fourier_fit" else None
            model = SARIMAX(y[:n], seasonal_order=(1, 1, 1, 24), exog=exog, **kw)
            rec.update(n=n, k_states=model.k_states)
            t0 = time.perf_counter()
            res = model.fit(disp=False, maxiter=50)
            rec.update(seconds=time.perf_counter() - t0, converged=bool(res.mle_retvals.get("converged", True)),
                       llf=float(res.llf))
        else:
            kind, nstr = name.rsplit("_n", 1)
            n = int(nstr)
            model = SARIMAX(y[:n], seasonal_order=(1, 1, 1, 168), **kw)
            rec.update(n=n, k_states=model.k_states)
            start = model.start_params
            t0 = time.perf_counter()
            if kind == "native_loglike":
                model.loglike(start)
            elif kind == "native_lowmem_iter1":
                res = model.fit(disp=False, maxiter=1, low_memory=True)
            else:
                model.smooth(start)
            rec.update(seconds=time.perf_counter() - t0)
        rec["status"] = "ok"
    except MemoryError:
        rec["status"] = f"MemoryError at {CAP_GB:g} GB cap"
    except Exception as e:
        rec["status"] = f"error: {e!r}"
    rec["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print("RESULT " + json.dumps(rec), flush=True)


def fmt(r):
    s = f"{r['job']:<22} k_states={r.get('k_states', '-'):>4}  n={r.get('n', '-'):>5}  "
    s += f"time={r['seconds']:>8.1f}s  " if "seconds" in r else "time=       -   "
    return s + f"peak_rss={r['peak_rss_gb']:.2f} GB  [{r['status']}]"


def driver():
    jobs = ["daily_fit", "fourier_fit"]
    jobs += [f"native_smooth_n{n}" for n in SMOOTH_NS] + [f"native_loglike_n{n}" for n in LOGLIKE_NS]
    jobs += [f"native_lowmem_iter1_n{LOWMEM_N}"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS=THREADS, OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS)
    mem = [l for l in open("/proc/meminfo") if l.startswith(("MemTotal", "SwapTotal"))]
    print(f"cpus={os.cpu_count()} threads={THREADS} cap={CAP_GB:g} GB zone={ZONE} | " + " | ".join(m.strip() for m in mem))
    results = []
    for j in jobs:
        print(f"[run] {j} ...", flush=True)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--job", j] + (["--synthetic"] if SYNTH else []),
                           capture_output=True, text=True, env=env, timeout=4 * 3600)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if line:
            r = json.loads(line[-1][7:])
        else:
            r = {"job": j, "status": f"child died (exit {p.returncode}): {p.stderr[-200:]}", "peak_rss_gb": float("nan")}
        results.append(r)
        print(fmt(r), flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(results, open(OUT, "w"), indent=1)

    by = {r["job"]: r for r in results}
    print("\n=== SUMMARY ===")
    for r in results:
        print(fmt(r))
    try:
        a, b = SMOOTH_NS
        sa, sb = by[f"native_smooth_n{a}"], by[f"native_smooth_n{b}"]
        slope = (sb["peak_rss_gb"] - sa["peak_rss_gb"]) / (b - a)
        print("\n[ESTIMATES by linear extrapolation in n, not measurements]")
        print(f"native s=168 default fit (smooth step) peak memory at n={N_TRAIN}: ~{sb['peak_rss_gb'] + slope * (N_TRAIN - b):.1f} GB "
              f"(slope {slope * 1000:.2f} MB per hour of data, k_states={sb['k_states']})")
        print(f"theory: one stored (k,k,n) covariance array = 8*k^2*n = {8 * sb['k_states'] ** 2 * N_TRAIN / 2**30:.2f} GB at n={N_TRAIN}")
        it = by.get(f"native_lowmem_iter1_n{LOWMEM_N}", {})
        if "seconds" in it:
            t4 = it["seconds"] * N_TRAIN / LOWMEM_N
            print(f"native s=168 low_memory fit, ONE optimizer iteration: {it['seconds']:.0f} s measured at n={LOWMEM_N}, "
                  f"~{t4 / 60:.0f} min extrapolated to n={N_TRAIN} -> 50 iterations ~{50 * t4 / 3600:.0f} h, "
                  f"500 iterations ~{500 * t4 / 3600:.0f} h per fit (estimate)")
    except Exception as e:
        print("[extrapolation skipped]", e)
    print("Wrote:", OUT)


if __name__ == "__main__":
    if "--job" in sys.argv:
        run_job(sys.argv[sys.argv.index("--job") + 1])
    else:
        driver()