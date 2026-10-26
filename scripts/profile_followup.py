#!/usr/bin/env python3
"""
Follow-up to profile_runtime_memory.py: like-for-like timings for the daily and Fourier models, using the SAME two
operations already measured for native s=168 (so the three models can be compared in one table):
  (1) one log-likelihood evaluation at n=4224            [native s=168 measured: 27.1 s, 0.23 GB]
  (2) fit(low_memory=True, maxiter=1) at n=1344          [native s=168 measured: 1859.8 s, 0.24 GB]
Runs on the same zone (default 230). Usage: python profile_followup.py [--synthetic]
"""
import os
import resource
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX

SYNTH = "--synthetic" in sys.argv
ZONE = int(os.environ.get("PROFILE_ZONE", "230"))
N_LL, N_IT = (600, 240) if SYNTH else (4224, 1344)


def series():
    if SYNTH:
        t = np.arange(N_LL + 10)
        return np.random.default_rng(0).poisson(100 * (1 + .5 * np.sin(2 * np.pi * t / 24))).astype(float)
    import pandas as pd
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("PGHOST", "localhost"), port=os.environ.get("PGPORT", "5432"),
                            dbname=os.environ.get("PGDATABASE", "rides"), user=os.environ.get("PGUSER", "rides"),
                            password=os.environ.get("PGPASSWORD", "rides"))
    df = pd.read_sql(f"SELECT pickup_count FROM hourly_demand_6mo_all WHERE zone_id={ZONE} ORDER BY time_bucket", conn)
    conn.close()
    return df["pickup_count"].values.astype(float)


def fourier(n, k=3, period=168):
    t = np.arange(n)
    return np.column_stack([f(2 * np.pi * i * t / period) for i in range(1, k + 1) for f in (np.sin, np.cos)])


def make(y, n, fourier_terms):
    return SARIMAX(y[:n], order=(2, 1, 2), seasonal_order=(1, 1, 1, 24),
                   exog=fourier(n) if fourier_terms else None,
                   enforce_stationarity=False, enforce_invertibility=False)


def main():
    y = series()
    print(f"zone={ZONE} n_loglike={N_LL} n_iter={N_IT} synthetic={SYNTH}")
    rows = []
    for name, ft in [("daily s=24", False), ("fourier K=3", True)]:
        m = make(y, N_LL, ft)
        p = m.start_params
        m.loglike(p)
        t0 = time.perf_counter()
        for _ in range(3):
            m.loglike(p)
        ll = (time.perf_counter() - t0) / 3
        t0 = time.perf_counter()
        make(y, N_IT, ft).fit(disp=False, maxiter=1, low_memory=True)
        it = time.perf_counter() - t0
        rows.append((name, ll, it))
        print(f"{name:<12} loglike(n={N_LL}) = {ll:6.2f} s   fit(low_memory, maxiter=1, n={N_IT}) = {it:7.1f} s", flush=True)
    print(f"{'native s=168':<12} loglike(n={N_LL}) =  27.10 s   fit(low_memory, maxiter=1, n={N_IT}) =  1859.8 s   (measured earlier)")
    if not SYNTH:
        d = rows[0]
        print(f"\nratios native/daily: loglike x{27.1 / d[1]:.0f}, one-iteration fit x{1859.8 / d[2]:.0f}")
    print(f"peak RSS this process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.2f} GB")


if __name__ == "__main__":
    main()