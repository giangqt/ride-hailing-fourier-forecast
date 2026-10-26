#!/usr/bin/env python3
"""
Is there evidence for fortnightly (336 h) or monthly (~720 h) cycles beyond the weekly one? (reviewer 2)

Training data only (hours [0, 3048), before the first test origin).
For each of the 20 zones and for their sum:
  1. remove the mean hour-of-week profile (168 values), which absorbs daily + weekly seasonality exactly
  2. periodogram of the residual (zero-padded x10 for a fine frequency grid)
  3. compare power at the candidate period with the MEDIAN power over periods 200-1500 h
     (a local reference that follows the red noise level), ratio = power(P) / median
Rule of thumb under a white-noise null the ratio exceeds 6.6 with probability about 0.01 at one frequency; residuals are
autocorrelated, so this is a screening threshold, not an exact test.
Control: the same ratio on the RAW series at 24 h and 168 h (should be huge).
Also prints how many cycles of each period the training window actually contains.
"""
import os

import numpy as np
import pandas as pd
import psycopg2

FIRST, PERIODS, THRESH = 3048, (336, 720), 6.6
EXCLUDE = {264, 265}


def load():
    conn = psycopg2.connect(host=os.environ.get("PGHOST", "localhost"), port=os.environ.get("PGPORT", "5432"),
                            dbname=os.environ.get("PGDATABASE", "rides"), user=os.environ.get("PGUSER", "rides"),
                            password=os.environ.get("PGPASSWORD", "rides"))
    df = pd.read_sql("SELECT time_bucket, zone_id, pickup_count FROM hourly_demand_6mo_all", conn)
    conn.close()
    P = df.pivot(index="time_bucket", columns="zone_id", values="pickup_count").sort_index()
    vol = P.iloc[:FIRST].sum().drop(labels=list(EXCLUDE), errors="ignore").sort_values(ascending=False, kind="stable")
    return P[vol.index[:20]].iloc[:FIRST].values.astype(float), vol.index[:20].tolist()


def spectrum(x, pad=10):
    n = len(x)
    x = x - x.mean()
    f = np.fft.rfft(x, n=n * pad)
    freq = np.fft.rfftfreq(n * pad, 1.0)[1:]
    return 1.0 / freq, (np.abs(f[1:]) ** 2) / n


def ratio_at(x, period):
    per, pw = spectrum(x)
    ref = np.median(pw[(per >= 200) & (per <= 1500)])
    return float(pw[np.argmin(np.abs(per - period))] / ref)


def residual(x):
    how = np.arange(len(x)) % 168
    prof = np.array([x[how == h].mean() for h in range(168)])
    return x - prof[how]


def analyse(X):
    rows = []
    for j in range(X.shape[1]):
        r = residual(X[:, j])
        rows.append({**{f"resid_{p}h": ratio_at(r, p) for p in PERIODS},
                     "raw_24h": ratio_at(X[:, j], 24), "raw_168h": ratio_at(X[:, j], 168)})
    tot = X.sum(1)
    agg = {**{f"resid_{p}h": ratio_at(residual(tot), p) for p in PERIODS},
           "raw_24h": ratio_at(tot, 24), "raw_168h": ratio_at(tot, 168)}
    return pd.DataFrame(rows), agg


def main():
    X, zones = load()
    n = len(X)
    print(f"training hours={n} ({n / 24:.0f} days): contains {n / 168:.1f} weekly, {n / 336:.1f} fortnightly (336 h), "
          f"{n / 720:.1f} monthly (720 h) cycles")
    d, agg = analyse(X)
    d.insert(0, "zone_id", zones)
    pd.set_option("display.width", 200)
    print("\npower ratio vs local median (threshold ~%.1f):" % THRESH)
    print(d.round(1).to_string(index=False))
    print("\nsum of the 20 zones:", {k: round(v, 1) for k, v in agg.items()})
    for p in PERIODS:
        c = d[f"resid_{p}h"]
        print(f"period {p} h: median ratio across zones = {c.median():.1f}; zones above {THRESH} = {(c > THRESH).sum()}/20")
    d.to_csv(os.path.expanduser("~/ride-hailing/experiments/weather_sarimax/seasonality_spectrum.csv"), index=False)


if __name__ == "__main__":
    main()