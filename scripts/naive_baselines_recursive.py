#!/usr/bin/env python3
"""Corrected naive baselines. The previous naive-24h sliced s[-192:-24], so for test hours beyond the
first 24 it read real test-period values (leakage). Fix: recursive construction -- the first 24 test
hours use the last real training day; every later hour copies the baseline's own prediction from 24h
earlier. naive-168h is unchanged (168h before any test hour is always inside the training period)."""
import os, numpy as np, pandas as pd, psycopg2
TOP_N = 20; TEST_HOURS = 168
def connect():
    return psycopg2.connect(host=os.environ.get("PGHOST","localhost"), port=os.environ.get("PGPORT","5432"),
        dbname=os.environ.get("PGDATABASE","rides"), user=os.environ.get("PGUSER","rides"),
        password=os.environ.get("PGPASSWORD","rides"))
def mae(a,p): return float(np.mean(np.abs(a-p)))
def rmse(a,p): return float(np.sqrt(np.mean((a-p)**2)))
def mape(a,p):
    m = a != 0
    return float(np.mean(np.abs((a[m]-p[m])/a[m]))*100) if m.sum() else np.nan
def main():
    conn = connect()
    df = pd.read_sql("SELECT time_bucket, zone_id, pickup_count FROM hourly_demand ORDER BY zone_id, time_bucket", conn)
    conn.close()
    df["time_bucket"] = pd.to_datetime(df["time_bucket"], utc=True)
    zones = df.groupby("zone_id")["pickup_count"].sum().sort_values(ascending=False).head(TOP_N).index.tolist()
    idx = pd.date_range(df["time_bucket"].min(), df["time_bucket"].max(), freq="h")
    rows = []
    for z in zones:
        s = df[df.zone_id == z].set_index("time_bucket").reindex(idx)["pickup_count"].fillna(0.0).values
        actual, train = s[-TEST_HOURS:], s[:-TEST_HOURS]
        n24 = np.zeros(TEST_HOURS)
        for h in range(TEST_HOURS):
            n24[h] = train[-24 + h] if h < 24 else n24[h - 24]
        n168 = train[-168:]
        rows.append({"zone_id": z, "naive24_fixed_mae": mae(actual,n24), "naive24_fixed_rmse": rmse(actual,n24),
            "naive24_fixed_mape": mape(actual,n24), "naive168_mae": mae(actual,n168),
            "naive168_rmse": rmse(actual,n168), "naive168_mape": mape(actual,n168)})
        print(f"[zone {z}] naive-24h(FIXED) MAE={rows[-1]['naive24_fixed_mae']:.1f}  naive-168h MAE={rows[-1]['naive168_mae']:.1f}")
    r = pd.DataFrame(rows)
    print("\n=== AGGREGATE ===")
    print(f"Naive-24h FIXED  MAE={r['naive24_fixed_mae'].mean():.2f} RMSE={r['naive24_fixed_rmse'].mean():.2f} MAPE={r['naive24_fixed_mape'].mean():.2f}%")
    print(f"Naive-168h       MAE={r['naive168_mae'].mean():.2f} RMSE={r['naive168_rmse'].mean():.2f} MAPE={r['naive168_mape'].mean():.2f}%")
    out = os.path.expanduser("~/ride-hailing/experiments/weather_sarimax/naive_baselines_recursive.csv")
    r.to_csv(out, index=False); print("Wrote:", out)
if __name__ == "__main__": main()
