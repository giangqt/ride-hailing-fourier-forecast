#!/usr/bin/env python3
"""
Full 20-zone test of Fourier weekly-seasonality terms (K=3 harmonics,
the pilot's best-performing config) vs plain SARIMAX(2,1,2)(1,1,1,24),
on the 6-month dataset (~4224 training hours, ~25 weekly cycles).

Safe by construction: exogenous regression only, no seasonal_order change,
so no risk of the state-space memory blowup that killed the native s=168
attempt. Resumable (per-zone CSV append) given today's history of
interruptions on longer-running scripts.
"""

import csv
import os
import time
import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

TOP_N_ZONES = [138, 132, 79, 61, 230, 161, 231, 68, 246, 76, 37, 234, 255, 164, 148, 181, 249, 7, 48, 112]
TEST_HOURS = 168
ORDER = (2, 1, 2)
SEASONAL_ORDER = (1, 1, 1, 24)
WEEKLY_PERIOD = 168
K_HARMONICS = 3
MAXITER_LADDER = (50, 200, 500)

OUTPUT_DIR = os.path.expanduser("~/ride-hailing/experiments/weather_sarimax")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "fourier_weekly_full_summary.csv")
FIELDNAMES = ["zone_id", "baseline_mae", "baseline_rmse", "baseline_mape", "baseline_converged",
              "fourier_mae", "fourier_rmse", "fourier_mape", "fourier_converged", "fit_seconds"]


def connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "rides"),
        user=os.environ.get("PGUSER", "rides"),
        password=os.environ.get("PGPASSWORD", "rides"),
    )


def load_zone_series(conn, zone_id):
    df = pd.read_sql("""
        SELECT time_bucket, pickup_count FROM hourly_demand_6mo
        WHERE zone_id = %s ORDER BY time_bucket""", conn, params=(zone_id,))
    df["time_bucket"] = pd.to_datetime(df["time_bucket"], utc=True)
    return df.set_index("time_bucket")["pickup_count"].astype(float)


def fourier_terms(n, k, period=WEEKLY_PERIOD):
    t = np.arange(n)
    cols = {}
    for i in range(1, k + 1):
        cols[f"sin_{i}"] = np.sin(2 * np.pi * i * t / period)
        cols[f"cos_{i}"] = np.cos(2 * np.pi * i * t / period)
    return pd.DataFrame(cols)


def mae(a, p):
    return float(np.mean(np.abs(a - p)))


def rmse(a, p):
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(a, p):
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.sum() else np.nan


def fit_forecast_robust(train_y, test_len, exog_train=None, exog_test=None):
    best_fit, best_conv = None, False
    for maxiter in MAXITER_LADDER:
        model = SARIMAX(train_y, order=ORDER, seasonal_order=SEASONAL_ORDER,
                         exog=exog_train, enforce_stationarity=False, enforce_invertibility=False)
        candidate = model.fit(disp=False, maxiter=maxiter)
        conv = bool(candidate.mle_retvals.get("converged", True)) if hasattr(candidate, "mle_retvals") else True
        if best_fit is None or candidate.llf > best_fit.llf:
            best_fit, best_conv = candidate, conv
        if conv:
            break
    fc = best_fit.get_forecast(test_len, exog=exog_test).predicted_mean
    return np.clip(fc.values, 0, None), best_conv


def already_done_zones():
    if not os.path.exists(SUMMARY_PATH):
        return set()
    return set(pd.read_csv(SUMMARY_PATH)["zone_id"].astype(int).tolist())


def append_result(row):
    is_new = not os.path.exists(SUMMARY_PATH)
    with open(SUMMARY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    done = already_done_zones()
    if done:
        print(f"[resume] {len(done)} zones already done, skipping: {sorted(done)}")

    conn = connect()
    remaining = [z for z in TOP_N_ZONES if z not in done]
    print(f"[zones] {len(remaining)} remaining to process\n")

    for z in remaining:
        t0 = time.time()
        series = load_zone_series(conn, z)
        n_total = len(series)
        train, test = series.iloc[:-TEST_HOURS], series.iloc[-TEST_HOURS:]

        base_pred, base_conv = fit_forecast_robust(train, TEST_HOURS)

        fterms = fourier_terms(n_total, K_HARMONICS)
        fterms.index = series.index
        exog_train, exog_test = fterms.iloc[:-TEST_HOURS], fterms.iloc[-TEST_HOURS:]
        fourier_pred, fourier_conv = fit_forecast_robust(train, TEST_HOURS, exog_train, exog_test)

        elapsed = time.time() - t0
        actual = test.values
        row = {
            "zone_id": z,
            "baseline_mae": mae(actual, base_pred), "baseline_rmse": rmse(actual, base_pred),
            "baseline_mape": mape(actual, base_pred), "baseline_converged": base_conv,
            "fourier_mae": mae(actual, fourier_pred), "fourier_rmse": rmse(actual, fourier_pred),
            "fourier_mape": mape(actual, fourier_pred), "fourier_converged": fourier_conv,
            "fit_seconds": round(elapsed, 1),
        }
        append_result(row)
        print(f"[zone {z}] baseline MAE={row['baseline_mae']:.1f}  "
              f"fourier(K={K_HARMONICS}) MAE={row['fourier_mae']:.1f}  "
              f"({elapsed:.0f}s)")

    conn.close()

    summary_df = pd.read_csv(SUMMARY_PATH)
    if len(summary_df) < len(TOP_N_ZONES):
        print(f"\n[incomplete] {len(summary_df)}/{len(TOP_N_ZONES)} zones done -- rerun to continue.")
        return

    print("\n=== FINAL AGGREGATE (mean across 20 zones) ===")
    print(f"Baseline (no Fourier)   MAE={summary_df['baseline_mae'].mean():.2f}  "
          f"RMSE={summary_df['baseline_rmse'].mean():.2f}  MAPE={summary_df['baseline_mape'].mean():.2f}%")
    print(f"Fourier (K={K_HARMONICS})           MAE={summary_df['fourier_mae'].mean():.2f}  "
          f"RMSE={summary_df['fourier_rmse'].mean():.2f}  MAPE={summary_df['fourier_mape'].mean():.2f}%")

    diff = summary_df["fourier_mae"].values - summary_df["baseline_mae"].values
    w_stat, w_p = stats.wilcoxon(diff)
    print(f"\nZones where Fourier terms improved MAE: {(diff < 0).sum()}/{len(TOP_N_ZONES)}")
    print(f"Wilcoxon signed-rank test: p={w_p:.4f}")
    print(f"Non-convergence: baseline={sum(~summary_df['baseline_converged'])}  "
          f"fourier={sum(~summary_df['fourier_converged'])}")
    print(f"Total fit time: {summary_df['fit_seconds'].sum():.0f}s "
          f"({summary_df['fit_seconds'].sum()/60:.1f} min)")

    print(f"\nWrote: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()