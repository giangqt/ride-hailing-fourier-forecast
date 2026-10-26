#!/usr/bin/env python3
"""
Multi-station weather experiment: M1 (baseline, unchanged) vs M2-multi
(SARIMAX-X using per-zone grid-cell weather from zone_weather_multi instead
of the citywide single-station series in hourly_demand).

Same robust fitting (maxiter retry ladder, best log-likelihood kept), same
metrics (MAE/RMSE/MAPE, rainy/non-rainy/anomalous breakdown), same resumable
design as final_weather_experiment.py. The ONLY thing that changes versus
that experiment is the source of the exogenous weather variables -- this
keeps the comparison controlled.

One meaningful side effect: "rainy hour" is now defined by each zone's OWN
grid-cell precipitation reading, not the citywide value -- arguably a more
correct definition of "raining in that zone" than the original experiment
could offer.
"""

import csv
import os
import warnings

import numpy as np
import pandas as pd
import psycopg2
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

TOP_N = 20
TEST_HOURS = 168
SEASONAL = 24
ORDER = (2, 1, 2)
SEASONAL_ORDER = (1, 1, 1, SEASONAL)
MAXITER_LADDER = (50, 200, 500)
RAIN_THRESHOLD_MM = 0.0

OUTPUT_DIR = os.path.expanduser("~/ride-hailing/experiments/weather_sarimax")
PRED_PATH = os.path.join(OUTPUT_DIR, "predictions_multi.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "per_zone_summary_multi.csv")
AGG_PATH = os.path.join(OUTPUT_DIR, "aggregate_multi.csv")

SUMMARY_FIELDS = ["zone_id", "zone_name", "borough",
                   "m1_all_mae", "m1_all_rmse", "m1_all_mape",
                   "m2multi_all_mae", "m2multi_all_rmse", "m2multi_all_mape",
                   "m1_rainy_mae", "m2multi_rainy_mae", "rainy_n",
                   "m1_non_rainy_mae", "m2multi_non_rainy_mae",
                   "m1_anomalous_mae", "m2multi_anomalous_mae", "anomalous_n",
                   "m1_converged", "m2multi_converged"]


def connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "rides"),
        user=os.environ.get("PGUSER", "rides"),
        password=os.environ.get("PGPASSWORD", "rides"),
    )


def load_demand(conn):
    df = pd.read_sql("""
        SELECT time_bucket, zone_id, pickup_count FROM hourly_demand ORDER BY time_bucket""", conn)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"], utc=True)
    return df


def load_zone_weather(conn):
    df = pd.read_sql("""
        SELECT zone_id, observation_time, temperature_c, precipitation_mm
        FROM zone_weather_multi ORDER BY observation_time""", conn)
    df["observation_time"] = pd.to_datetime(df["observation_time"], utc=True)
    return df


def load_zone_meta(conn):
    return pd.read_sql("SELECT zone_id, zone_name, borough FROM taxi_zones", conn).set_index("zone_id")


def load_anomaly_hours(conn):
    try:
        df = pd.read_sql("SELECT DISTINCT detected_at FROM temporal_anomalies", conn)
        return set(pd.to_datetime(df["detected_at"], utc=True).dt.floor("h"))
    except Exception as e:
        print(f"[anomaly join skipped] {e}")
        return set()


def build_zone_frames(demand_df, weather_df):
    full_idx = pd.date_range(demand_df["time_bucket"].min(), demand_df["time_bucket"].max(), freq="h")
    top_zones = (demand_df.groupby("zone_id")["pickup_count"].sum()
                 .sort_values(ascending=False).head(TOP_N).index.tolist())

    zone_frames = {}
    for z in top_zones:
        d = demand_df[demand_df.zone_id == z].set_index("time_bucket").reindex(full_idx)
        d["pickup_count"] = d["pickup_count"].fillna(0.0)

        w = weather_df[weather_df.zone_id == z].set_index("observation_time").reindex(full_idx)
        d["temperature_c"] = w["temperature_c"].ffill().bfill()
        d["precipitation_mm"] = w["precipitation_mm"].fillna(0.0)
        zone_frames[z] = d
    return zone_frames


def mae(a, p):
    return float(np.mean(np.abs(a - p)))


def rmse(a, p):
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(a, p):
    mask = a != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


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


def append_predictions(rows_df):
    is_new = not os.path.exists(PRED_PATH)
    rows_df.to_csv(PRED_PATH, mode="a", header=is_new, index=False)


def append_summary(row):
    is_new = not os.path.exists(SUMMARY_PATH)
    with open(SUMMARY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_zone(zone_id, zdf, anomaly_hours, zone_meta):
    train, test = zdf.iloc[:-TEST_HOURS], zdf.iloc[-TEST_HOURS:]
    exog_cols = ["temperature_c", "precipitation_mm"]
    y_train, y_test = train["pickup_count"], test["pickup_count"].values

    m1_pred, m1_conv = fit_forecast_robust(y_train, TEST_HOURS)
    m2_pred, m2_conv = fit_forecast_robust(
        y_train, TEST_HOURS, exog_train=train[exog_cols], exog_test=test[exog_cols])

    rainy = test["precipitation_mm"].values > RAIN_THRESHOLD_MM
    is_anom = np.array([t in anomaly_hours for t in test.index])

    rows = pd.DataFrame({
        "timestamp": test.index, "zone_id": zone_id, "actual": y_test,
        "m1_pred": m1_pred, "m2multi_pred": m2_pred,
        "temperature_c": test["temperature_c"].values,
        "precipitation_mm": test["precipitation_mm"].values,
        "is_rainy": rainy, "is_anomalous": is_anom,
    })

    def m(mask, col, fn):
        a, p = rows.loc[mask, "actual"].values, rows.loc[mask, col].values
        return fn(a, p) if len(a) else np.nan

    all_mask = np.ones(len(rows), bool)
    zn = zone_meta.loc[zone_id] if zone_id in zone_meta.index else {}
    summary = {
        "zone_id": zone_id,
        "zone_name": zn.get("zone_name", "") if hasattr(zn, "get") else "",
        "borough": zn.get("borough", "") if hasattr(zn, "get") else "",
        "m1_all_mae": m(all_mask, "m1_pred", mae), "m1_all_rmse": m(all_mask, "m1_pred", rmse),
        "m1_all_mape": m(all_mask, "m1_pred", mape),
        "m2multi_all_mae": m(all_mask, "m2multi_pred", mae), "m2multi_all_rmse": m(all_mask, "m2multi_pred", rmse),
        "m2multi_all_mape": m(all_mask, "m2multi_pred", mape),
        "m1_rainy_mae": m(rainy, "m1_pred", mae), "m2multi_rainy_mae": m(rainy, "m2multi_pred", mae),
        "rainy_n": int(rainy.sum()),
        "m1_non_rainy_mae": m(~rainy, "m1_pred", mae), "m2multi_non_rainy_mae": m(~rainy, "m2multi_pred", mae),
        "m1_anomalous_mae": m(is_anom, "m1_pred", mae), "m2multi_anomalous_mae": m(is_anom, "m2multi_pred", mae),
        "anomalous_n": int(is_anom.sum()),
        "m1_converged": m1_conv, "m2multi_converged": m2_conv,
    }
    return rows, summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    done = already_done_zones()
    if done:
        print(f"[resume] {len(done)} zones already done, skipping: {sorted(done)}")

    conn = connect()
    demand_df = load_demand(conn)
    weather_df = load_zone_weather(conn)
    zone_meta = load_zone_meta(conn)
    anomaly_hours = load_anomaly_hours(conn)
    conn.close()

    zone_frames = build_zone_frames(demand_df, weather_df)
    remaining = {z: zdf for z, zdf in zone_frames.items() if z not in done}
    print(f"[zones] {len(remaining)} remaining to process; {len(anomaly_hours)} anomalous hours known")

    for z, zdf in remaining.items():
        try:
            rows, summary = run_zone(z, zdf, anomaly_hours, zone_meta)
        except Exception as e:
            print(f"[zone {z}] FAILED: {e}")
            continue
        append_predictions(rows)
        append_summary(summary)
        print(f"[zone {z}] M1 MAE={summary['m1_all_mae']:.1f}  M2-multi MAE={summary['m2multi_all_mae']:.1f}  "
              f"rainy(n={summary['rainy_n']}) M1={summary['m1_rainy_mae']:.1f} M2multi={summary['m2multi_rainy_mae']:.1f}")

    summary_df = pd.read_csv(SUMMARY_PATH)
    if len(summary_df) < TOP_N:
        print(f"\n[incomplete] {len(summary_df)}/{TOP_N} zones done so far -- rerun to continue.")
        return

    agg = {c: summary_df[c].mean(skipna=True) for c in summary_df.columns
           if c not in ("zone_id", "zone_name", "borough", "m1_converged", "m2multi_converged")}
    pd.DataFrame([agg]).to_csv(AGG_PATH, index=False)

    print("\n=== FINAL AGGREGATE (mean across 20 zones) -- MULTI-STATION WEATHER ===")
    print(f"M1 (no weather)      MAE={agg['m1_all_mae']:.2f}  RMSE={agg['m1_all_rmse']:.2f}  MAPE={agg['m1_all_mape']:.2f}%")
    print(f"M2-multi (per-zone)  MAE={agg['m2multi_all_mae']:.2f}  RMSE={agg['m2multi_all_rmse']:.2f}  MAPE={agg['m2multi_all_mape']:.2f}%")
    print(f"Rainy hours -> M1={agg['m1_rainy_mae']:.2f}  M2-multi={agg['m2multi_rainy_mae']:.2f}  (avg n/zone={agg['rainy_n']:.0f})")
    print(f"Non-rainy   -> M1={agg['m1_non_rainy_mae']:.2f}  M2-multi={agg['m2multi_non_rainy_mae']:.2f}")
    print(f"Anomalous   -> M1={agg['m1_anomalous_mae']:.2f}  M2-multi={agg['m2multi_anomalous_mae']:.2f}  (avg n/zone={agg['anomalous_n']:.0f})")

    from scipy import stats
    diff = summary_df["m2multi_all_mae"].values - summary_df["m1_all_mae"].values
    w_stat, w_p = stats.wilcoxon(diff)
    print(f"\nWilcoxon signed-rank test (M2-multi vs M1): p={w_p:.4f}  "
          f"({(diff < 0).sum()}/{TOP_N} zones improved)")

    print(f"\nWrote:\n  {PRED_PATH}\n  {SUMMARY_PATH}\n  {AGG_PATH}")


if __name__ == "__main__":
    main()