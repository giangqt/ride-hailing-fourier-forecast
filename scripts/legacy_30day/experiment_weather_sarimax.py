#!/usr/bin/env python3
"""
Experiment: does weather (temperature + precipitation) as a SARIMAX exogenous
input improve short-term ride-hailing demand forecasting, especially during
rainy hours?

Controlled comparison (per experimental design):
  M1 - SARIMAX(2,1,2)(1,1,1,24), demand history only
  M2 - SARIMAX-X: identical order, + temperature/precipitation exogenous regressors

Same 20 zones, same 23-day train / 7-day test split as the production
forecasting layer (scripts/forecast_demand.py). This script is standalone —
it does NOT overwrite models/*.pkl or forecast_results (the live dashboard
path). Outputs are written to experiments/weather_sarimax/.

Retrospective note: test-period weather is the actual recorded weather, not
a forecast of weather, so results are "weather-informed retrospective
forecasting," not a complete operational forecast system.
"""

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
OUTPUT_DIR = os.path.expanduser("~/ride-hailing/experiments/weather_sarimax")
RAIN_THRESHOLD_MM = 0.0  # rainy hour := precipitation_mm > this


def connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "rides"),
        user=os.environ.get("PGUSER", "rides"),
        password=os.environ.get("PGPASSWORD", "rides"),
    )


def load_data(conn):
    """Full hourly demand + weather series, reindexed to a continuous hourly grid per zone."""
    df = pd.read_sql("""
        SELECT time_bucket, zone_id, pickup_count, avg_temperature, precipitation_mm
        FROM hourly_demand ORDER BY time_bucket""", conn)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    full_idx = pd.date_range(df["time_bucket"].min(), df["time_bucket"].max(), freq="h")

    top_zones = (df.groupby("zone_id")["pickup_count"].sum()
                   .sort_values(ascending=False).head(TOP_N).index.tolist())

    zone_data = {}
    for z in top_zones:
        zdf = df[df.zone_id == z].set_index("time_bucket").reindex(full_idx)
        zdf["pickup_count"] = zdf["pickup_count"].fillna(0.0)
        # weather is citywide (confirmed identical across zones at a given hour);
        # forward/back-fill the rare reindex gap rather than zeroing it out
        zdf["avg_temperature"] = zdf["avg_temperature"].ffill().bfill()
        zdf["precipitation_mm"] = zdf["precipitation_mm"].fillna(0.0)
        zone_data[z] = zdf
    return zone_data


def load_anomaly_hours(conn):
    """Optional: hours flagged anomalous in temporal_anomalies, for the M1/M2 breakdown.
    Column name is a best guess (schema wasn't in the audit) — fails gracefully if wrong."""
    try:
        df = pd.read_sql("SELECT DISTINCT detected_at FROM temporal_anomalies", conn)
        return set(pd.to_datetime(df["detected_at"]).dt.floor("h"))
    except Exception as e:
        print(f"[anomaly join skipped — check column name] {e}")
        return set()


def mape(actual, pred):
    """MAPE over non-zero-actual hours only (undefined otherwise)."""
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))


def fit_forecast(train_y, test_len, exog_train=None, exog_test=None):
    model = SARIMAX(train_y, order=ORDER, seasonal_order=SEASONAL_ORDER,
                     exog=exog_train, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.get_forecast(test_len, exog=exog_test).predicted_mean
    return np.clip(fc.values, 0, None)


def run_zone(zone_id, zdf, anomaly_hours):
    train, test = zdf.iloc[:-TEST_HOURS], zdf.iloc[-TEST_HOURS:]
    exog_cols = ["avg_temperature", "precipitation_mm"]

    y_train = train["pickup_count"]
    y_test = test["pickup_count"].values

    m1_pred = fit_forecast(y_train, TEST_HOURS)
    m2_pred = fit_forecast(y_train, TEST_HOURS,
                            exog_train=train[exog_cols], exog_test=test[exog_cols])

    rainy = test["precipitation_mm"].values > RAIN_THRESHOLD_MM
    is_anom = np.array([t in anomaly_hours for t in test.index])

    rows = pd.DataFrame({
        "timestamp": test.index, "zone_id": zone_id, "actual": y_test,
        "m1_pred": m1_pred, "m2_pred": m2_pred,
        "temperature_c": test["avg_temperature"].values,
        "precipitation_mm": test["precipitation_mm"].values,
        "is_rainy": rainy, "is_anomalous": is_anom,
    })

    def metrics(mask, pred_col):
        a, p = rows.loc[mask, "actual"].values, rows.loc[mask, pred_col].values
        if len(a) == 0:
            return dict(mae=np.nan, rmse=np.nan, mape=np.nan, n=0)
        return dict(mae=mae(a, p), rmse=rmse(a, p), mape=mape(a, p), n=len(a))

    summary = {"zone_id": zone_id}
    for label, mask in [("all", np.ones(len(rows), bool)),
                         ("rainy", rainy), ("non_rainy", ~rainy),
                         ("anomalous", is_anom), ("normal", ~is_anom)]:
        for model_name, col in [("m1", "m1_pred"), ("m2", "m2_pred")]:
            for k, v in metrics(mask, col).items():
                summary[f"{model_name}_{label}_{k}"] = v

    return rows, summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = connect()
    zone_data = load_data(conn)
    anomaly_hours = load_anomaly_hours(conn)
    conn.close()

    print(f"[zones] {len(zone_data)} zones, {len(anomaly_hours)} anomalous hours known")

    all_rows, summaries = [], []
    for z, zdf in zone_data.items():
        try:
            rows, summary = run_zone(z, zdf, anomaly_hours)
        except Exception as e:
            print(f"[zone {z}] FAILED: {e}")
            continue
        all_rows.append(rows)
        summaries.append(summary)
        print(f"[zone {z}] M1 MAE={summary['m1_all_mae']:.1f}  M2 MAE={summary['m2_all_mae']:.1f}  "
              f"(rainy hours in test: {summary.get('m1_rainy_n', 0)})")

    detail = pd.concat(all_rows, ignore_index=True)
    detail_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    detail.to_csv(detail_path, index=False)

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(OUTPUT_DIR, "per_zone_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    agg = {}
    for label in ["all", "rainy", "non_rainy", "anomalous", "normal"]:
        for model_name in ["m1", "m2"]:
            for metric in ["mae", "rmse", "mape"]:
                col = f"{model_name}_{label}_{metric}"
                agg[col] = summary_df[col].mean(skipna=True)
    agg_df = pd.DataFrame([agg])
    agg_path = os.path.join(OUTPUT_DIR, "aggregate_results.csv")
    agg_df.to_csv(agg_path, index=False)

    print("\n=== AGGREGATE (mean across zones) ===")
    print(f"M1 (no weather)  MAE={agg['m1_all_mae']:.2f}  RMSE={agg['m1_all_rmse']:.2f}  MAPE={agg['m1_all_mape']:.2f}%")
    print(f"M2 (weather)     MAE={agg['m2_all_mae']:.2f}  RMSE={agg['m2_all_rmse']:.2f}  MAPE={agg['m2_all_mape']:.2f}%")
    print(f"Rainy hours  -> M1 MAE={agg['m1_rainy_mae']:.2f}  M2 MAE={agg['m2_rainy_mae']:.2f}")
    print(f"Non-rainy    -> M1 MAE={agg['m1_non_rainy_mae']:.2f}  M2 MAE={agg['m2_non_rainy_mae']:.2f}")
    if not np.isnan(agg.get("m1_anomalous_mae", np.nan)):
        print(f"Anomalous    -> M1 MAE={agg['m1_anomalous_mae']:.2f}  M2 MAE={agg['m2_anomalous_mae']:.2f}")

    print(f"\nWrote:\n  {detail_path}\n  {summary_path}\n  {agg_path}")


if __name__ == "__main__":
    main()