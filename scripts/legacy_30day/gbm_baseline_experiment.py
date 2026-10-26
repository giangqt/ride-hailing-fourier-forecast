#!/usr/bin/env python3
"""
GBM comparison against the existing SARIMAX baseline (M1) and, separately,
a direct test of whether a nonlinear model can extract value from the same
weather data that failed to help linear SARIMAX-X.

Design notes (read before modifying):

- Pooled, not per-zone: unlike SARIMAX (one model per zone, ~552 training
  hours each), a single LightGBM model is trained across all 20 zones'
  pooled training rows (zone_id as a categorical feature). Tree models need
  more data than 552 rows to be reliable; pooling gives ~7,000+ rows.

- Recursive walk-forward forecasting is REQUIRED for the 168-hour test
  horizon, not optional. lag_168 always references real training data for
  the entire test window (168 hours back from any test hour falls within
  the 552-hour training period), but lag_1/lag_2/lag_24 increasingly
  reference the test window itself as the horizon progresses -- those must
  use this loop's own prior predictions, never real future values, or the
  comparison is invalid (silent leakage -> falsely good GBM numbers).

- Two model variants trained: GBM-baseline (lags + calendar + zone only,
  directly comparable to SARIMAX M1) and GBM-weather (same features +
  contemporaneous citywide temperature/precipitation, directly comparable
  to SARIMAX-X M2 -- but testing whether a nonlinear model can extract
  value from the same weather data that a linear model could not).

- Hyperparameters are intentionally left near-default (no extensive tuning)
  to keep this a fair baseline-vs-baseline comparison, not a tuned-model-
  vs-untuned-model comparison.
"""

import os
import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("LightGBM not installed. Run: pip install lightgbm --break-system-packages")

TOP_N = 20
TEST_HOURS = 168
LAGS = [1, 2, 24, 168]

OUTPUT_DIR = os.path.expanduser("~/ride-hailing/experiments/weather_sarimax")
SARIMAX_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "per_zone_summary_final.csv")
PRED_PATH = os.path.join(OUTPUT_DIR, "predictions_gbm.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "per_zone_summary_gbm.csv")


def connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "rides"),
        user=os.environ.get("PGUSER", "rides"),
        password=os.environ.get("PGPASSWORD", "rides"),
    )


def load_data(conn):
    df = pd.read_sql("""
        SELECT time_bucket, zone_id, pickup_count, avg_temperature, precipitation_mm
        FROM hourly_demand ORDER BY zone_id, time_bucket""", conn)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"], utc=True)

    top_zones = (df.groupby("zone_id")["pickup_count"].sum()
                 .sort_values(ascending=False).head(TOP_N).index.tolist())
    df = df[df.zone_id.isin(top_zones)].copy()

    full_idx = pd.date_range(df["time_bucket"].min(), df["time_bucket"].max(), freq="h")
    frames = []
    for z in top_zones:
        zdf = df[df.zone_id == z].set_index("time_bucket").reindex(full_idx)
        zdf["zone_id"] = z
        zdf["pickup_count"] = zdf["pickup_count"].fillna(0.0)
        zdf["avg_temperature"] = zdf["avg_temperature"].ffill().bfill()
        zdf["precipitation_mm"] = zdf["precipitation_mm"].fillna(0.0)
        zdf.index.name = "time_bucket"
        frames.append(zdf.reset_index())
    return pd.concat(frames, ignore_index=True), top_zones


def add_calendar_features(df):
    df["hour"] = df["time_bucket"].dt.hour
    df["dow"] = df["time_bucket"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def add_lag_features(df):
    df = df.sort_values(["zone_id", "time_bucket"]).reset_index(drop=True)
    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby("zone_id")["pickup_count"].shift(lag)
    return df


def load_anomaly_hours(conn):
    try:
        a = pd.read_sql("SELECT DISTINCT detected_at FROM temporal_anomalies", conn)
        return set(pd.to_datetime(a["detected_at"], utc=True).dt.floor("h"))
    except Exception as e:
        print(f"[anomaly join skipped] {e}")
        return set()


def train_model(train_df, feature_cols):
    ds = lgb.Dataset(train_df[feature_cols], label=train_df["pickup_count"],
                      categorical_feature=["zone_id"])
    params = dict(objective="regression", metric="mae", num_leaves=31,
                  learning_rate=0.05, min_data_in_leaf=20, verbose=-1)
    model = lgb.train(params, ds, num_boost_round=300)
    return model


def recursive_forecast(model, zone_id, history, weather_test, feature_cols, use_weather):
    """history: list of real training pickup_counts for this zone, in order.
    Extended in-place with predictions; returns the TEST_HOURS predictions."""
    n_train = len(history)
    preds = []
    for h in range(TEST_HOURS):
        idx = n_train + h  # position in the conceptual train+test series
        row = {
            "zone_id": zone_id,
            "hour": weather_test.iloc[h]["hour"],
            "dow": weather_test.iloc[h]["dow"],
            "is_weekend": weather_test.iloc[h]["is_weekend"],
        }
        for lag in LAGS:
            row[f"lag_{lag}"] = history[idx - lag]  # always valid: history grows each step
        if use_weather:
            row["avg_temperature"] = weather_test.iloc[h]["avg_temperature"]
            row["precipitation_mm"] = weather_test.iloc[h]["precipitation_mm"]

        x = pd.DataFrame([row])[feature_cols]
        pred = max(0.0, float(model.predict(x)[0]))
        preds.append(pred)
        history.append(pred)  # feed forward for subsequent lag features
    return preds


def mae(a, p):
    return float(np.mean(np.abs(a - p)))


def rmse(a, p):
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(a, p):
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.sum() else np.nan


def main():
    conn = connect()
    df, zones = load_data(conn)
    anomaly_hours = load_anomaly_hours(conn)
    conn.close()

    df = add_calendar_features(df)
    df = add_lag_features(df)

    base_features = ["zone_id", "hour", "dow", "is_weekend"] + [f"lag_{l}" for l in LAGS]
    weather_features = base_features + ["avg_temperature", "precipitation_mm"]

    # per-zone chronological split, then pool the training rows across zones
    train_frames, test_frames = [], []
    for z in zones:
        zdf = df[df.zone_id == z].reset_index(drop=True)
        train_frames.append(zdf.iloc[:-TEST_HOURS])
        test_frames.append(zdf.iloc[-TEST_HOURS:])
    train_df = pd.concat(train_frames, ignore_index=True).dropna(subset=[f"lag_{l}" for l in LAGS])
    print(f"[data] pooled training rows after dropping lag warm-up: {len(train_df)}")

    print("[fit] GBM-baseline (no weather)...")
    model_base = train_model(train_df, base_features)
    print("[fit] GBM-weather...")
    model_weather = train_model(train_df, weather_features)

    all_rows, summaries = [], []
    for z, test_zdf in zip(zones, test_frames):
        train_hist = df[(df.zone_id == z)].iloc[:-TEST_HOURS]["pickup_count"].tolist()

        hist_base = list(train_hist)
        base_preds = recursive_forecast(model_base, z, hist_base, test_zdf.reset_index(drop=True),
                                         base_features, use_weather=False)
        hist_weather = list(train_hist)
        weather_preds = recursive_forecast(model_weather, z, hist_weather, test_zdf.reset_index(drop=True),
                                            weather_features, use_weather=True)

        actual = test_zdf["pickup_count"].values
        rainy = test_zdf["precipitation_mm"].values > 0
        is_anom = np.array([t in anomaly_hours for t in test_zdf["time_bucket"]])

        rows = pd.DataFrame({
            "timestamp": test_zdf["time_bucket"].values, "zone_id": z, "actual": actual,
            "gbm_base_pred": base_preds, "gbm_weather_pred": weather_preds,
            "is_rainy": rainy, "is_anomalous": is_anom,
        })
        all_rows.append(rows)

        def m(mask, col, fn):
            a, p = rows.loc[mask, "actual"].values, rows.loc[mask, col].values
            return fn(a, p) if len(a) else np.nan

        all_mask = np.ones(len(rows), bool)
        summary = {
            "zone_id": z,
            "gbm_base_mae": m(all_mask, "gbm_base_pred", mae),
            "gbm_base_rmse": m(all_mask, "gbm_base_pred", rmse),
            "gbm_base_mape": m(all_mask, "gbm_base_pred", mape),
            "gbm_weather_mae": m(all_mask, "gbm_weather_pred", mae),
            "gbm_weather_rmse": m(all_mask, "gbm_weather_pred", rmse),
            "gbm_weather_mape": m(all_mask, "gbm_weather_pred", mape),
            "gbm_base_rainy_mae": m(rainy, "gbm_base_pred", mae),
            "gbm_weather_rainy_mae": m(rainy, "gbm_weather_pred", mae),
            "gbm_base_anomalous_mae": m(is_anom, "gbm_base_pred", mae),
            "gbm_weather_anomalous_mae": m(is_anom, "gbm_weather_pred", mae),
        }
        summaries.append(summary)
        print(f"[zone {z}] GBM-base MAE={summary['gbm_base_mae']:.1f}  "
              f"GBM-weather MAE={summary['gbm_weather_mae']:.1f}")

    pd.concat(all_rows, ignore_index=True).to_csv(PRED_PATH, index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n=== GBM AGGREGATE (mean across 20 zones) ===")
    for col in ["gbm_base_mae", "gbm_base_rmse", "gbm_base_mape",
                "gbm_weather_mae", "gbm_weather_rmse", "gbm_weather_mape"]:
        print(f"{col}: {summary_df[col].mean():.2f}")
    print(f"Rainy hours -> GBM-base={summary_df['gbm_base_rainy_mae'].mean():.2f}  "
          f"GBM-weather={summary_df['gbm_weather_rainy_mae'].mean():.2f}")

    # Compare directly against the already-saved SARIMAX M1 per-zone results
    if os.path.exists(SARIMAX_SUMMARY_PATH):
        sarimax_df = pd.read_csv(SARIMAX_SUMMARY_PATH)[["zone_id", "m1_all_mae", "m2_all_mae"]]
        merged = summary_df.merge(sarimax_df, on="zone_id")

        print("\n=== GBM-base vs SARIMAX M1 (paired, 20 zones) ===")
        diff = merged["gbm_base_mae"].values - merged["m1_all_mae"].values
        w_stat, w_p = stats.wilcoxon(diff)
        print(f"Mean SARIMAX M1 MAE: {merged['m1_all_mae'].mean():.2f}   "
              f"Mean GBM-base MAE: {merged['gbm_base_mae'].mean():.2f}")
        print(f"Zones where GBM-base beats SARIMAX M1: {(diff < 0).sum()}/20")
        print(f"Wilcoxon p={w_p:.4f}")

        print("\n=== GBM-weather vs GBM-base (paired, 20 zones) ===")
        diff2 = merged["gbm_weather_mae"].values - merged["gbm_base_mae"].values
        w_stat2, w_p2 = stats.wilcoxon(diff2)
        print(f"Zones where weather helps GBM: {(diff2 < 0).sum()}/20")
        print(f"Wilcoxon p={w_p2:.4f}")

        merged.to_csv(os.path.join(OUTPUT_DIR, "gbm_vs_sarimax_comparison.csv"), index=False)
    else:
        print(f"\n[note] {SARIMAX_SUMMARY_PATH} not found -- skipping direct SARIMAX comparison. "
              f"GBM results are still saved above.")

    print(f"\nWrote:\n  {PRED_PATH}\n  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()