#!/usr/bin/env python3
"""
Build an isolated 6-month hourly demand + weather dataset (hourly_demand_6mo)
for the 20 already-modeled zones, spanning the FULL original TLC collection
window (April-September 2024) rather than the 30-day replay subset used
throughout the rest of this work.

Does NOT touch trip_events, hourly_demand, weather_data, or any other table
used by the live Grafana dashboard / thesis defense demo -- this is a fully
separate, isolated table. Purpose: test whether a longer training window
(i) supports a weekly (s=168) SARIMAX seasonal term the 552-hour window
could not (only ~3.3 weekly cycles there vs ~26 here), and (ii) changes the
weather-exogenous conclusion under far more data.

Aggregates directly from the raw Parquet files in memory rather than loading
118M trip-level records into Postgres -- unnecessary disk/time cost when
only hourly per-zone counts are needed for this analysis.

Real 2024 calendar dates are used throughout -- no anchor-date shifting,
since this table is not displayed on any live dashboard.
"""

import os

import pandas as pd
import psycopg2
import pyarrow.parquet as pq
import requests
from psycopg2.extras import execute_values

PROJECT_ROOT = os.path.expanduser("~/ride-hailing")
RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw_trips")
MONTHS = ["2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09"]
ZONE_IDS = [138, 132, 79, 61, 230, 161, 231, 68, 246, 76, 37, 234, 255, 164, 148, 181, 249, 7, 48, 112]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CENTRAL_PARK_LAT, CENTRAL_PARK_LON = 40.7794, -73.9692


def connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "rides"),
        user=os.environ.get("PGUSER", "rides"),
        password=os.environ.get("PGPASSWORD", "rides"),
    )


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hourly_demand_6mo (
                time_bucket       TIMESTAMPTZ NOT NULL,
                zone_id           INTEGER NOT NULL REFERENCES taxi_zones(zone_id),
                pickup_count      INTEGER NOT NULL DEFAULT 0,
                avg_temperature   DOUBLE PRECISION,
                precipitation_mm  DOUBLE PRECISION,
                PRIMARY KEY (time_bucket, zone_id)
            )
        """)
    conn.commit()


def aggregate_month(path):
    """Aggregate one monthly parquet file to hourly pickup counts for the 20 modeled zones."""
    pf = pq.ParquetFile(path)
    partials = []
    for rb in pf.iter_batches(batch_size=1_000_000, columns=["pickup_datetime", "PULocationID"]):
        df = rb.to_pandas()
        df = df[df["PULocationID"].isin(ZONE_IDS)]
        if df.empty:
            continue
        df["hour"] = pd.to_datetime(df["pickup_datetime"]).dt.floor("h")
        partial = df.groupby(["hour", "PULocationID"]).size().reset_index(name="pickup_count")
        partials.append(partial)
    if not partials:
        return pd.DataFrame(columns=["hour", "PULocationID", "pickup_count"])
    combined = pd.concat(partials, ignore_index=True)
    return combined.groupby(["hour", "PULocationID"], as_index=False)["pickup_count"].sum()


def fetch_weather(start, end):
    r = requests.get(ARCHIVE_URL, params={
        "latitude": CENTRAL_PARK_LAT, "longitude": CENTRAL_PARK_LON,
        "start_date": start, "end_date": end,
        "hourly": "temperature_2m,precipitation",
        "timezone": "America/New_York",
    }, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({
        "hour": pd.to_datetime(h["time"]),
        "avg_temperature": h["temperature_2m"],
        "precipitation_mm": h["precipitation"],
    })


def main():
    conn = connect()
    create_table(conn)

    print("[aggregate] processing 6 monthly parquet files (~2.7GB total on disk, "
          "aggregating down to 20 zones only)...")
    all_months = []
    for month in MONTHS:
        path = os.path.join(RAW_DIR, f"fhvhv_tripdata_{month}.parquet")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        agg = aggregate_month(path)
        print(f"[{month}] {len(agg)} zone-hour rows aggregated")
        all_months.append(agg)

    demand = pd.concat(all_months, ignore_index=True)
    demand = demand.rename(columns={"PULocationID": "zone_id"})

    full_start, full_end = demand["hour"].min(), demand["hour"].max()
    print(f"[range] {full_start} .. {full_end}")

    print("[weather] fetching Central Park hourly weather for the full range...")
    weather = fetch_weather(full_start.strftime("%Y-%m-%d"), full_end.strftime("%Y-%m-%d"))

    full_idx = pd.date_range(full_start, full_end, freq="h")
    rows = []
    for z in ZONE_IDS:
        zdf = demand[demand.zone_id == z].set_index("hour").reindex(full_idx)
        zdf["pickup_count"] = zdf["pickup_count"].fillna(0).astype(int)
        w = weather.set_index("hour").reindex(full_idx)
        zdf["avg_temperature"] = w["avg_temperature"].ffill().bfill()
        zdf["precipitation_mm"] = w["precipitation_mm"].fillna(0.0)
        for ts, r in zdf.iterrows():
            rows.append((ts.strftime("%Y-%m-%d %H:%M:%S"), z, int(r["pickup_count"]),
                         float(r["avg_temperature"]), float(r["precipitation_mm"])))

    print(f"[write] inserting {len(rows)} rows into hourly_demand_6mo "
          f"({len(ZONE_IDS)} zones x {len(full_idx)} hours)...")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE hourly_demand_6mo")
        execute_values(cur,
            """INSERT INTO hourly_demand_6mo (time_bucket, zone_id, pickup_count, avg_temperature, precipitation_mm)
               VALUES %s""", rows, page_size=5000)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), MIN(time_bucket), MAX(time_bucket), SUM(pickup_count) FROM hourly_demand_6mo")
        n, mn, mx, total = cur.fetchone()
    print(f"\n[done] hourly_demand_6mo: {n} rows, {mn}..{mx}, total pickups={total:,}")
    conn.close()


if __name__ == "__main__":
    main()