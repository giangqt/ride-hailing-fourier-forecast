#!/usr/bin/env python3
"""
Build hourly_demand_6mo_all: hourly pickup counts for ALL taxi zones over the
full 2024-04-01 .. 2024-09-30 window (4392 hours), plus the same Central Park
weather already stored in hourly_demand_6mo.

Why: reviewers asked for (i) zone selection using training data only and
(ii) results beyond the top-20 zones. hourly_demand_6mo only holds 20 zones that
were picked by total volume INCLUDING the test period, so neither can be done
from it. This table is separate; no existing table is modified.

Resumable: each monthly aggregate is cached as CSV and reused on re-run.
Self-check at the end compares the 20 original zones against hourly_demand_6mo
(counts must match exactly).
"""
import os

import numpy as np
import pandas as pd
import psycopg2
import pyarrow.parquet as pq
from psycopg2.extras import execute_values

ROOT = os.path.expanduser("~/ride-hailing")
RAW_DIR = os.path.join(ROOT, "data/raw_trips")
CACHE_DIR = os.path.join(ROOT, "experiments/weather_sarimax/agg_cache")
MONTHS = ["2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09"]
START, END = pd.Timestamp("2024-04-01 00:00"), pd.Timestamp("2024-09-30 23:00")


def connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"), port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "rides"), user=os.environ.get("PGUSER", "rides"),
        password=os.environ.get("PGPASSWORD", "rides"))


def aggregate_month(path):
    partials = []
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=1_000_000, columns=["pickup_datetime", "PULocationID"]):
        df = rb.to_pandas()
        df["hour"] = pd.to_datetime(df["pickup_datetime"]).dt.floor("h")
        partials.append(df.groupby(["hour", "PULocationID"]).size().reset_index(name="pickup_count"))
    allp = pd.concat(partials, ignore_index=True)
    return allp.groupby(["hour", "PULocationID"], as_index=False)["pickup_count"].sum()


def month_counts(month):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{month}.csv")
    if os.path.exists(cache):
        print(f"[{month}] cached")
        return pd.read_csv(cache, parse_dates=["hour"])
    path = os.path.join(RAW_DIR, f"fhvhv_tripdata_{month}.parquet")
    agg = aggregate_month(path)
    agg.to_csv(cache, index=False)
    print(f"[{month}] aggregated {len(agg)} zone-hour rows")
    return agg


def main():
    conn = connect()
    zone_ids = pd.read_sql("SELECT zone_id FROM taxi_zones ORDER BY zone_id", conn)["zone_id"].tolist()
    demand = pd.concat([month_counts(m) for m in MONTHS], ignore_index=True)
    demand = demand.rename(columns={"PULocationID": "zone_id"})
    demand = demand[demand["zone_id"].isin(zone_ids) & (demand["hour"] >= START) & (demand["hour"] <= END)]
    demand = demand.groupby(["hour", "zone_id"], as_index=False)["pickup_count"].sum()

    full_idx = pd.date_range(START, END, freq="h")
    grid = pd.MultiIndex.from_product([full_idx, zone_ids], names=["hour", "zone_id"])
    demand = demand.set_index(["hour", "zone_id"]).reindex(grid, fill_value=0).reset_index()
    print(f"[grid] {len(zone_ids)} zones x {len(full_idx)} hours = {len(demand)} rows, "
          f"total trips {int(demand['pickup_count'].sum())}")

    demand["ts"] = demand["hour"].dt.strftime("%Y-%m-%d %H:%M:%S")
    rows = list(zip(demand["ts"], demand["zone_id"].astype(int), demand["pickup_count"].astype(int)))

    with conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS hourly_demand_6mo_all (
            time_bucket TIMESTAMPTZ NOT NULL, zone_id INTEGER NOT NULL REFERENCES taxi_zones(zone_id),
            pickup_count INTEGER NOT NULL DEFAULT 0, avg_temperature DOUBLE PRECISION,
            precipitation_mm DOUBLE PRECISION, PRIMARY KEY (time_bucket, zone_id))""")
        cur.execute("TRUNCATE hourly_demand_6mo_all")
        execute_values(cur, "INSERT INTO hourly_demand_6mo_all (time_bucket, zone_id, pickup_count) VALUES %s",
                       rows, page_size=20000)
        cur.execute("""UPDATE hourly_demand_6mo_all a SET avg_temperature = w.t, precipitation_mm = w.p
                       FROM (SELECT time_bucket, max(avg_temperature) t, max(precipitation_mm) p
                             FROM hourly_demand_6mo GROUP BY time_bucket) w
                       WHERE a.time_bucket = w.time_bucket""")
        cur.execute("""SELECT count(*), count(avg_temperature), count(DISTINCT zone_id), min(time_bucket), max(time_bucket)
                       FROM hourly_demand_6mo_all""")
        print("[table] rows, rows_with_weather, zones, min, max:", cur.fetchone())
        cur.execute("""SELECT count(*), coalesce(sum(abs(a.pickup_count - b.pickup_count)), 0)
                       FROM hourly_demand_6mo_all a JOIN hourly_demand_6mo b USING (time_bucket, zone_id)""")
        n, diff = cur.fetchone()
        print(f"[check] overlap with hourly_demand_6mo: {n} rows, total abs count difference = {diff} (must be 0)")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()