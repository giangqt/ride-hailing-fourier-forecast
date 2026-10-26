# Fourier-enhanced SARIMAX for ride-hailing demand forecasting: code and results

Code and summary results for the paper "Fourier-Enhanced SARIMAX for Ride-Hailing Demand Forecasting"
(Truong Giang Nguyen, Thi Phuong Nghiem, Giang Son Tran), TSE 2026.

The paper compares a daily-seasonal SARIMAX(2,1,2)(1,1,1,24), the same model with weekly Fourier terms
(period 168 h, K=3), naive baselines and a pooled LightGBM model on hourly pickup counts per taxi zone,
using eight weekly rolling forecast origins (6 Aug - 30 Sep 2024).

The Kafka / Spark / TimescaleDB / Grafana pipeline that produced the hourly tables is part of a separate
repository: https://github.com/giangqt/Ride-Hailing

## Data

NYC TLC High-Volume For-Hire Vehicle trip records, April-September 2024 (public):
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Raw files are not included. Hourly pickup counts per taxi zone (zone ids 264 and 265 are excluded from
zone selection) are stored in the PostgreSQL table hourly_demand_6mo_all, built by
scripts/build_6month_all_zones.py. Weather is the Open-Meteo archive (Central Park, hourly temperature and
precipitation), already joined in that table.

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Tested with Python 3.12.3 on Ubuntu 24.04 (4 vCPU, about 13 GB RAM). Scripts read PostgreSQL settings from
PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD (defaults: localhost, 5432, rides, rides, rides). Output
files are written to ~/ride-hailing/experiments/weather_sarimax/ (constant at the top of each script).
Run the scripts from the scripts/ directory, because some import each other.

## Order of execution and link to the paper

| Step | Script | Produces |
|---|---|---|
| 1 | build_6month_all_zones.py | hourly table for all zones |
| 2 | naive_baselines_recursive.py | recursive naive-24 and naive-168 baselines |
| 3 | rolling_origin_cheap.py | naive baselines, LightGBM and ablations over 8 origins (Table II) |
| 4 | fourier_k_validation.py, order_validation.py | K and model-order selection on validation weeks |
| 5 | rolling_sarimax_fixedparam.py | daily, Fourier and Fourier+weather SARIMAX over 8 origins (Table II) |
| 6 | uncertainty_block_bootstrap.py | paired differences, block bootstrap, HAC test (Table III) |
| 7 | profile_runtime_memory.py, profile_followup.py | runtime and memory of s=24, Fourier, native s=168 (Table IV) |
| 8 | rolling_sarimax_other_zones.py | zones outside the top 20 (Table V) |
| 9 | seasonality_spectrum_check.py | fortnightly / monthly periodogram screening |

The SARIMAX runs take about 7 CPU hours (rolling protocol, 60 fits) and are resumable: finished fits are
cached and skipped on re-run.

## Protocol notes

- Zones, K and the model order are chosen from training or validation data only (before 6 Aug 2024).
- SARIMAX parameters are fitted once on the first 3,048 hours; later origins reuse them through the Kalman
  filter (no re-estimation). LightGBM is refitted at every origin.
- Naive-24 is recursive (the last observed day repeated).

## Results

results/ holds the summary CSV files and saved absolute errors behind the paper's tables
(for example rolling_sarimax_cases.csv, fourier_k_validation.csv, uncertainty_pairs.csv).

## legacy_30day

Scripts for the original 30-day experiments and the single-origin six-month run, kept for reference.

## License

MIT, see LICENSE.
