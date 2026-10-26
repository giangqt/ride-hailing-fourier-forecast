#!/usr/bin/env python3
"""
Paired uncertainty for model comparisons that respects shared temporal and spatial structure (reviewer 1).

Why not the per-zone Wilcoxon test alone: the 20 zones share the same weeks, weather and city-wide shocks, so zone-level
differences are not independent draws.

Method (applied to saved hourly absolute errors; identical origins, zones, horizon for every model):
  1. loss differential d[o,h,z] = |e_A| - |e_B|  (negative = A better)
  2. average over zones within each hour -> D[o,h]; this removes the spatial dependence from the inference unit
  3. the 8 weekly test windows are consecutive, so D is one continuous 1344-hour series
  4. circular moving-block bootstrap of the mean of D (blocks of 24 h and of 168 h, 10,000 resamples), percentile 95% CI
  5. Diebold-Mariano-style test: regress D on a constant with Newey-West (HAC, 168 lags) standard errors
  6. transparent counts: weeks (of 8) and zones (of 20) in which A has the lower MAE
  For weather comparisons the mean differential is also reported on rainy and dry hours separately
  (ratio estimator inside the same block bootstrap; rainy = precipitation > 0 at that hour).
Usage: python uncertainty_block_bootstrap.py
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.expanduser("~/ride-hailing/experiments/weather_sarimax")
B_RESAMPLES, BLOCKS, HAC_LAGS, SEED = 10000, (24, 168), 168, 0
PAIRS = [  # (A, B, label)  -> A minus B
    ("sarimax_fourier3", "sarimax_daily", "Fourier K=3 vs daily SARIMAX"),
    ("sarimax_fourier3_weather", "sarimax_fourier3", "weather added to Fourier SARIMAX"),
    ("gbm_weather", "gbm_full", "weather added to LightGBM"),
    ("gbm_full", "naive168", "LightGBM vs naive-168"),
    ("sarimax_fourier3", "naive168", "Fourier SARIMAX vs naive-168"),
    ("sarimax_fourier3", "gbm_full", "Fourier SARIMAX vs LightGBM"),
    ("sarimax_fourier3_weather", "gbm_full", "Fourier+weather SARIMAX vs LightGBM"),
    ("gbm_no_lag168", "gbm_full", "ablation: LightGBM without lag-168"),
    ("gbm_no_calendar", "gbm_full", "ablation: LightGBM without calendar"),
    ("gbm_lags_only", "sarimax_daily", "LightGBM with no weekly info vs daily SARIMAX"),
]


def load_errors():
    errs, zones = {}, None
    for fn in ["rolling_cheap_abs_err_top20.npz", "rolling_sarimax_abs_err.npz"]:
        d = np.load(os.path.join(BASE, fn))
        z = d["zones"].astype(int).tolist()
        zones = zones or z
        order = [z.index(k) for k in zones]
        for k in d.files:
            if k not in ("zones", "origins"):
                errs[k] = d[k][:, :, order].astype(float)
    return errs, zones


def rain_mask(n_test_hours, first=3048):
    try:
        import psycopg2
        conn = psycopg2.connect(host=os.environ.get("PGHOST", "localhost"), port=os.environ.get("PGPORT", "5432"),
                                dbname=os.environ.get("PGDATABASE", "rides"), user=os.environ.get("PGUSER", "rides"),
                                password=os.environ.get("PGPASSWORD", "rides"))
        w = pd.read_sql("SELECT time_bucket, max(precipitation_mm) p FROM hourly_demand_6mo_all GROUP BY time_bucket "
                        "ORDER BY time_bucket", conn)
        conn.close()
        return (w["p"].values[first:first + n_test_hours] > 0)
    except Exception as e:
        print(f"[rain mask unavailable: {e!r}]")
        return None


def block_indices(n, L, B, rng):
    nb = -(-n // L)
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
    return idx.reshape(B, -1)[:, :n].astype(np.int32)


def boot(D, L, rng, mask=None):
    n = len(D)
    idx = block_indices(n, L, B_RESAMPLES, rng)
    if mask is None:
        stat = D[idx].mean(1)
        obs = D.mean()
    else:
        num, den = (D * mask)[idx].sum(1), mask[idx].sum(1)
        stat = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        obs = (D * mask).sum() / mask.sum()
    lo, hi = np.nanpercentile(stat, [2.5, 97.5])
    p = float(np.nanmean(np.abs(stat - np.nanmean(stat)) >= abs(obs)))
    return obs, lo, hi, p


def analyse(errs, zones, rain=None):
    rng = np.random.default_rng(SEED)
    rows = []
    for a, b, label in PAIRS:
        if a not in errs or b not in errs:
            continue
        d = errs[a] - errs[b]                      # (origin, hour, zone)
        D = d.mean(2).reshape(-1)                  # average over zones, then concatenate the weeks
        base = errs[b].mean()
        r = dict(comparison=label, A=a, B=b, MAE_A=errs[a].mean(), MAE_B=base,
                 diff=D.mean(), diff_pct=100 * D.mean() / base,
                 weeks_A_better=int((d.mean((1, 2)) < 0).sum()), zones_A_better=int((d.mean((0, 1)) < 0).sum()))
        for L in BLOCKS:
            obs, lo, hi, p = boot(D, L, rng)
            r.update({f"ci_lo_L{L}": lo, f"ci_hi_L{L}": hi, f"p_boot_L{L}": p})
        ols = sm.OLS(D, np.ones(len(D))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        r.update(DM_t=float(ols.tvalues[0]), DM_p=float(ols.pvalues[0]))
        if rain is not None and "weather" in a:
            for nm, m in (("rainy", rain.astype(float)), ("dry", (~rain).astype(float))):
                obs, lo, hi, p = boot(D, 168, rng, mask=m)
                r.update({f"diff_{nm}": obs, f"ci_lo_{nm}": lo, f"ci_hi_{nm}": hi})
            r["rainy_hours"] = int(rain.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    errs, zones = load_errors()
    n_hours = next(iter(errs.values())).shape[0] * next(iter(errs.values())).shape[1]
    res = analyse(errs, zones, rain_mask(n_hours))
    res.to_csv(os.path.join(BASE, "uncertainty_pairs.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 48)
    c = ["comparison", "diff", "diff_pct", "ci_lo_L24", "ci_hi_L24", "ci_lo_L168", "ci_hi_L168", "DM_p",
         "weeks_A_better", "zones_A_better"]
    print("diff = MAE(A) - MAE(B) in trips/hour; negative means A is better. CI = block-bootstrap 95%.")
    print(res[c].round(3).to_string(index=False))
    if "diff_rainy" in res:
        print("\nWeather comparisons split by hour type (block 168 h):")
        print(res[res.diff_rainy.notna()][["comparison", "rainy_hours", "diff_rainy", "ci_lo_rainy", "ci_hi_rainy",
                                          "diff_dry", "ci_lo_dry", "ci_hi_dry"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()