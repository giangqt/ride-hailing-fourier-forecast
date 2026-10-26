#!/usr/bin/env python3
"""
Rolling-origin evaluation of the cheap models (naive-24h recursive, naive-168h, pooled LightGBM
and its ablations) on hourly_demand_6mo_all.

Protocol (identical for every model; the SARIMAX/Fourier runs later reuse the SAME origins):
  - 8 weekly forecast origins, expanding training window, 168-hour horizon, no overlap between test windows.
    Origins (hour index from 2024-04-01 00:00): 3048, 3216, ..., 4224 -> test weeks start 2024-08-06 ... 2024-09-24.
  - Zone sets are chosen from training-period volume ONLY (hours before the first origin, i.e. before 2024-08-06).
      top20     : ranks 1-20
      mid20     : 20 zones evenly spaced over ranks 21-120
      low20     : 20 zones evenly spaced over ranks 121-220
      orig20    : the 20 zones used in the paper (selected on full-window volume) - for continuity only
      tail20    : 20 zones evenly spaced over ranks 221-end (the sparsest zones)
    Taxi-zone ids 264/265 are excluded from selection.
  - LightGBM: pooled per zone set, refit at every origin, fixed hyperparameters (no tuning), recursive
    walk-forward (a lag that falls inside the test window uses the model's own earlier prediction).
  - Metrics per (origin, zone): MAE, RMSE, MAPE (zero-demand hours excluded from MAPE only; MAE/RMSE keep them).
    Aggregates are means over (origin, zone) cases.
Results are written per zone set and skipped on re-run if already present.
"""
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

T_HOURS, HORIZON, N_ORIGINS = 4392, 168, 8
ORIGINS = [T_HOURS - HORIZON * j for j in range(N_ORIGINS, 0, -1)]
EXCLUDE = {264, 265}
ORIG20 = [138, 132, 79, 61, 230, 161, 231, 68, 246, 76, 37, 234, 255, 164, 148, 181, 249, 7, 48, 112]
LAGS = [1, 2, 24, 168]
START = pd.Timestamp("2024-04-01 00:00")
OUT = os.path.expanduser("~/ride-hailing/experiments/weather_sarimax")
PARAMS = dict(objective="regression", metric="mae", num_leaves=31, learning_rate=0.05, min_data_in_leaf=20,
              verbose=-1, num_threads=4, seed=0, deterministic=True, force_row_wise=True)
ROUNDS = 300

CAL = ["hour", "dow", "is_weekend"]
LAGF = ["lag_1", "lag_2", "lag_24", "lag_168"]
VARIANTS = {
    "gbm_full": ["zone"] + CAL + LAGF,
    "gbm_no_lag168": ["zone"] + CAL + ["lag_1", "lag_2", "lag_24"],
    "gbm_no_calendar": ["zone"] + LAGF,
    "gbm_lags_only": ["zone", "lag_1", "lag_2", "lag_24"],
    "gbm_weather": ["zone"] + CAL + LAGF + ["temp", "precip"],
}
MODELS = ["naive24_rec", "naive168"] + list(VARIANTS)


def connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"), port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "rides"), user=os.environ.get("PGUSER", "rides"),
        password=os.environ.get("PGPASSWORD", "rides"))


def load_data():
    conn = connect()
    df = pd.read_sql("""SELECT time_bucket, zone_id, pickup_count, avg_temperature, precipitation_mm
                        FROM hourly_demand_6mo_all""", conn)
    conn.close()
    df["time_bucket"] = pd.to_datetime(df["time_bucket"], utc=True)
    Y = df.pivot(index="time_bucket", columns="zone_id", values="pickup_count").sort_index()
    w = df.groupby("time_bucket")[["avg_temperature", "precipitation_mm"]].mean().sort_index()
    assert len(Y) == T_HOURS, f"expected {T_HOURS} hours, got {len(Y)}"
    assert (np.diff(Y.index.values.astype("datetime64[h]").astype(int)) == 1).all(), "hourly grid has gaps"
    return Y, w


def select_zones(Y):
    vol = Y.iloc[:ORIGINS[0]].sum(axis=0)
    vol = vol[~vol.index.isin(EXCLUDE)].sort_values(ascending=False, kind="stable")
    ids = vol.index.tolist()
    pick = lambda lo, hi: [ids[i] for i in np.unique(np.linspace(lo - 1, hi - 1, 20).round().astype(int))]
    sets = {"top20": ids[:20], "mid20": pick(21, 120), "low20": pick(121, 220), "orig20": ORIG20,
            "tail20": pick(221, len(ids))}
    desc = []
    for name, zs in sets.items():
        for z in zs:
            col = Y[z].iloc[:ORIGINS[0]].values
            desc.append(dict(stratum=name, zone_id=z, train_rank=ids.index(z) + 1, train_volume=int(col.sum()),
                             train_mean_per_hour=col.mean(), train_zero_frac=(col == 0).mean()))
    return sets, pd.DataFrame(desc)


def build_features(Yz, W, ts):
    T, n = Yz.shape
    F = {"zone": np.tile(np.arange(n, dtype=float), (T, 1)),
         "hour": np.repeat(ts.hour.values[:, None], n, 1).astype(float),
         "dow": np.repeat(ts.dayofweek.values[:, None], n, 1).astype(float),
         "temp": np.repeat(W[:, 0:1], n, 1), "precip": np.repeat(W[:, 1:2], n, 1)}
    F["is_weekend"] = (F["dow"] >= 5).astype(float)
    for k in LAGS:
        L = np.full((T, n), np.nan)
        L[k:] = Yz[:-k]
        F[f"lag_{k}"] = L
    return F


def metrics(a, p):
    mae = np.abs(a - p).mean(0)
    rmse = np.sqrt(((a - p) ** 2).mean(0))
    nz = a != 0
    mape = np.where(nz.sum(0) > 0,
                    (np.where(nz, np.abs(a - p) / np.where(nz, a, 1), 0).sum(0) / np.maximum(nz.sum(0), 1)) * 100,
                    np.nan)
    return mae, rmse, mape, (~nz).sum(0)


def recursive_gbm(model, feats, F, Yz, o):
    T, n = Yz.shape
    Hh = Yz.copy()
    Hh[o:] = np.nan
    preds = np.zeros((HORIZON, n))
    for h in range(HORIZON):
        t = o + h
        cols = []
        for f in feats:
            if f.startswith("lag_"):
                cols.append(Hh[t - int(f.split("_")[1])])
            else:
                cols.append(F[f][t])
        X = np.stack(cols, axis=1)
        assert np.isfinite(X).all(), "non-finite feature -> possible read of unforecast future"
        p = np.clip(model.predict(X), 0, None)
        Hh[t] = p
        preds[h] = p
    return preds


def run_stratum(name, zones, Y, W, ts):
    Yz = Y[zones].values.astype(float)
    F = build_features(Yz, W, ts)
    n = len(zones)
    cases, errs = [], {m: np.zeros((N_ORIGINS, HORIZON, n), dtype=np.float32) for m in MODELS}
    for oi, o in enumerate(ORIGINS):
        A = Yz[o:o + HORIZON]
        P = {}
        n24 = np.zeros((HORIZON, n))
        for h in range(HORIZON):
            n24[h] = Yz[o - 24 + h] if h < 24 else n24[h - 24]
        P["naive24_rec"] = n24
        P["naive168"] = Yz[o - 168:o]
        y_tr = Yz[168:o].reshape(-1)
        for vname, feats in VARIANTS.items():
            X_tr = np.stack([F[f][168:o].reshape(-1) for f in feats], axis=1)
            model = lgb.train(PARAMS, lgb.Dataset(X_tr, label=y_tr, categorical_feature=[0]), num_boost_round=ROUNDS)
            P[vname] = recursive_gbm(model, feats, F, Yz, o)
        for m in MODELS:
            errs[m][oi] = np.abs(A - P[m])
            mae, rmse, mape, nzero = metrics(A, P[m])
            for zi, z in enumerate(zones):
                cases.append(dict(origin_idx=oi, origin_start=str(ts[o]), stratum=name, zone_id=z, model=m,
                                  mae=mae[zi], rmse=rmse[zi], mape=mape[zi], n_zero=int(nzero[zi]),
                                  mean_actual=A[:, zi].mean()))
        print(f"[{name}] origin {oi + 1}/{N_ORIGINS} ({ts[o]:%Y-%m-%d}) "
              f"naive168={np.mean(errs['naive168'][oi]):.1f} gbm_full={np.mean(errs['gbm_full'][oi]):.1f} "
              f"gbm_no_lag168={np.mean(errs['gbm_no_lag168'][oi]):.1f}", flush=True)
    return pd.DataFrame(cases), errs


def summarize(cases):
    out = []
    for s, g in cases.groupby("stratum"):
        base = g[g.model == "naive168"].set_index(["origin_idx", "zone_id"])["mae"]
        for m in MODELS:
            gm = g[g.model == m].set_index(["origin_idx", "zone_id"])
            ratio = (gm["mae"] / base).replace([np.inf, -np.inf], np.nan)
            ratio = ratio[(base > 0) & (gm["mae"] > 0)]
            out.append(dict(stratum=s, model=m, MAE=gm["mae"].mean(), RMSE=gm["rmse"].mean(),
                            MAPE=gm["mape"].mean(), rMAE_vs_naive168=float(np.exp(np.log(ratio).mean()))))
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    Y, w = load_data()
    ts = pd.date_range(START, periods=T_HOURS, freq="h")
    W = w[["avg_temperature", "precipitation_mm"]].values.astype(float)
    sets, desc = select_zones(Y)
    desc.to_csv(os.path.join(OUT, "rolling_cheap_zones.csv"), index=False)
    print(desc.groupby("stratum")[["train_mean_per_hour", "train_zero_frac"]].agg(["min", "mean", "max"]).round(3))
    print("top20 (train-only) vs paper's 20 zones, overlap:", len(set(sets["top20"]) & set(ORIG20)), "/ 20")
    print("zones in paper set but not in train-only top20:", sorted(set(ORIG20) - set(sets["top20"])))
    print("origins:", [(i, str(ts[o])) for i, o in enumerate(ORIGINS)])
    for name, zones in sets.items():
        path = os.path.join(OUT, f"rolling_cheap_{name}.csv")
        if os.path.exists(path):
            print(f"[{name}] already done, skipping")
            continue
        cases, errs = run_stratum(name, zones, Y, W, ts)
        np.savez_compressed(os.path.join(OUT, f"rolling_cheap_abs_err_{name}.npz"),
                            zones=np.array(zones), origins=np.array(ORIGINS), **errs)
        cases.to_csv(path, index=False)
    allc = pd.concat([pd.read_csv(os.path.join(OUT, f"rolling_cheap_{n}.csv")) for n in sets], ignore_index=True)
    summ = summarize(allc)
    summ.to_csv(os.path.join(OUT, "rolling_cheap_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    for s in sets:
        print(f"\n=== {s}: mean over {N_ORIGINS} origins x 20 zones ===")
        print(summ[summ.stratum == s].drop(columns="stratum").round(2).to_string(index=False))
    last = allc[(allc.origin_idx == N_ORIGINS - 1) & (allc.stratum == "orig20")]
    print("\n=== orig20, last origin only (test week 2024-09-24..30, the paper's Table III week) ===")
    print(last.groupby("model")[["mae", "rmse", "mape"]].mean().round(2))
    t = allc[allc.stratum == "top20"].pivot_table(index="origin_idx", columns="model", values="mae", aggfunc="mean")
    print("\n=== top20: MAE by origin ===")
    print(t[["naive168", "naive24_rec", "gbm_full", "gbm_no_lag168", "gbm_no_calendar", "gbm_lags_only",
             "gbm_weather"]].round(1).to_string())


if __name__ == "__main__":
    main()