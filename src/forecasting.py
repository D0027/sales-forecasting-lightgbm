"""
Core library for the Store-Item Demand Forecasting project.

Everything here mirrors the notebook (notebooks/sales_forecasting.ipynb):

  * 90-day direct forecasting with ONE global LightGBM model (all store-item series)
  * every feature is built from data that is at least 91 days old, so the model needs
    no recursive forecasting and cannot leak the future
  * time-based split: last 90 days = test, the 90 days before = validation

Data format (Kaggle "Store Item Demand Forecasting Challenge" train.csv):
    date, store, item, sales
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Config (same as the notebook)
# --------------------------------------------------------------------------- #
SEED = 42
HORIZON = 90
MIN_LAG = HORIZON + 1
LAGS = [91, 98, 105, 112, 119, 126, 133, 140, 182, 350, 357, 364, 371, 378, 728]  # multiples of 7
ROLL_WINDOWS = [7, 14, 28, 56, 91, 182, 364]
LGB_PARAMS = dict(
    objective="regression", metric="l2", learning_rate=0.08, num_leaves=63,
    min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbose=-1, seed=SEED,
)
MAX_ROUNDS = 2000
EARLY_STOP = 50
T_TRAIN0 = 364  # first column whose lag_364 exists


# --------------------------------------------------------------------------- #
# Data / panel
# --------------------------------------------------------------------------- #
def load_train_csv(path: str) -> pd.DataFrame:
    """Read train.csv and return a clean (date, store, item, sales) frame."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    missing = {"date", "store", "item", "sales"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df = df[["date", "store", "item", "sales"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


@dataclass
class Panel:
    """Sales as a (series x days) matrix, with HORIZON empty (NaN) future days appended."""
    S_all: np.ndarray
    sid: pd.DataFrame
    hist_dates: pd.DatetimeIndex
    all_dates: pd.DatetimeIndex

    @property
    def n_series(self) -> int:
        return self.S_all.shape[0]

    @property
    def n_hist(self) -> int:
        return len(self.hist_dates)

    @property
    def S_hist(self) -> np.ndarray:
        return self.S_all[:, : self.n_hist]


def build_panel(df: pd.DataFrame, horizon: int = HORIZON) -> Panel:
    hist_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    wide = df.set_index(["store", "item", "date"])["sales"].unstack("date").reindex(columns=hist_dates)
    wide = wide.fillna(0)  # a missing store-item-day means no sales
    sid = wide.index.to_frame(index=False)
    future = pd.date_range(hist_dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    S_all = np.concatenate(
        [wide.values.astype("float64"), np.full((len(sid), horizon), np.nan)], axis=1
    )
    return Panel(S_all=S_all, sid=sid, hist_dates=hist_dates, all_dates=hist_dates.append(future))


@dataclass
class Splits:
    t_train0: int
    t_val: int
    t_test: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    future_idx: np.ndarray


def make_splits(panel: Panel, horizon: int = HORIZON) -> Splits:
    n_hist = panel.n_hist
    t_test = n_hist - horizon
    t_val = t_test - horizon
    if t_val - T_TRAIN0 < 365:
        raise ValueError("Not enough history: need roughly 2.5+ years of daily data.")
    return Splits(
        t_train0=T_TRAIN0, t_val=t_val, t_test=t_test,
        train_idx=np.arange(T_TRAIN0, t_val),
        val_idx=np.arange(t_val, t_test),
        test_idx=np.arange(t_test, n_hist),
        future_idx=np.arange(n_hist, n_hist + horizon),
    )


# --------------------------------------------------------------------------- #
# Features (all >= 91 days old => leak-free for a 90-day horizon)
# --------------------------------------------------------------------------- #
def build_features(S: np.ndarray, dates) -> dict:
    """S: (series, days) with NaN for unknown days. Returns name -> (series, days) float32."""
    n, T = S.shape
    D = pd.DataFrame(S.T, index=dates)
    F = {}

    for L in LAGS:
        F[f"lag_{L}"] = D.shift(L).values.T

    for w in ROLL_WINDOWS:
        F[f"rm_{w}"] = D.rolling(w, min_periods=w).mean().shift(MIN_LAG).values.T
    F["rstd_28"] = D.rolling(28, min_periods=28).std().shift(MIN_LAG).values.T

    lag = lambda L: F[f"lag_{L}"]
    F["sdow_4"] = np.mean([lag(L) for L in (91, 98, 105, 112)], axis=0)
    F["sdow_8"] = np.mean([lag(L) for L in (91, 98, 105, 112, 119, 126, 133, 140)], axis=0)
    F["ya_5"] = np.mean([lag(L) for L in (350, 357, 364, 371, 378)], axis=0)

    rm91_prev = D.rolling(91, min_periods=91).mean().shift(MIN_LAG + 364).values.T
    with np.errstate(divide="ignore", invalid="ignore"):
        F["growth"] = np.where(rm91_prev > 0, F["rm_91"] / rm91_prev, np.nan)

    di = pd.DatetimeIndex(dates)
    cal = {
        "dow": di.dayofweek, "month": di.month, "day": di.day, "doy": di.dayofyear,
        "week": di.isocalendar().week.values.astype(int), "quarter": di.quarter,
        "is_month_start": di.is_month_start.astype(int), "is_month_end": di.is_month_end.astype(int),
        "year": di.year,
    }
    for k, v in cal.items():
        F[k] = np.broadcast_to(np.asarray(v)[None, :], (n, T))

    return {k: np.asarray(v, dtype="float32") for k, v in F.items()}


def feature_names(feats: dict) -> list:
    return list(feats.keys()) + ["store", "item"]


def check_no_leak(panel: Panel, feats: dict, first_hidden_col: int, eval_cols: np.ndarray) -> None:
    """Hide actual sales from `first_hidden_col` on; features on `eval_cols` must not change."""
    S_masked = panel.S_all.copy()
    S_masked[:, first_hidden_col:] = np.nan
    masked = build_features(S_masked, panel.all_dates)
    bad = [k for k in feats
           if not np.allclose(feats[k][:, eval_cols], masked[k][:, eval_cols],
                              equal_nan=True, rtol=1e-5, atol=1e-5)]
    if bad:
        raise AssertionError(f"LEAKAGE in features: {bad}")


def make_rows(panel: Panel, feats: dict, col_idx: np.ndarray, store_cats, item_cats):
    """Long-format rows (series x selected days) -> (X DataFrame, y vector)."""
    k = len(col_idx)
    X = pd.DataFrame({name: M[:, col_idx].ravel() for name, M in feats.items()})
    X["store"] = pd.Categorical(np.repeat(panel.sid["store"].values, k), categories=store_cats)
    X["item"] = pd.Categorical(np.repeat(panel.sid["item"].values, k), categories=item_cats)
    y = panel.S_all[:, col_idx].ravel()
    return X[feature_names(feats)], y


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def smape(a, f) -> float:
    a, f = np.asarray(a, float).ravel(), np.asarray(f, float).ravel()
    denom = np.abs(a) + np.abs(f)
    out = np.where(denom == 0, 0.0, np.abs(f - a) / np.where(denom == 0, 1, denom))
    return float(200 * out.mean())


def wape(a, f) -> float:
    a, f = np.asarray(a, float).ravel(), np.asarray(f, float).ravel()
    return float(100 * np.abs(f - a).sum() / np.abs(a).sum())


def mae(a, f) -> float:
    return float(np.abs(np.asarray(a, float) - np.asarray(f, float)).mean())


def rmse(a, f) -> float:
    return float(np.sqrt(((np.asarray(a, float) - np.asarray(f, float)) ** 2).mean()))


def score_row(name: str, actual: np.ndarray, forecast: np.ndarray) -> dict:
    """actual / forecast: (series x horizon) matrices."""
    return {
        "Model": name,
        "SMAPE": smape(actual, forecast),
        "WAPE %": wape(actual, forecast),
        "MAE": mae(actual, forecast),
        "RMSE": rmse(actual, forecast),
        "Total-level WAPE %": wape(actual.sum(axis=0), forecast.sum(axis=0)),
    }


# --------------------------------------------------------------------------- #
# Baselines + ETS
# --------------------------------------------------------------------------- #
def baseline_forecasts(panel: Panel, feats: dict, t_test: int, horizon: int = HORIZON) -> dict:
    """Seasonal-naive family. Returns name -> (series x horizon) forecast for the test window."""
    S = panel.S_all
    cols = slice(t_test, t_test + horizon)
    f_naive = feats["lag_364"][:, cols].astype("float64")
    f_smooth = feats["ya_5"][:, cols].astype("float64")

    recent = S[:, t_test - horizon:t_test].sum(axis=1)
    prev = S[:, t_test - horizon - 364:t_test - 364].sum(axis=1)
    growth = np.clip(np.where(prev > 0, recent / prev, 1.0), 0.7, 1.5)
    return {
        "Seasonal Naive (t-364)": f_naive,
        "Smoothed Seasonal Naive": f_smooth,
        "Smoothed SNaive x Growth": f_smooth * growth[:, None],
    }


def ets_total_forecast(total_history: np.ndarray, horizon: int = HORIZON):
    """Holt-Winters (weekly seasonality) on total sales. Returns (name, forecast)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    try:
        fit = ExponentialSmoothing(total_history, trend="add", damped_trend=True, seasonal="mul",
                                   seasonal_periods=7, initialization_method="estimated").fit()
        name = "Holt-Winters ETS (mul, weekly)"
    except Exception:
        fit = ExponentialSmoothing(total_history, trend="add", damped_trend=True, seasonal="add",
                                   seasonal_periods=7, initialization_method="estimated").fit()
        name = "Holt-Winters ETS (add, weekly)"
    return name, np.asarray(fit.forecast(horizon))


# --------------------------------------------------------------------------- #
# LightGBM
# --------------------------------------------------------------------------- #
def _dataset(X, y, reference=None):
    import lightgbm as lgb
    return lgb.Dataset(X, label=np.log1p(y), categorical_feature=["store", "item"],
                       reference=reference, free_raw_data=False)


def fit_lgbm(X_tr, y_tr, X_va=None, y_va=None, rounds=None, params=None, log_every=100):
    """Train on log1p(sales). With validation data -> early stopping. Returns (booster, n_rounds)."""
    import lightgbm as lgb

    params = params or LGB_PARAMS
    dtrain = _dataset(X_tr, y_tr)
    if X_va is not None:
        dval = _dataset(X_va, y_va, reference=dtrain)
        booster = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain, dval], valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(log_every)],
        )
        return booster, int(booster.best_iteration)
    if rounds is None:
        raise ValueError("Pass `rounds` when no validation set is given.")
    return lgb.train(params, dtrain, num_boost_round=rounds), int(rounds)


def predict_sales(booster, X, num_iteration=None) -> np.ndarray:
    return np.expm1(booster.predict(X, num_iteration=num_iteration)).clip(0)


def forecast_future(panel: Panel, booster, feats: dict = None, store_cats=None, item_cats=None,
                    horizon: int = HORIZON) -> pd.DataFrame:
    """Forecast the `horizon` days after the last observed date. Returns long DataFrame."""
    feats = feats or build_features(panel.S_all, panel.all_dates)
    store_cats = store_cats if store_cats is not None else sorted(panel.sid["store"].unique())
    item_cats = item_cats if item_cats is not None else sorted(panel.sid["item"].unique())
    future_idx = np.arange(panel.n_hist, panel.n_hist + horizon)
    X, _ = make_rows(panel, feats, future_idx, store_cats, item_cats)
    F = predict_sales(booster, X).reshape(panel.n_series, horizon)
    return pd.DataFrame({
        "date": np.tile(panel.all_dates[future_idx].values, panel.n_series),
        "store": np.repeat(panel.sid["store"].values, horizon),
        "item": np.repeat(panel.sid["item"].values, horizon),
        "forecast": F.ravel().round(2),
    })


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #
def save_model(booster, path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    booster.save_model(path)
    meta_path = os.path.join(os.path.dirname(os.path.abspath(path)), "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_model(path: str):
    import lightgbm as lgb
    return lgb.Booster(model_file=path)
