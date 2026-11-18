"""Pure helper functions for the Streamlit dashboard (no Streamlit imports, easy to test)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_history_csv(file_like) -> pd.DataFrame:
    """Read an uploaded Kaggle train.csv (date, store, item, sales)."""
    df = pd.read_csv(file_like)
    df.columns = [c.lower() for c in df.columns]
    missing = {"date", "store", "item", "sales"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df[["date", "store", "item", "sales"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def filter_df(df: pd.DataFrame, store=None, item=None) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    if store is not None:
        mask &= (df["store"] == store).values
    if item is not None:
        mask &= (df["item"] == item).values
    return df[mask]


def daily_series(df: pd.DataFrame, value_col: str, store=None, item=None) -> pd.Series:
    sub = filter_df(df, store, item)
    return sub.groupby("date")[value_col].sum().sort_index()


def weekly_complete(s: pd.Series) -> pd.Series:
    """Weekly (Mon-Sun) totals, keeping only complete 7-day weeks."""
    g = s.resample("W-SUN")
    out = g.sum()
    return out[g.count() == 7]


def summary_kpis(s: pd.Series) -> dict:
    wk = s[s.index.dayofweek < 5].mean()
    we = s[s.index.dayofweek >= 5].mean()
    peak_date = s.idxmax()
    return {
        "total": float(s.sum()),
        "avg_day": float(s.mean()),
        "peak_date": peak_date,
        "peak_value": float(s.max()),
        "weekend_uplift": float(we / wk - 1) if wk > 0 else float("nan"),
    }


def same_period_last_year(hist: pd.Series, index: pd.DatetimeIndex, shift_days: int = 364) -> pd.Series:
    """History values for the same weekdays one year earlier (364 days back keeps the weekday)."""
    return hist.reindex(index - pd.Timedelta(days=shift_days)).set_axis(index)


def yoy_growth(forecast: pd.Series, last_year: pd.Series):
    """Forecast total vs the same period last year (None if history is incomplete)."""
    if last_year.isna().any() or last_year.sum() <= 0:
        return None
    return float(forecast.sum() / last_year.sum() - 1)


def store_item_table(fc: pd.DataFrame) -> pd.DataFrame:
    g = fc.groupby(["store", "item"])["forecast"]
    out = g.agg(total="sum", avg_per_day="mean", peak="max").reset_index()
    return out.sort_values("total", ascending=False).reset_index(drop=True)


def improvement_vs_best_baseline(results: pd.DataFrame, model_name: str = "LightGBM (global)") -> dict | None:
    """SMAPE and total-level WAPE improvement of the model over the best non-ML baseline."""
    row = results[results["Model"] == model_name]
    base = results[results["Model"].str.contains("Naive|SNaive", regex=True)].dropna(subset=["SMAPE"])
    if row.empty or base.empty:
        return None
    best = base.sort_values("SMAPE").iloc[0]
    m = row.iloc[0]
    return {
        "model_smape": float(m["SMAPE"]),
        "baseline_name": str(best["Model"]),
        "baseline_smape": float(best["SMAPE"]),
        "smape_gain_pct": float(100 * (best["SMAPE"] - m["SMAPE"]) / best["SMAPE"]),
        "model_total_wape": float(m["Total-level WAPE %"]),
        "baseline_total_wape": float(best["Total-level WAPE %"]),
    }
