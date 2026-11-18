"""
Reproduce the full pipeline on the Kaggle train.csv:
  load -> leak-free features -> baselines + ETS -> LightGBM (validation) -> test scores
  -> final fit on all history -> 90-day forecast -> save model + results.

Usage:
    python train.py --data data/train.csv
    python train.py --data data/train.csv --out-dir outputs --skip-leak-check
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from src import forecasting as fc


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate the sales forecasting pipeline.")
    ap.add_argument("--data", required=True, help="path to Kaggle train.csv (date, store, item, sales)")
    ap.add_argument("--out-dir", default="outputs", help="where to write model, forecast and results")
    ap.add_argument("--skip-leak-check", action="store_true", help="skip the (fast) leakage assertion")
    ap.add_argument("--no-ets", action="store_true", help="skip the Holt-Winters baseline (needs statsmodels)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t_all = time.time()
    print("Loading data ...")
    df = fc.load_train_csv(args.data)
    panel = fc.build_panel(df)
    sp = fc.make_splits(panel)
    print(f"  {panel.n_series} series x {panel.n_hist} days "
          f"({panel.hist_dates[0].date()} -> {panel.hist_dates[-1].date()})")
    print(f"  train {panel.all_dates[sp.train_idx[0]].date()} -> {panel.all_dates[sp.train_idx[-1]].date()} | "
          f"val {panel.all_dates[sp.val_idx[0]].date()} -> {panel.all_dates[sp.val_idx[-1]].date()} | "
          f"test {panel.all_dates[sp.test_idx[0]].date()} -> {panel.all_dates[sp.test_idx[-1]].date()}")

    print("Building features ...")
    feats = fc.build_features(panel.S_all, panel.all_dates)
    if not args.skip_leak_check:
        fc.check_no_leak(panel, feats, sp.t_test, sp.test_idx)
        fc.check_no_leak(panel, feats, sp.t_val, sp.val_idx)
        print(f"  leakage check OK ({len(feats)} features)")

    store_cats = sorted(panel.sid["store"].unique())
    item_cats = sorted(panel.sid["item"].unique())
    A_test = panel.S_all[:, sp.test_idx]
    results = []

    print("Baselines ...")
    for name, F in fc.baseline_forecasts(panel, feats, sp.t_test).items():
        results.append(fc.score_row(name, A_test, F))

    if not args.no_ets:
        try:
            name, ets_total = fc.ets_total_forecast(panel.S_hist.sum(axis=0)[: sp.t_test])
            results.append({"Model": name, "SMAPE": np.nan, "WAPE %": np.nan, "MAE": np.nan, "RMSE": np.nan,
                            "Total-level WAPE %": fc.wape(A_test.sum(axis=0), ets_total)})
        except ImportError:
            print("  statsmodels not installed, skipping ETS")

    print("LightGBM (validation, early stopping) ...")
    X_tr, y_tr = fc.make_rows(panel, feats, sp.train_idx, store_cats, item_cats)
    X_va, y_va = fc.make_rows(panel, feats, sp.val_idx, store_cats, item_cats)
    X_te, _ = fc.make_rows(panel, feats, sp.test_idx, store_cats, item_cats)
    booster_val, best_iter = fc.fit_lgbm(X_tr, y_tr, X_va, y_va)
    val_pred = fc.predict_sales(booster_val, X_va, num_iteration=best_iter)
    print(f"  best iteration {best_iter} | validation SMAPE {fc.smape(y_va, val_pred):.2f}")

    print("LightGBM (refit on train + validation, score on test) ...")
    X_trva = pd.concat([X_tr, X_va], ignore_index=True)
    y_trva = np.concatenate([y_tr, y_va])
    booster_test, _ = fc.fit_lgbm(X_trva, y_trva, rounds=best_iter)
    F_lgb = fc.predict_sales(booster_test, X_te).reshape(panel.n_series, len(sp.test_idx))
    results.append(fc.score_row("LightGBM (global)", A_test, F_lgb))

    res = pd.DataFrame(results)
    res.round(4).to_csv(os.path.join(args.out_dir, "results.csv"), index=False)
    print("\nTest-set results (last 90 days):")
    print(res.round(3).to_string(index=False))

    print("\nFinal model on all history -> next 90 days ...")
    X_all, y_all = fc.make_rows(panel, feats, np.arange(sp.t_train0, panel.n_hist), store_cats, item_cats)
    booster, _ = fc.fit_lgbm(X_all, y_all, rounds=best_iter)
    forecast = fc.forecast_future(panel, booster, feats, store_cats, item_cats)
    forecast.to_csv(os.path.join(args.out_dir, "forecast_next_90_days.csv"), index=False)

    meta = {
        "horizon": fc.HORIZON, "min_lag": fc.MIN_LAG, "lags": fc.LAGS, "roll_windows": fc.ROLL_WINDOWS,
        "features": fc.feature_names(feats), "best_iteration": best_iter, "lgb_params": fc.LGB_PARAMS,
        "store_categories": [int(s) for s in store_cats], "item_categories": [int(i) for i in item_cats],
        "train_range": [str(panel.all_dates[sp.t_train0].date()), str(panel.hist_dates[-1].date())],
        "forecast_range": [str(forecast["date"].min().date()), str(forecast["date"].max().date())],
    }
    fc.save_model(booster, os.path.join(args.out_dir, "lgbm_sales_model.txt"), meta)

    imp = pd.DataFrame({"feature": booster.feature_name(),
                        "gain": booster.feature_importance(importance_type="gain")})
    imp["gain %"] = 100 * imp["gain"] / imp["gain"].sum()
    imp.sort_values("gain", ascending=False).round(4).to_csv(
        os.path.join(args.out_dir, "feature_importance.csv"), index=False)

    print(f"Saved model, forecast and results to '{args.out_dir}/' in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
