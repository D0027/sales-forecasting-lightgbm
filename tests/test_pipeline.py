import numpy as np
import pandas as pd
import pytest

from src import dashboard_utils as du
from src import forecasting as fc


def make_df(n_days=1300, n_stores=2, n_items=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-01", periods=n_days)
    t = np.arange(n_days)
    weekly = np.array([0.8, 0.85, 0.9, 1.0, 1.15, 1.3, 1.25])[dates.dayofweek]
    rows = []
    for s in range(1, n_stores + 1):
        for i in range(1, n_items + 1):
            lam = (10 + 3 * i + 2 * s) * (1 + 0.0003 * t) * (1 + 0.2 * np.sin(2 * np.pi * t / 365.25)) * weekly
            rows.append(pd.DataFrame({"date": dates, "store": s, "item": i, "sales": rng.poisson(lam)}))
    return pd.concat(rows).sort_values(["store", "item", "date"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def setup():
    df = make_df()
    panel = fc.build_panel(df)
    feats = fc.build_features(panel.S_all, panel.all_dates)
    return df, panel, feats, fc.make_splits(panel)


def test_panel_shape(setup):
    df, panel, feats, sp = setup
    assert panel.n_series == 6
    assert panel.S_all.shape == (6, 1300 + fc.HORIZON)
    assert np.isnan(panel.S_all[:, panel.n_hist:]).all()
    assert len(sp.test_idx) == len(sp.val_idx) == fc.HORIZON


def test_features_are_leak_free(setup):
    _, panel, feats, sp = setup
    fc.check_no_leak(panel, feats, sp.t_test, sp.test_idx)
    fc.check_no_leak(panel, feats, sp.t_val, sp.val_idx)


def test_leak_check_catches_a_leak(setup):
    _, panel, feats, sp = setup
    bad = dict(feats)
    # a "feature" that peeks at the actual value of the day itself must be flagged
    leaky = np.array(feats["lag_91"])
    leaky[:, sp.test_idx] = panel.S_all[:, sp.test_idx]
    bad["lag_91"] = leaky
    with pytest.raises(AssertionError):
        fc.check_no_leak(panel, bad, sp.t_test, sp.test_idx)


def test_metrics():
    a = np.array([10.0, 20.0, 0.0])
    assert fc.smape(a, a) == 0
    assert fc.wape(a, a) == 0
    assert fc.smape([0, 0], [0, 0]) == 0
    assert fc.smape([10], [30]) == pytest.approx(100.0)   # 200 * 20 / 40


def test_forecast_future_shape_and_range(setup):
    df, panel, feats, sp = setup
    store_cats, item_cats = [1, 2], [1, 2, 3]
    X, y = fc.make_rows(panel, feats, np.arange(sp.t_train0, panel.n_hist), store_cats, item_cats)
    booster, _ = fc.fit_lgbm(X, y, rounds=25)
    out = fc.forecast_future(panel, booster, feats, store_cats, item_cats)
    assert len(out) == panel.n_series * fc.HORIZON
    assert out["date"].min() == panel.hist_dates[-1] + pd.Timedelta(days=1)
    assert out["date"].nunique() == fc.HORIZON
    assert (out["forecast"] >= 0).all() and np.isfinite(out["forecast"]).all()


def test_weekly_complete_drops_partial_weeks():
    s = pd.Series(1.0, index=pd.date_range("2018-01-01", "2018-03-31"))   # Mon .. Sat
    w = du.weekly_complete(s)
    assert len(w) == 12 and (w == 7).all()


def test_yoy_and_baseline_gain():
    idx = pd.date_range("2018-01-01", periods=7)
    hist = pd.Series(10.0, index=idx - pd.Timedelta(days=364))
    ly = du.same_period_last_year(hist, idx)
    assert du.yoy_growth(pd.Series(11.0, index=idx), ly) == pytest.approx(0.1)
    res = pd.DataFrame({"Model": ["Smoothed Seasonal Naive", "LightGBM (global)"],
                        "SMAPE": [20.0, 15.0], "Total-level WAPE %": [5.0, 2.0]})
    assert du.improvement_vs_best_baseline(res)["smape_gain_pct"] == pytest.approx(25.0)
