import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import dashboard_utils as du

st.set_page_config(page_title="Sales Forecast Studio", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

BASE = os.path.dirname(os.path.abspath(__file__))
P = lambda *parts: os.path.join(BASE, *parts)
FORECAST_PATH = P("results", "forecast_next_90_days.csv")
RESULTS_PATH = P("results", "results.csv")
IMPORTANCE_PATH = P("results", "feature_importance.csv")

VIOLET, PINK, ORANGE, TEAL = "#7C5CFF", "#FF5C8A", "#FFB86B", "#2DD4BF"
REPO_URL = "https://github.com/D0027/sales-forecasting-lightgbm"
DATA_URL = "https://www.kaggle.com/competitions/demand-forecasting-kernels-only"

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
footer { visibility: hidden; }
:root { --card: rgba(255,255,255,.045); --border: rgba(255,255,255,.10); --muted: rgba(232,234,240,.62); }

.hero { position: relative; padding: 2rem 2.2rem; border-radius: 24px; margin-bottom: 1.2rem; overflow: hidden;
  background: radial-gradient(900px 380px at 0% 0%, rgba(124,92,255,.38), transparent 60%),
              radial-gradient(800px 380px at 100% 100%, rgba(255,92,138,.30), transparent 60%),
              linear-gradient(135deg, #141a2e, #0f1424);
  border: 1px solid var(--border); }
.hero .badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: .78rem; font-weight: 600;
  background: rgba(255,255,255,.08); border: 1px solid var(--border); }
.hero h1 { font-size: 2.4rem; line-height: 1.1; margin: .7rem 0 .4rem; padding: 0; font-weight: 800; letter-spacing: -.02em;
  background: linear-gradient(90deg, #fff, #c9bfff 45%, #ff9bb8); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { margin: 0; color: var(--muted); max-width: 760px; font-size: 1.02rem; }
.pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1rem; }
.pill { padding: 5px 12px; border-radius: 999px; font-size: .8rem; background: rgba(255,255,255,.07); border: 1px solid var(--border); }
.pill b { color: #fff; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: .6rem 0 1rem; }
.kpi { background: var(--card); border: 1px solid var(--border); border-left: 4px solid var(--a); border-radius: 16px; padding: 14px 16px;
  transition: transform .15s ease, box-shadow .15s ease; }
.kpi:hover { transform: translateY(-3px); box-shadow: 0 12px 28px rgba(0,0,0,.30); }
.kpi-l { font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; }
.kpi-v { font-size: 1.65rem; font-weight: 800; margin: 2px 0; }
.kpi-s { font-size: .78rem; color: var(--muted); }

.note { border: 1px solid var(--border); background: var(--card); border-radius: 14px; padding: 12px 16px; font-size: .9rem; color: var(--muted); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 12px; padding: 8px 16px; }
.stButton > button, .stDownloadButton > button { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub="", accent=VIOLET):
    return (f'<div class="kpi" style="--a:{accent}"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>')


def kpi_row(cards):
    st.markdown('<div class="kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=13),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)")
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def fmt(x, d=0):
    return f"{x:,.{d}f}"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data
def load_forecast():
    return pd.read_csv(FORECAST_PATH, parse_dates=["date"])


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner="Reading uploaded history ...")
def load_history(raw: bytes):
    return du.load_history_csv(io.BytesIO(raw))


if not os.path.exists(FORECAST_PATH):
    st.error("results/forecast_next_90_days.csv not found.")
    st.info("Create it with:  `python predict.py --data data/train.csv --out results/forecast_next_90_days.csv`")
    st.stop()

fc = load_forecast()
results = load_csv(RESULTS_PATH) if os.path.exists(RESULTS_PATH) else None
importance = load_csv(IMPORTANCE_PATH) if os.path.exists(IMPORTANCE_PATH) else None
stores = sorted(fc["store"].unique())
items = sorted(fc["item"].unique())

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 📈 Sales Forecast Studio")
    st.caption("90-day store-item demand forecast built with a global LightGBM model.")
    st.markdown("---")
    st.markdown("**Optional: add history**")
    up = st.file_uploader("Upload Kaggle train.csv", type="csv", label_visibility="collapsed")
    st.caption("Adds recent history, a same-period-last-year line and YoY growth to the Forecast tab. "
               "The file is only read in your session.")
    st.markdown("---")
    st.markdown(f"[Dataset: Store Item Demand Forecasting]({DATA_URL})")
    st.markdown(f"[GitHub repository]({REPO_URL})")
    st.caption("Built by Deepak")

hist_df = None
if up is not None:
    try:
        hist_df = load_history(up.getvalue())
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
d0, d1 = fc["date"].min(), fc["date"].max()
st.markdown(f"""
<div class="hero">
  <span class="badge">LightGBM · leak-free 90-day forecasting</span>
  <h1>Sales Forecast Studio</h1>
  <p>Explore the next 90 days of demand for every store and item, compare the model against baselines,
  and see which signals drive the forecast.</p>
  <div class="pills">
    <span class="pill"><b>{len(stores)}</b> stores</span>
    <span class="pill"><b>{len(items)}</b> items</span>
    <span class="pill"><b>{fc.groupby(["store", "item"]).ngroups}</b> series</span>
    <span class="pill">Forecast <b>{d0:%d %b %Y}</b> to <b>{d1:%d %b %Y}</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_fc, tab_si, tab_model, tab_ins = st.tabs(
    ["📈 Forecast", "🏬 Stores & Items", "🧪 Model performance", "🔍 Insights"])

# --------------------------------------------------------------------------- #
# Tab 1: Forecast explorer
# --------------------------------------------------------------------------- #
with tab_fc:
    c1, c2, c3 = st.columns([1, 1, 1.3])
    store_sel = c1.selectbox("Store", ["All stores"] + stores)
    item_sel = c2.selectbox("Item", ["All items"] + items)
    gran = c3.segmented_control("View", ["Daily", "Weekly"], default="Daily")
    gran = gran or "Daily"
    store_f = None if store_sel == "All stores" else store_sel
    item_f = None if item_sel == "All items" else item_sel

    s = du.daily_series(fc, "forecast", store_f, item_f)
    k = du.summary_kpis(s)

    cards = [
        kpi("Forecast total (90 days)", fmt(k["total"]), "units", VIOLET),
        kpi("Average per day", fmt(k["avg_day"], 1), "units / day", PINK),
        kpi("Peak day", fmt(k["peak_value"]), f"{k['peak_date']:%a, %d %b %Y}", ORANGE),
        kpi("Weekend uplift", f"{k['weekend_uplift'] * 100:+.0f}%", "Sat/Sun vs Mon-Fri", TEAL),
    ]
    hist_s, ly = None, None
    if hist_df is not None:
        hist_s = du.daily_series(hist_df, "sales", store_f, item_f)
        ly = du.same_period_last_year(hist_s, s.index)
        g = du.yoy_growth(s, ly)
        cards.append(kpi("vs same days last year", "n/a" if g is None else f"{g * 100:+.1f}%",
                         "364 days earlier (weekday aligned)", "#8AB4F8"))
    kpi_row(cards)

    fig = go.Figure()
    if gran == "Weekly":
        w = du.weekly_complete(s)
        fig.add_bar(x=w.index, y=w.values, name="Forecast (weekly total)", marker_color=VIOLET)
        if ly is not None and not ly.isna().any():
            wl = du.weekly_complete(ly)
            fig.add_scatter(x=wl.index, y=wl.values, name="Same weeks last year", mode="lines+markers",
                            line=dict(color=ORANGE, dash="dot"))
        fig.update_layout(title="Weekly forecast (complete Mon-Sun weeks only)")
    else:
        if hist_s is not None:
            h = hist_s[hist_s.index >= s.index[0] - pd.Timedelta(days=120)]
            h = h[h.index < s.index[0]]
            fig.add_scatter(x=h.index, y=h.values, name="History", mode="lines",
                            line=dict(color="rgba(200,205,225,.6)", width=1.6))
        if ly is not None and not ly.isna().any():
            fig.add_scatter(x=ly.index, y=ly.values, name="Same days last year", mode="lines",
                            line=dict(color=ORANGE, dash="dot", width=1.4))
        fig.add_scatter(x=s.index, y=s.values, name="Forecast", mode="lines",
                        line=dict(color=PINK, width=2.4), fill="tozeroy", fillcolor="rgba(255,92,138,.08)")
        fig.add_scatter(x=s.index, y=s.rolling(7, center=True, min_periods=4).mean(), name="7-day average",
                        mode="lines", line=dict(color="#fff", width=1.4, dash="dash"))
        fig.update_layout(title="Daily forecast")
    show(style_fig(fig, 420))

    a, b = st.columns(2)
    dow = s.groupby(s.index.dayofweek).mean().reindex(range(7))
    f1 = px.bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=dow.values,
                labels={"x": "", "y": "avg units / day"}, title="Average forecast by weekday",
                color_discrete_sequence=[VIOLET])
    a.plotly_chart(style_fig(f1, 320), width="stretch", config={"displaylogo": False})
    mon = s.groupby(s.index.to_period("M")).sum()
    f2 = px.bar(x=[p.strftime("%b %Y") for p in mon.index], y=mon.values,
                labels={"x": "", "y": "units"}, title="Forecast by month", color_discrete_sequence=[PINK])
    b.plotly_chart(style_fig(f2, 320), width="stretch", config={"displaylogo": False})

    with st.expander("Data and download"):
        sel = du.filter_df(fc, store_f, item_f)
        out = sel.groupby("date", as_index=False)["forecast"].sum() if (store_f is None or item_f is None) else sel
        st.dataframe(out, width="stretch", hide_index=True, height=260)
        st.download_button("Download this view (CSV)", out.to_csv(index=False).encode(),
                           file_name="forecast_selection.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Tab 2: Stores & items
# --------------------------------------------------------------------------- #
with tab_si:
    tbl = du.store_item_table(fc)
    st.subheader("Where the demand is")
    heat = tbl.pivot(index="store", columns="item", values="total").reindex(index=stores, columns=items)
    fh = px.imshow(heat.values, x=[str(i) for i in items], y=[str(s_) for s_ in stores], aspect="auto",
                   color_continuous_scale=[[0, "#14172b"], [0.5, VIOLET], [1, PINK]],
                   labels=dict(x="Item", y="Store", color="units (90d)"), title="Forecast units by store and item")
    fh.update_layout(hovermode="closest")
    show(style_fig(fh, 360))

    a, b = st.columns(2)
    st_tot = tbl.groupby("store")["total"].sum().reindex(stores)
    share = 100 * st_tot / st_tot.sum()
    f3 = go.Figure(go.Bar(x=[str(x) for x in st_tot.index], y=st_tot.values, marker_color=VIOLET,
                          text=[f"{v:.1f}%" for v in share], textposition="outside"))
    f3.update_layout(title="Total forecast by store (share of total)")
    a.plotly_chart(style_fig(f3, 340), width="stretch", config={"displaylogo": False})

    scope = b.selectbox("Items ranking for", ["All stores"] + stores, key="rank_scope")
    sub = tbl if scope == "All stores" else tbl[tbl["store"] == scope]
    it = sub.groupby("item")["total"].sum().sort_values(ascending=False)
    show_df = pd.concat([it.head(8), it.tail(8)])
    f4 = go.Figure(go.Bar(x=show_df.values[::-1], y=[f"item {i}" for i in show_df.index][::-1], orientation="h",
                          marker_color=[PINK] * 8 + [VIOLET] * 8))
    f4.update_layout(title="Top 8 and bottom 8 items", hovermode="closest")
    b.plotly_chart(style_fig(f4, 340), width="stretch", config={"displaylogo": False})

    st.subheader("Store-item leaderboard")
    st.dataframe(
        tbl.rename(columns={"total": "total_units", "avg_per_day": "avg_per_day", "peak": "peak_day_units"}),
        width="stretch", hide_index=True, height=380,
        column_config={
            "total_units": st.column_config.ProgressColumn("Total units (90d)", format="%.0f",
                                                           min_value=0, max_value=float(tbl["total"].max())),
            "avg_per_day": st.column_config.NumberColumn("Avg / day", format="%.1f"),
            "peak_day_units": st.column_config.NumberColumn("Peak day", format="%.0f"),
        })

# --------------------------------------------------------------------------- #
# Tab 3: Model performance
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader("How good is the model?")
    st.caption("Honest test: the last 90 days of history (Oct 3 to Dec 31, 2017) were never used for training or tuning.")
    if results is None:
        st.info("results/results.csv not found.")
    else:
        imp = du.improvement_vs_best_baseline(results)
        if imp:
            kpi_row([
                kpi("LightGBM SMAPE", f"{imp['model_smape']:.2f}", "store-item daily level", VIOLET),
                kpi("Best baseline SMAPE", f"{imp['baseline_smape']:.2f}", imp["baseline_name"], ORANGE),
                kpi("Improvement", f"{imp['smape_gain_pct']:.1f}%", "lower SMAPE than best baseline", TEAL),
                kpi("Total-level WAPE", f"{imp['model_total_wape']:.2f}%",
                    f"baseline: {imp['baseline_total_wape']:.2f}%", PINK),
            ])
        metric_cols = [c for c in ["SMAPE", "WAPE %", "MAE", "RMSE", "Total-level WAPE %"] if c in results.columns]
        styled = (results.style.format({c: "{:.2f}" for c in metric_cols}, na_rep="-")
                  .highlight_min(subset=metric_cols, color="rgba(45,212,191,.28)"))
        st.dataframe(styled, width="stretch", hide_index=True)
        st.caption("Green = best value in each column. SMAPE is the competition metric. "
                   "The Holt-Winters ETS model is evaluated on total sales only.")

    for img, cap in [("model_comparison.png", "Model comparison"), ("forecast_total.png", "Total sales: forecast vs actual (test period)")]:
        if os.path.exists(P("images", img)):
            st.image(P("images", img), caption=cap)
    a, b = st.columns(2)
    if os.path.exists(P("images", "diagnostics.png")):
        a.image(P("images", "diagnostics.png"), caption="Error by horizon, weekday and store")
    if os.path.exists(P("images", "forecast_series.png")):
        b.image(P("images", "forecast_series.png"), caption="Best, median and worst series")

# --------------------------------------------------------------------------- #
# Tab 4: Insights
# --------------------------------------------------------------------------- #
with tab_ins:
    a, b = st.columns([1.2, 1])
    with a:
        st.subheader("What drives the forecast?")
        if importance is not None:
            top = importance.sort_values("gain %", ascending=False).head(15).iloc[::-1]
            fi = go.Figure(go.Bar(x=top["gain %"], y=top["feature"], orientation="h", marker_color=VIOLET,
                                  text=[f"{v:.1f}%" for v in top["gain %"]], textposition="outside"))
            fi.update_layout(title="LightGBM feature importance (gain %)", hovermode="closest")
            show(style_fig(fi, 480))
    with b:
        st.subheader("Reading the features")
        st.markdown(
            "- **`ya_5`, `lag_364`, `lag_357`, `lag_371`**: sales on the same weekday one year earlier. "
            "Demand has strong yearly seasonality, so these dominate.\n"
            "- **`rm_*`, `growth`**: recent level and year-over-year trend.\n"
            "- **`sdow_*`**: same-weekday averages of the last weeks.\n"
            "- **calendar + `store`, `item`**: weekday, month, and series identity.\n\n"
            "Every feature is built from data at least **91 days old**, so a 90-day forecast needs no "
            "recursive prediction and cannot leak the future (verified by an automatic check).")

    st.subheader("Data exploration")
    t1, t2, t3, t4 = st.tabs(["Trend", "Seasonality", "Stores x months", "Validation split"])
    for tab, img in [(t1, "eda_trend.png"), (t2, "eda_seasonality.png"), (t3, "eda_items_heatmap.png"), (t4, "split.png")]:
        with tab:
            if os.path.exists(P("images", img)):
                st.image(P("images", img))

    with st.expander("Limitations and honest notes"):
        st.markdown(
            "- The test period (Q4) is harder than the validation period, so test SMAPE is higher.\n"
            "- The model leans heavily on the year-ago pattern; a sudden structural change "
            "(new store, promotion, price change) would not be anticipated.\n"
            "- The model slightly under-forecasts on the test window (about -1.5% in total).\n"
            "- No holidays, prices or promotions are available in this dataset.\n"
            "- Metrics here are not comparable with the Kaggle leaderboard (different test period).")
