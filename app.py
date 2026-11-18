import html
import inspect
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

VIOLET, PINK, ORANGE, TEAL, SKY = "#7C5CFF", "#FF5C8A", "#FFB86B", "#2DD4BF", "#8AB4F8"
COLORWAY = [VIOLET, PINK, ORANGE, TEAL, SKY, "#C084FC", "#F472B6"]
GROUP_COLORS = {
    "Year-ago pattern": PINK, "Recent lags": VIOLET, "Recent level & trend": TEAL,
    "Calendar": ORANGE, "Store / item identity": SKY,
}
REPO_URL = "https://github.com/D0027/sales-forecasting-lightgbm"
DATA_URL = "https://www.kaggle.com/competitions/demand-forecasting-kernels-only"
LINK_KW = {"width": "stretch"} if "width" in inspect.signature(st.link_button).parameters else {"use_container_width": True}

# --------------------------------------------------------------------------- #
# Design system (CSS)
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #070b14; --card: rgba(255,255,255,.045); --card-2: rgba(255,255,255,.02);
  --border: rgba(255,255,255,.09); --muted: rgba(226,230,240,.62);
  --violet: #7C5CFF; --pink: #FF5C8A; --orange: #FFB86B; --teal: #2DD4BF; --sky: #8AB4F8;
}
.stApp { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  background:
    radial-gradient(900px 500px at 8% -5%, rgba(124,92,255,.20), transparent 60%),
    radial-gradient(800px 500px at 95% 0%, rgba(255,92,138,.14), transparent 60%),
    radial-gradient(900px 600px at 50% 110%, rgba(45,212,191,.08), transparent 60%),
    var(--bg); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input { font-family: inherit; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }
footer, #MainMenu { visibility: hidden; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.14); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }

@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes shimmer { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(45,212,191,.55); } 50% { box-shadow: 0 0 0 7px rgba(45,212,191,0); } }

/* ---------- hero ---------- */
.hero { position: relative; padding: 2.1rem 2.3rem 1.8rem; border-radius: 26px; margin-bottom: 1.1rem; overflow: hidden;
  border: 1px solid transparent; animation: fadeUp .5s ease both;
  background:
    radial-gradient(700px 320px at 0% 0%, rgba(124,92,255,.42), transparent 60%) padding-box,
    radial-gradient(700px 320px at 100% 100%, rgba(255,92,138,.32), transparent 60%) padding-box,
    linear-gradient(135deg, #141a2e, #0e1322) padding-box,
    linear-gradient(135deg, rgba(124,92,255,.75), rgba(255,92,138,.45), rgba(255,184,107,.35)) border-box; }
.hero-top { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.badge { display: inline-flex; align-items: center; gap: 8px; padding: 5px 13px; border-radius: 999px; font-size: .76rem; font-weight: 600;
  background: rgba(255,255,255,.07); border: 1px solid var(--border); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--teal); animation: pulse 2s infinite; display: inline-block; }
.hero h1 { font-size: 2.7rem; line-height: 1.08; margin: .75rem 0 .5rem; padding: 0; font-weight: 800; letter-spacing: -.03em;
  background: linear-gradient(90deg, #ffffff, #cbbfff, #ff9bb8, #ffd2a0, #ffffff); background-size: 200% auto;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: shimmer 9s linear infinite; }
.hero p { margin: 0; color: var(--muted); max-width: 780px; font-size: 1.02rem; line-height: 1.55; }
.pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1.1rem; }
.pill { padding: 5px 13px; border-radius: 999px; font-size: .8rem; background: rgba(255,255,255,.06); border: 1px solid var(--border); color: var(--muted); }
.pill b { color: #fff; font-weight: 700; }

/* ---------- context chips ---------- */
.ctx { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: .2rem 0 .8rem; color: var(--muted); font-size: .82rem; }
.chip { padding: 4px 11px; border-radius: 999px; font-size: .78rem; font-weight: 600; color: #fff;
  background: rgba(124,92,255,.22); border: 1px solid rgba(124,92,255,.5); }
.chip.warn { background: rgba(255,184,107,.18); border-color: rgba(255,184,107,.55); }

/* ---------- KPI cards ---------- */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: .5rem 0 1rem; }
.kpi { position: relative; overflow: hidden; border-radius: 18px; padding: 15px 17px 14px; border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,.065), rgba(255,255,255,.02)); animation: fadeUp .5s ease both;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
.kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--a); }
.kpi:hover { transform: translateY(-4px); box-shadow: 0 16px 34px rgba(0,0,0,.35); border-color: var(--a); }
.kpi-l { font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); font-weight: 700; }
.kpi-v { font-size: 1.75rem; font-weight: 800; margin: 3px 0 1px; letter-spacing: -.02em; }
.kpi-s { font-size: .78rem; color: var(--muted); }
.spark { position: absolute; right: 12px; bottom: 12px; width: 92px; height: 34px; opacity: .95; }

/* ---------- insight cards ---------- */
.insights { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; margin: .2rem 0 1rem; }
.insight { display: flex; gap: 11px; align-items: flex-start; padding: 12px 14px; border-radius: 14px; border: 1px solid var(--border);
  background: var(--card); font-size: .88rem; line-height: 1.4; color: rgba(236,239,247,.88); animation: fadeUp .6s ease both; }
.insight .ic { font-size: 1.25rem; line-height: 1; margin-top: 1px; }
.insight b { color: #fff; }

/* ---------- section headings ---------- */
.sec { margin: 1.1rem 0 .55rem; }
.sec-t { font-size: 1.12rem; font-weight: 700; letter-spacing: -.01em; }
.sec-s { font-size: .84rem; color: var(--muted); margin-top: 2px; }

/* ---------- leaderboard table ---------- */
.lb { border: 1px solid var(--border); border-radius: 18px; overflow-x: auto; background: var(--card-2); margin: .3rem 0 1rem; }
.lb table { width: 100%; border-collapse: collapse; font-size: .88rem; }
.lb th { text-align: left; padding: 12px 14px; font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; color: var(--muted);
  border-bottom: 1px solid var(--border); font-weight: 700; white-space: nowrap; }
.lb td { padding: 12px 14px; border-bottom: 1px solid rgba(255,255,255,.05); white-space: nowrap; }
.lb tr:last-child td { border-bottom: none; }
.lb tr.top td { background: linear-gradient(90deg, rgba(45,212,191,.10), transparent); }
.lb td.best { color: var(--teal); font-weight: 700; }
.lb .mname { font-weight: 600; }
.lb .tag { margin-left: 8px; padding: 2px 8px; border-radius: 999px; font-size: .66rem; font-weight: 700; color: #04201c; background: var(--teal); }
.lb .bar { display: inline-block; vertical-align: middle; height: 7px; border-radius: 99px; margin-right: 9px;
  background: linear-gradient(90deg, var(--violet), var(--pink)); }

/* ---------- pipeline steps ---------- */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: .4rem 0 1rem; }
.step { position: relative; padding: 16px 16px 14px; border-radius: 18px; border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.02)); }
.step .n { display: inline-flex; width: 28px; height: 28px; border-radius: 9px; align-items: center; justify-content: center;
  font-weight: 800; font-size: .85rem; background: linear-gradient(135deg, var(--violet), var(--pink)); margin-bottom: 8px; }
.step .t { font-weight: 700; margin-bottom: 4px; }
.step .d { font-size: .83rem; color: var(--muted); line-height: 1.45; }

/* ---------- streamlit widgets ---------- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0e1424, #090d19); border-right: 1px solid var(--border); }
.brand { display: flex; align-items: center; gap: 10px; margin-bottom: .3rem; }
.brand .logo { width: 38px; height: 38px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
  background: linear-gradient(135deg, var(--violet), var(--pink)); box-shadow: 0 8px 20px rgba(124,92,255,.35); }
.brand .nm { font-weight: 800; letter-spacing: -.02em; line-height: 1.1; }
.brand .sb { font-size: .74rem; color: var(--muted); }
.stTabs [data-baseweb="tab-list"] { gap: 4px; padding: 4px; border-radius: 14px; background: rgba(255,255,255,.04); border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height: 40px; border-radius: 10px; padding: 0 16px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, rgba(124,92,255,.38), rgba(255,92,138,.26)); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; border-color: var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.015)); }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--border); }
.stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a { border-radius: 12px; font-weight: 600; transition: transform .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); }
[data-testid="stImage"] img { border-radius: 14px; border: 1px solid var(--border); }
.foot { text-align: center; color: var(--muted); font-size: .8rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); }

@media (max-width: 760px) {
  .hero { padding: 1.3rem 1.2rem; } .hero h1 { font-size: 1.9rem; } .spark { display: none; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
esc = html.escape


def sparkline(values, color, w=92, h=34, max_points=45):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2 or v.max() == v.min():
        return ""
    if len(v) > max_points:
        v = v[np.linspace(0, len(v) - 1, max_points).astype(int)]
    xs = np.linspace(1, w - 1, len(v))
    ys = (h - 3) - (v - v.min()) / (v.max() - v.min()) * (h - 6)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"1,{h - 1} {pts} {w - 1},{h - 1}"
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polygon points="{area}" fill="{color}" opacity="0.18"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" '
            f'stroke-linejoin="round" stroke-linecap="round"/></svg>')


def kpi(label, value, sub="", accent=VIOLET, spark=None):
    return (f'<div class="kpi" style="--a:{accent}"><div class="kpi-l">{esc(label)}</div>'
            f'<div class="kpi-v">{esc(str(value))}</div><div class="kpi-s">{esc(sub)}</div>'
            f'{sparkline(spark, accent) if spark is not None else ""}</div>')


def kpi_row(cards):
    st.markdown('<div class="kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def section(title, subtitle=""):
    sub = f'<div class="sec-s">{esc(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="sec"><div class="sec-t">{esc(title)}</div>{sub}</div>', unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_dark", height=height, colorway=COLORWAY, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=44, b=8), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13),
        title=dict(font=dict(size=15), x=0.01, xanchor="left"),
        hoverlabel=dict(bgcolor="#161c30", bordercolor="rgba(255,255,255,.2)", font=dict(color="#fff")),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def show(fig, height=380, border=True):
    fig = style_fig(fig, height)
    cfg = {"displaylogo": False, "displayModeBar": False}
    if border:
        with st.container(border=True):
            st.plotly_chart(fig, width="stretch", config=cfg)
    else:
        st.plotly_chart(fig, width="stretch", config=cfg)


def fmt(x, d=0):
    return f"{x:,.{d}f}"


def leaderboard_html(res: pd.DataFrame) -> str:
    cols = [c for c in ["SMAPE", "WAPE %", "MAE", "RMSE", "Total-level WAPE %"] if c in res.columns]
    ordered = res.assign(_k=res["SMAPE"].fillna(1e9)).sort_values("_k").drop(columns="_k").reset_index(drop=True)
    best = {c: ordered[c].min() for c in cols}
    smax = ordered["SMAPE"].max()
    head = "".join(f"<th>{esc(c)}</th>" for c in ["#", "Model"] + cols)
    rows = []
    for i, r in ordered.iterrows():
        top = "top" if i == 0 else ""
        tag = '<span class="tag">BEST</span>' if i == 0 else ""
        cells = [f"<td>{i + 1}</td>", f'<td class="mname">{esc(str(r["Model"]))}{tag}</td>']
        for c in cols:
            v = r[c]
            if pd.isna(v):
                cells.append("<td>-</td>")
                continue
            cls = ' class="best"' if np.isclose(v, best[c]) else ""
            bar = ""
            if c == "SMAPE" and smax > 0:
                bar = f'<span class="bar" style="width:{max(6, 70 * v / smax):.0f}px"></span>'
            cells.append(f"<td{cls}>{bar}{v:.2f}</td>")
        rows.append(f'<tr class="{top}">' + "".join(cells) + "</tr>")
    return f'<div class="lb"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


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
d0, d1 = fc["date"].min(), fc["date"].max()

# --------------------------------------------------------------------------- #
# Sidebar: brand + global filters
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">📈</div><div><div class="nm">Sales Forecast<br>Studio</div>'
                '<div class="sb">LightGBM · 90-day horizon</div></div></div>', unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("**Filters**  \n<span style='color:rgba(226,230,240,.62);font-size:.78rem'>apply to the Forecast tab</span>",
                unsafe_allow_html=True)
    store_sel = st.selectbox("Store", ["All stores"] + stores, key="f_store")
    item_sel = st.selectbox("Item", ["All items"] + items, key="f_item")
    gran = st.segmented_control("Chart view", ["Daily", "Weekly"], default="Daily", key="f_view") or "Daily"
    adj = st.slider("What-if demand shift", -30, 30, 0, 1, format="%d%%", key="f_adj",
                    help="Scenario tool: scales the forecast up or down. It is NOT part of the model.")
    st.markdown("---")
    st.markdown("**Add history** (optional)")
    up = st.file_uploader("Upload Kaggle train.csv", type="csv", label_visibility="collapsed")
    st.caption("Adds recent history, a same-period-last-year line and YoY growth. Read only in your session.")
    st.markdown("---")
    st.link_button("📂 Dataset on Kaggle", DATA_URL, **LINK_KW)
    st.link_button("⭐ Source on GitHub", REPO_URL, **LINK_KW)
    st.caption("Built by Deepak")

hist_df = None
if up is not None:
    try:
        hist_df = load_history(up.getvalue())
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

store_f = None if store_sel == "All stores" else store_sel
item_f = None if item_sel == "All items" else item_sel

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
n_series = fc.groupby(["store", "item"]).ngroups
st.markdown(f"""
<div class="hero">
  <div class="hero-top"><span class="badge"><span class="dot"></span>Live forecast · leak-free features</span></div>
  <h1>Sales Forecast Studio</h1>
  <p>Explore the next 90 days of demand for every store and item, see how the model beats strong baselines,
  and understand which signals drive each forecast.</p>
  <div class="pills">
    <span class="pill"><b>{len(stores)}</b> stores</span>
    <span class="pill"><b>{len(items)}</b> items</span>
    <span class="pill"><b>{n_series}</b> series</span>
    <span class="pill"><b>{fc['date'].nunique()}</b> days ahead</span>
    <span class="pill"><b>{d0:%d %b %Y}</b> to <b>{d1:%d %b %Y}</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_fc, tab_si, tab_model, tab_how = st.tabs(
    ["📈 Forecast", "🏬 Stores & Items", "🧪 Model performance", "🔍 How it works"])

# --------------------------------------------------------------------------- #
# Tab 1: Forecast
# --------------------------------------------------------------------------- #
with tab_fc:
    scale = 1 + adj / 100
    s = du.daily_series(fc, "forecast", store_f, item_f) * scale
    k = du.summary_kpis(s)

    chips = [f'<span class="chip">{esc("All stores" if store_f is None else f"Store {store_f}")}</span>',
             f'<span class="chip">{esc("All items" if item_f is None else f"Item {item_f}")}</span>',
             f'<span class="chip">{esc(gran)} view</span>']
    if adj != 0:
        chips.append(f'<span class="chip warn">What-if {adj:+d}% applied</span>')
    st.markdown('<div class="ctx">Showing:&nbsp;' + "".join(chips) + "</div>", unsafe_allow_html=True)

    dow_means = s.groupby(s.index.dayofweek).mean().reindex(range(7))
    cards = [
        kpi("Forecast total", fmt(k["total"]), "units over 90 days", VIOLET, s.values),
        kpi("Average per day", fmt(k["avg_day"], 1), "units / day", PINK, s.rolling(7, min_periods=3).mean().values),
        kpi("Peak day", fmt(k["peak_value"]), f"{k['peak_date']:%a, %d %b %Y}", ORANGE),
        kpi("Weekend uplift", f"{k['weekend_uplift'] * 100:+.0f}%", "Sat/Sun vs Mon-Fri", TEAL, dow_means.values),
    ]
    hist_s, ly = None, None
    if hist_df is not None:
        hist_s = du.daily_series(hist_df, "sales", store_f, item_f)
        ly = du.same_period_last_year(hist_s, s.index)
        g = du.yoy_growth(s, ly)
        cards.append(kpi("vs same days last year", "n/a" if g is None else f"{g * 100:+.1f}%",
                         "364 days earlier, weekday aligned", SKY))
    kpi_row(cards)

    ins = du.narrative(s)
    st.markdown('<div class="insights">' + "".join(
        f'<div class="insight"><span class="ic">{ic}</span><span>{txt}</span></div>' for ic, txt in ins) + "</div>",
        unsafe_allow_html=True)

    fig = go.Figure()
    if gran == "Weekly":
        w = du.weekly_complete(s)
        fig.add_bar(x=w.index, y=w.values, name="Forecast (weekly total)",
                    marker=dict(color=w.values, colorscale=[[0, VIOLET], [1, PINK]], line=dict(width=0)))
        if ly is not None and not ly.isna().any():
            wl = du.weekly_complete(ly)
            fig.add_scatter(x=wl.index, y=wl.values, name="Same weeks last year", mode="lines+markers",
                            line=dict(color=ORANGE, dash="dot"))
        fig.update_layout(title="Weekly forecast (complete Mon-Sun weeks only)")
    else:
        if hist_s is not None:
            h = hist_s[(hist_s.index >= s.index[0] - pd.Timedelta(days=120)) & (hist_s.index < s.index[0])]
            fig.add_scatter(x=h.index, y=h.values, name="History", mode="lines",
                            line=dict(color="rgba(205,210,230,.55)", width=1.6))
        if ly is not None and not ly.isna().any():
            fig.add_scatter(x=ly.index, y=ly.values, name="Same days last year", mode="lines",
                            line=dict(color=ORANGE, dash="dot", width=1.4))
        fig.add_scatter(x=s.index, y=s.values, name="Forecast", mode="lines",
                        line=dict(color=PINK, width=2.6), fill="tozeroy", fillcolor="rgba(255,92,138,.10)")
        fig.add_scatter(x=s.index, y=s.rolling(7, center=True, min_periods=4).mean(), name="7-day average",
                        mode="lines", line=dict(color="#ffffff", width=1.4, dash="dash"))
        fig.add_scatter(x=[k["peak_date"]], y=[k["peak_value"]], name="Peak", mode="markers+text",
                        text=["Peak"], textposition="top center", showlegend=False,
                        marker=dict(size=11, color=ORANGE, line=dict(color="#fff", width=1.6)))
        fig.update_xaxes(rangeselector=dict(
            buttons=[dict(count=7, label="1W", step="day", stepmode="backward"),
                     dict(count=30, label="1M", step="day", stepmode="backward"),
                     dict(step="all", label="All")],
            bgcolor="rgba(255,255,255,.06)", activecolor=VIOLET, font=dict(color="#fff")))
        fig.update_layout(title="Daily forecast")
    show(fig, 440)

    left, right = st.columns([1.7, 1])
    with left:
        z, labels, dates = du.calendar_matrix(s)
        cal = go.Figure(go.Heatmap(
            z=z, x=labels, y=du.DAY_NAMES, customdata=dates, xgap=4, ygap=4,
            colorscale=[[0, "#1b2040"], [0.5, VIOLET], [1, PINK]], showscale=False,
            hovertemplate="%{customdata}<br><b>%{z:,.0f}</b> units<extra></extra>"))
        cal.update_yaxes(autorange="reversed")
        cal.update_xaxes(side="top", tickangle=0)
        cal.update_layout(title="Calendar view (brighter = more demand)", hovermode="closest")
        show(cal, 330)
    with right:
        dw = go.Figure(go.Bar(x=du.DAY_NAMES, y=dow_means.values,
                              marker=dict(color=dow_means.values, colorscale=[[0, VIOLET], [1, PINK]]),
                              hovertemplate="%{x}: %{y:,.0f} units / day<extra></extra>"))
        dw.update_layout(title="Average by weekday", hovermode="closest")
        show(dw, 330)

    c1, c2 = st.columns(2)
    with c1:
        mon = s.groupby(s.index.to_period("M")).sum()
        mf = go.Figure(go.Bar(x=[p.strftime("%b %Y") for p in mon.index], y=mon.values, marker_color=VIOLET,
                              text=[fmt(v) for v in mon.values], textposition="outside",
                              hovertemplate="%{x}: %{y:,.0f} units<extra></extra>"))
        mf.update_layout(title="Forecast by month", hovermode="closest")
        mf.update_yaxes(range=[0, mon.values.max() * 1.18])
        show(mf, 320)
    with c2:
        cum = s.cumsum()
        cf = go.Figure(go.Scatter(x=cum.index, y=cum.values, mode="lines", fill="tozeroy",
                                  line=dict(color=TEAL, width=2.4), fillcolor="rgba(45,212,191,.12)",
                                  hovertemplate="%{x|%d %b}: %{y:,.0f} units<extra></extra>"))
        cf.update_layout(title="Cumulative units")
        show(cf, 320)

    with st.expander("Data and download"):
        sel = du.filter_df(fc, store_f, item_f)
        out = sel.groupby("date", as_index=False)["forecast"].sum() if (store_f is None or item_f is None) else sel.copy()
        out["forecast"] = (out["forecast"] * scale).round(2)
        st.dataframe(out, width="stretch", hide_index=True, height=260)
        st.download_button("Download this view (CSV)", out.to_csv(index=False).encode(),
                           file_name="forecast_selection.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Tab 2: Stores & items
# --------------------------------------------------------------------------- #
with tab_si:
    tbl = du.store_item_table(fc)
    section("Where the demand is", "Forecast units over the next 90 days for every store-item pair")
    heat = tbl.pivot(index="store", columns="item", values="total").reindex(index=stores, columns=items)
    fh = px.imshow(heat.values, x=[str(i) for i in items], y=[f"Store {s_}" for s_ in stores], aspect="auto",
                   color_continuous_scale=[[0, "#14172b"], [0.5, VIOLET], [1, PINK]],
                   labels=dict(x="Item", y="", color="units"))
    fh.update_layout(hovermode="closest", title="Store x item heatmap")
    show(fh, 350)

    a, b = st.columns(2)
    with a:
        tm = tbl.copy()
        tm["Store"] = "Store " + tm["store"].astype(str)
        tm["Item"] = "Item " + tm["item"].astype(str)
        tr = px.treemap(tm, path=[px.Constant("All stores"), "Store", "Item"], values="total", color="total",
                        color_continuous_scale=[[0, "#232a52"], [0.5, VIOLET], [1, PINK]], maxdepth=2)
        tr.update_traces(root_color="rgba(0,0,0,0)", marker=dict(line=dict(width=1.5, color="#0a0f1c")),
                         hovertemplate="<b>%{label}</b><br>%{value:,.0f} units<extra></extra>")
        tr.update_layout(coloraxis_showscale=False, hovermode="closest", title="Treemap: click a store to drill in")
        show(tr, 420)
    with b:
        picks = st.multiselect("Compare stores (7-day average)", stores, default=stores[:3], key="cmp_stores")
        cmp = go.Figure()
        for i, sid in enumerate(picks):
            ss = du.daily_series(fc, "forecast", sid, None)
            cmp.add_scatter(x=ss.index, y=ss.rolling(7, center=True, min_periods=4).mean(), mode="lines",
                            name=f"Store {sid}", line=dict(width=2.4, color=COLORWAY[i % len(COLORWAY)]))
        cmp.update_layout(title="Store comparison")
        show(cmp, 365)

    a, b = st.columns(2)
    with a:
        st_tot = tbl.groupby("store")["total"].sum().reindex(stores)
        share = 100 * st_tot / st_tot.sum()
        f3 = go.Figure(go.Bar(x=[f"Store {x}" for x in st_tot.index], y=st_tot.values,
                              marker=dict(color=st_tot.values, colorscale=[[0, VIOLET], [1, PINK]]),
                              text=[f"{v:.1f}%" for v in share], textposition="outside",
                              hovertemplate="%{x}: %{y:,.0f} units<extra></extra>"))
        f3.update_layout(title="Total forecast by store (share of total)", hovermode="closest")
        f3.update_yaxes(range=[0, st_tot.values.max() * 1.15])
        show(f3, 360)
    with b:
        scope = st.selectbox("Rank items for", ["All stores"] + stores, key="rank_scope")
        sub = tbl if scope == "All stores" else tbl[tbl["store"] == scope]
        it = sub.groupby("item")["total"].sum().sort_values(ascending=False)
        show_df = pd.concat([it.head(8), it.tail(8)])
        f4 = go.Figure(go.Bar(x=show_df.values[::-1], y=[f"Item {i}" for i in show_df.index][::-1], orientation="h",
                              marker_color=([PINK] * 8 + [VIOLET] * 8)[::-1],
                              hovertemplate="%{y}: %{x:,.0f} units<extra></extra>"))
        f4.update_layout(title="Top 8 (pink) and bottom 8 (violet) items", hovermode="closest")
        show(f4, 330)

    section("Store-item leaderboard", "Sort any column; bars show total units")
    st.dataframe(
        tbl.rename(columns={"total": "total_units", "peak": "peak_day_units"}),
        width="stretch", hide_index=True, height=380,
        column_config={
            "store": st.column_config.NumberColumn("Store", format="%d"),
            "item": st.column_config.NumberColumn("Item", format="%d"),
            "total_units": st.column_config.ProgressColumn("Total units (90d)", format="%.0f",
                                                           min_value=0, max_value=float(tbl["total"].max())),
            "avg_per_day": st.column_config.NumberColumn("Avg / day", format="%.1f"),
            "peak_day_units": st.column_config.NumberColumn("Peak day", format="%.0f"),
        })

# --------------------------------------------------------------------------- #
# Tab 3: Model performance
# --------------------------------------------------------------------------- #
with tab_model:
    section("How good is the model?",
            "Honest test: the last 90 days of history (Oct 3 to Dec 31, 2017) were never used for training or tuning")
    if results is None:
        st.info("results/results.csv not found.")
    else:
        imp = du.improvement_vs_best_baseline(results)
        if imp:
            kpi_row([
                kpi("LightGBM SMAPE", f"{imp['model_smape']:.2f}", "store-item daily level", VIOLET),
                kpi("Best baseline SMAPE", f"{imp['baseline_smape']:.2f}", imp["baseline_name"], ORANGE),
                kpi("Improvement", f"{imp['smape_gain_pct']:.1f}%", "lower SMAPE than best baseline", TEAL),
                kpi("Total-sales WAPE", f"{imp['model_total_wape']:.2f}%",
                    f"best baseline: {imp['baseline_total_wape']:.2f}%", PINK),
            ])
        st.markdown(leaderboard_html(results), unsafe_allow_html=True)
        st.caption("Teal = best value in each column. SMAPE is the competition metric. "
                   "Holt-Winters ETS is evaluated on total sales only.")

        ordered = results.assign(_k=results["SMAPE"].fillna(1e9)).sort_values("_k")
        c1, c2 = st.columns(2)
        with c1:
            r1 = ordered.dropna(subset=["SMAPE"]).iloc[::-1]
            f1 = go.Figure(go.Bar(x=r1["SMAPE"], y=r1["Model"], orientation="h", text=[f"{v:.2f}" for v in r1["SMAPE"]],
                                  textposition="outside", marker_color=[VIOLET] * (len(r1) - 1) + [TEAL],
                                  hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
            f1.update_layout(title="Store-item daily SMAPE (lower is better)", hovermode="closest")
            f1.update_xaxes(range=[0, r1["SMAPE"].max() * 1.2])
            show(f1, 340)
        with c2:
            r2 = results.sort_values("Total-level WAPE %", ascending=False)
            f2 = go.Figure(go.Bar(x=r2["Total-level WAPE %"], y=r2["Model"], orientation="h",
                                  text=[f"{v:.2f}%" for v in r2["Total-level WAPE %"]], textposition="outside",
                                  marker_color=[PINK] * (len(r2) - 1) + [TEAL],
                                  hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
            f2.update_layout(title="Total sales WAPE % (lower is better)", hovermode="closest")
            f2.update_xaxes(range=[0, r2["Total-level WAPE %"].max() * 1.2])
            show(f2, 340)

    section("Test-window diagnostics")
    g1, g2, g3 = st.tabs(["Forecast vs actual", "Error breakdown", "Best, median, worst series"])
    for tab, img, cap in [(g1, "forecast_total.png", "Total daily sales: forecast vs actual"),
                          (g2, "diagnostics.png", "Error by horizon, weekday and store"),
                          (g3, "forecast_series.png", "Best, median and worst store-item series")]:
        with tab:
            if os.path.exists(P("images", img)):
                st.image(P("images", img), caption=cap)

# --------------------------------------------------------------------------- #
# Tab 4: How it works
# --------------------------------------------------------------------------- #
with tab_how:
    section("The pipeline", "From raw daily sales to a 90-day forecast")
    st.markdown("""
<div class="steps">
  <div class="step"><div class="n">1</div><div class="t">Clean panel</div>
    <div class="d">913,000 daily rows reshaped into 500 store-item series with no missing days.</div></div>
  <div class="step"><div class="n">2</div><div class="t">Leak-free features</div>
    <div class="d">Lags, rolling means and year-ago patterns, all at least 91 days old, so no future data can leak in.</div></div>
  <div class="step"><div class="n">3</div><div class="t">One global LightGBM</div>
    <div class="d">A single model learns from all series. Rounds are chosen on a validation window.</div></div>
  <div class="step"><div class="n">4</div><div class="t">Honest evaluation</div>
    <div class="d">Last 90 days held out as a test. Compared with seasonal-naive baselines and Holt-Winters.</div></div>
</div>
""", unsafe_allow_html=True)

    if importance is not None:
        a, b = st.columns([1.25, 1])
        imp_df = importance.copy()
        imp_df["group"] = imp_df["feature"].map(du.feature_group)
        with a:
            top = imp_df.sort_values("gain %", ascending=False).head(15).iloc[::-1]
            fi = go.Figure(go.Bar(x=top["gain %"], y=top["feature"], orientation="h",
                                  marker_color=[GROUP_COLORS[g] for g in top["group"]],
                                  text=[f"{v:.1f}%" for v in top["gain %"]], textposition="outside",
                                  customdata=top["group"], hovertemplate="%{y} (%{customdata}): %{x:.1f}%<extra></extra>"))
            fi.update_layout(title="Top 15 features (share of total gain)", hovermode="closest")
            fi.update_xaxes(range=[0, top["gain %"].max() * 1.18])
            show(fi, 470)
        with b:
            grp = imp_df.groupby("group")["gain %"].sum().sort_values(ascending=False)
            dn = go.Figure(go.Pie(labels=grp.index, values=grp.values, hole=0.62, sort=False,
                                  marker=dict(colors=[GROUP_COLORS[g] for g in grp.index],
                                              line=dict(color="#0a0f1c", width=2)),
                                  textinfo="percent", hovertemplate="%{label}: %{value:.1f}%<extra></extra>"))
            dn.update_layout(title="Importance by feature group", hovermode="closest",
                             legend=dict(orientation="v", x=1.0, y=0.5))
            show(dn, 470)

    st.markdown(
        "**Reading the features:** `ya_5` and `lag_364/357/371` are sales on the same weekday one year earlier, "
        "so yearly seasonality dominates. `rm_*` and `growth` capture the recent level and year-over-year trend, "
        "`sdow_*` are same-weekday averages of recent weeks, and calendar features add weekday and month effects.")

    section("Data exploration")
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
            "- The what-if slider is a scenario tool, not a model output.\n"
            "- Metrics here are not comparable with the Kaggle leaderboard (different test period).")

st.markdown('<div class="foot">Sales Forecast Studio · LightGBM · Streamlit · Plotly · Built by Deepak</div>',
            unsafe_allow_html=True)
