"""
app.py
------
FinTech Stock Prediction Dashboard  –  Streamlit Application

Launch
------
    streamlit run app.py

Features
--------
  • Stock selector (populated from yahoo-stocks-data.xlsx)
  • Interactive historical price chart (Plotly)
  • One-click model training (XGBoost + LSTM)
  • Prediction overlay on the test period
  • Next-day price forecast with confidence ribbon
  • Metrics comparison panel (RMSE / MAE / MAPE)
"""

import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Make 'src' importable from any working directory ──────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_preprocessing import (
    prepare_data,
    load_cached_data,
    get_symbols,
    load_stock_list,
    download_historical_data,
    create_sequences,
    WINDOW_SIZE,
)
from src.baseline_model import (
    train_xgboost,
    predict_xgboost,
    load_xgboost_model,
    xgboost_model_exists,
)
from src.lstm_model import (
    train_lstm,
    predict_lstm,
    load_lstm_model,
    lstm_model_exists,
)
from src.evaluation import (
    inverse_transform_1d,
    calculate_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinTech Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
        color: #f1f5f9;
    }
    .metric-label  { font-size: 13px; color: #94a3b8; margin-bottom: 4px; }
    .metric-value  { font-size: 28px; font-weight: 700; }
    .metric-delta  { font-size: 12px; margin-top: 4px; }
    .winner-badge  {
        display: inline-block;
        background: #22c55e;
        color: white;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 6px;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-money-256.png/240px-24701-money-256.png",
        width=60,
    )
    st.title("📈 Stock Predictor")
    st.caption("AI/ML FinTech · XGBoost + LSTM")
    st.divider()

    # Stock selector
    xlsx_path = os.path.join(_ROOT, "data", "yahoo-stocks-data.xlsx")
    try:
        stock_df  = load_stock_list(xlsx_path)
        # Build a display list: "AAPL – Apple Inc."
        options   = [
            f"{row.symbol} – {row['name']}"
            for _, row in stock_df.iterrows()
        ]
        # Default to AAPL
        default_idx = next(
            (i for i, s in enumerate(options) if s.startswith("AAPL")), 0
        )
        selected  = st.selectbox(
            "Select Stock", options, index=default_idx,
            help="Stocks from yahoo-stocks-data.xlsx",
        )
        SYMBOL = selected.split(" – ")[0]
    except Exception:
        SYMBOL   = st.text_input("Ticker symbol", value="AAPL").upper()
        stock_df = None

    st.divider()

    # Data period
    period = st.selectbox(
        "Historical period",
        ["1y", "2y", "3y", "5y", "10y"],
        index=3,
        help="How far back to fetch data from Yahoo Finance",
    )

    st.divider()

    # Training button
    train_btn = st.button(
        "🚀 Train / Retrain Models",
        use_container_width=True,
        type="primary",
    )
    st.caption("First run downloads data & trains both models (~2–5 min).")

    st.divider()
    st.markdown(
        "<small>Data source: Yahoo Finance via **yfinance**</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────
STATE_KEY = f"results_{SYMBOL}"


def _get_state() -> dict | None:
    return st.session_state.get(STATE_KEY)


def _set_state(d: dict) -> None:
    st.session_state[STATE_KEY] = d


# ─────────────────────────────────────────────────────────────────────────────
# Training / loading logic
# ─────────────────────────────────────────────────────────────────────────────
def train_and_store(symbol: str, period: str, retrain: bool = False) -> None:
    """Run the full pipeline and store results in session state."""
    progress = st.progress(0, text="Preparing…")

    # Step 1 – data
    progress.progress(5, "Downloading & preprocessing data…")
    cached = load_cached_data(symbol) if not retrain else None
    if cached is not None:
        data_dict = cached
    else:
        data_dict = prepare_data(symbol=symbol, period=period)
    progress.progress(30, "Data ready ✓")

    # Step 2 – XGBoost
    progress.progress(35, "Training XGBoost…")
    if not retrain and xgboost_model_exists(symbol):
        xgb_model = load_xgboost_model(symbol)
    else:
        xgb_model = train_xgboost(
            data_dict["X_train"], data_dict["y_train"], symbol=symbol
        )
    xgb_preds = predict_xgboost(xgb_model, data_dict["X_test"])
    progress.progress(60, "XGBoost done ✓")

    # Step 3 – LSTM
    progress.progress(65, "Training LSTM (early stopping)…")
    if not retrain and lstm_model_exists(symbol):
        lstm_model = load_lstm_model(symbol)
    else:
        lstm_model, _ = train_lstm(
            data_dict["X_train"], data_dict["y_train"], symbol=symbol
        )
    lstm_preds = predict_lstm(lstm_model, data_dict["X_test"])
    progress.progress(90, "LSTM done ✓")

    # Step 4 – Inverse-transform & metrics
    scaler     = data_dict["scaler"]
    n_feat     = len(data_dict["feature_cols"])
    y_actual   = inverse_transform_1d(data_dict["y_test"], scaler, n_feat)
    y_xgb      = inverse_transform_1d(xgb_preds,           scaler, n_feat)
    y_lstm     = inverse_transform_1d(lstm_preds,           scaler, n_feat)

    metrics_xgb  = calculate_metrics(y_actual, y_xgb,  "XGBoost")
    metrics_lstm = calculate_metrics(y_actual, y_lstm,  "LSTM")

    # Test dates
    raw_df     = data_dict["raw_df"]
    win        = data_dict["window_size"]
    ts         = data_dict["train_size"]
    start_idx  = ts + win
    test_dates = raw_df.index[start_idx : start_idx + len(y_actual)]

    # Next-day forecast
    last_window = data_dict["scaled_df"].values[-win:]           # (60, n_feat)
    last_3d     = last_window[np.newaxis, :, :]                  # (1, 60, n_feat)
    nxt_xgb_sc  = predict_xgboost(xgb_model,  last_3d)[0]
    nxt_lstm_sc = predict_lstm(lstm_model,     last_3d)[0]

    dummy_xgb  = np.zeros((1, n_feat)); dummy_xgb[0, -1]  = nxt_xgb_sc
    dummy_lstm = np.zeros((1, n_feat)); dummy_lstm[0, -1] = nxt_lstm_sc
    nxt_xgb  = float(scaler.inverse_transform(dummy_xgb)[0, -1])
    nxt_lstm = float(scaler.inverse_transform(dummy_lstm)[0, -1])

    progress.progress(100, "Done ✓")

    _set_state({
        "data_dict":    data_dict,
        "y_actual":     y_actual,
        "y_xgb":        y_xgb,
        "y_lstm":       y_lstm,
        "metrics_xgb":  metrics_xgb,
        "metrics_lstm": metrics_lstm,
        "test_dates":   test_dates,
        "nxt_xgb":      nxt_xgb,
        "nxt_lstm":     nxt_lstm,
        "symbol":       symbol,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Handle train button
# ─────────────────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Running pipeline — this may take a few minutes…"):
        try:
            train_and_store(SYMBOL, period, retrain=True)
            st.success("Models trained successfully!")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.stop()
else:
    # Auto-load if models exist and no session state yet
    if _get_state() is None:
        if xgboost_model_exists(SYMBOL) and lstm_model_exists(SYMBOL):
            with st.spinner(f"Loading saved models for {SYMBOL}…"):
                try:
                    train_and_store(SYMBOL, period, retrain=False)
                except Exception:
                    pass   # silently skip; user will press Train


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.title(f"📊  {SYMBOL}  ·  AI Stock Price Prediction")

state = _get_state()

# ── 0. Raw historical price chart (always shown) ──────────────────────────
st.subheader("Historical Price Data")
try:
    with st.spinner("Fetching historical data…"):
        raw_df = download_historical_data(SYMBOL, period=period)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Candlestick(
        x=raw_df.index,
        open=raw_df["Open"],
        high=raw_df["High"],
        low=raw_df["Low"],
        close=raw_df["Close"],
        name=SYMBOL,
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ))
    fig_hist.update_layout(
        title=f"{SYMBOL}  Candlestick  ({period})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=420,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Quick KPIs
    latest = raw_df["Close"].iloc[-1]
    prev   = raw_df["Close"].iloc[-2]
    chg    = latest - prev
    chg_pct = chg / prev * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close",   f"${latest:.2f}")
    col2.metric("Daily Change",   f"${chg:+.2f}", f"{chg_pct:+.2f}%")
    col3.metric("52-wk High",
                f"${raw_df['High'].tail(252).max():.2f}")
    col4.metric("52-wk Low",
                f"${raw_df['Low'].tail(252).min():.2f}")

except Exception as exc:
    st.warning(f"Could not fetch live data: {exc}")

st.divider()

# ── 1. Prediction chart ───────────────────────────────────────────────────
if state is None:
    st.info(
        "👈  Press **Train / Retrain Models** in the sidebar to run the "
        "XGBoost + LSTM pipeline on this stock.",
        icon="🤖",
    )
    st.stop()

# Unpack state
y_actual     = state["y_actual"]
y_xgb        = state["y_xgb"]
y_lstm       = state["y_lstm"]
metrics_xgb  = state["metrics_xgb"]
metrics_lstm = state["metrics_lstm"]
test_dates   = state["test_dates"]
nxt_xgb      = state["nxt_xgb"]
nxt_lstm     = state["nxt_lstm"]

st.subheader("Model Predictions  (Test Period)")

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(
    x=test_dates, y=y_actual,
    mode="lines", name="Actual",
    line=dict(color="#60a5fa", width=2.5),
))
fig_pred.add_trace(go.Scatter(
    x=test_dates, y=y_xgb,
    mode="lines", name="XGBoost",
    line=dict(color="#fb923c", width=1.8, dash="dash"),
))
fig_pred.add_trace(go.Scatter(
    x=test_dates, y=y_lstm,
    mode="lines", name="LSTM",
    line=dict(color="#4ade80", width=1.8, dash="dot"),
))
fig_pred.update_layout(
    template="plotly_dark",
    height=420,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend=dict(orientation="h", y=1.05),
    hovermode="x unified",
)
st.plotly_chart(fig_pred, use_container_width=True)

# ── Absolute error chart ──────────────────────────────────────────────────
with st.expander("📉  Absolute Prediction Errors"):
    err_xgb  = np.abs(y_actual - y_xgb)
    err_lstm = np.abs(y_actual - y_lstm)

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=test_dates, y=err_xgb,
        fill="tozeroy", name="|XGBoost error|",
        line=dict(color="#fb923c"), fillcolor="rgba(251,146,60,0.25)",
    ))
    fig_err.add_trace(go.Scatter(
        x=test_dates, y=err_lstm,
        fill="tozeroy", name="|LSTM error|",
        line=dict(color="#4ade80"), fillcolor="rgba(74,222,128,0.25)",
    ))
    fig_err.update_layout(
        template="plotly_dark", height=280,
        xaxis_title="Date", yaxis_title="Error (USD)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_err, use_container_width=True)

st.divider()

# ── 2. Metrics comparison ─────────────────────────────────────────────────
st.subheader("Model Performance Comparison")

# Helpers — built outside f-strings to avoid Streamlit treating bare "$" as LaTeX
def _badge(is_winner: bool) -> str:
    return " <span class='winner-badge'>&#10003; Best</span>" if is_winner else ""

def _metric_card_html(
    label: str,
    xgb_val: float,
    lstm_val: float,
    prefix: str = "&#36;",   # HTML entity for "$" — avoids Streamlit LaTeX mode
    suffix: str = "",
    fmt: str = ".4f",
) -> str:
    xgb_str  = prefix + format(xgb_val, fmt) + suffix
    lstm_str = prefix + format(lstm_val, fmt) + suffix
    xgb_wins  = xgb_val < lstm_val
    lstm_wins = lstm_val < xgb_val
    return (
        '<div class="metric-card">'
        '<div class="metric-label">' + label + '</div>'
        '<div class="metric-delta">XGBoost: ' + xgb_str  + _badge(xgb_wins)  + '</div>'
        '<div class="metric-delta">LSTM:&nbsp;&nbsp;&nbsp;&nbsp;'  + lstm_str + _badge(lstm_wins) + '</div>'
        '</div>'
    )

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        _metric_card_html("RMSE (USD)", metrics_xgb["RMSE"], metrics_lstm["RMSE"]),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        _metric_card_html("MAE (USD)", metrics_xgb["MAE"], metrics_lstm["MAE"]),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        _metric_card_html(
            "MAPE (%)", metrics_xgb["MAPE"], metrics_lstm["MAPE"],
            prefix="", suffix="%", fmt=".3f",
        ),
        unsafe_allow_html=True,
    )

# Bar chart comparison
st.markdown("<br>", unsafe_allow_html=True)
metric_labels = ["RMSE", "MAE", "MAPE"]
xgb_vals  = [metrics_xgb[m]  for m in metric_labels]
lstm_vals = [metrics_lstm[m] for m in metric_labels]

fig_bar = go.Figure(data=[
    go.Bar(name="XGBoost", x=metric_labels, y=xgb_vals,
           marker_color="#fb923c", text=[f"{v:.4f}" for v in xgb_vals],
           textposition="outside"),
    go.Bar(name="LSTM",    x=metric_labels, y=lstm_vals,
           marker_color="#4ade80", text=[f"{v:.4f}" for v in lstm_vals],
           textposition="outside"),
])
fig_bar.update_layout(
    barmode="group",
    template="plotly_dark",
    height=360,
    yaxis_title="Score",
    legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── 3. Next-day forecast ──────────────────────────────────────────────────
st.subheader("🔮  Next-Day Price Forecast")

last_actual = float(y_actual[-1])
avg_forecast = (nxt_xgb + nxt_lstm) / 2

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Last Actual Close",    f"${last_actual:.2f}")
col_b.metric("XGBoost Forecast",     f"${nxt_xgb:.2f}",
             f"{(nxt_xgb - last_actual):+.2f} ({(nxt_xgb - last_actual)/last_actual*100:+.2f}%)")
col_c.metric("LSTM Forecast",        f"${nxt_lstm:.2f}",
             f"{(nxt_lstm - last_actual):+.2f} ({(nxt_lstm - last_actual)/last_actual*100:+.2f}%)")
col_d.metric("Ensemble Average",     f"${avg_forecast:.2f}",
             f"{(avg_forecast - last_actual):+.2f} ({(avg_forecast - last_actual)/last_actual*100:+.2f}%)")

st.divider()

# ── 4. Raw data table ─────────────────────────────────────────────────────
with st.expander("📋  View Raw Test-Period Data"):
    display_df = pd.DataFrame({
        "Date":         test_dates,
        "Actual ($)":   np.round(y_actual, 4),
        "XGBoost ($)":  np.round(y_xgb, 4),
        "LSTM ($)":     np.round(y_lstm, 4),
        "XGB Error ($)": np.round(np.abs(y_actual - y_xgb), 4),
        "LSTM Error ($)":np.round(np.abs(y_actual - y_lstm), 4),
    }).set_index("Date")
    st.dataframe(display_df, use_container_width=True)

# ── 5. Stock universe from xlsx ───────────────────────────────────────────
if stock_df is not None:
    with st.expander("🌐  Yahoo Stocks Universe (from xlsx)"):
        st.dataframe(
            stock_df.rename(columns={
                "symbol":       "Ticker",
                "name":         "Company",
                "price_usd":    "Price (USD)",
                "change":       "Change",
                "change_pct":   "Chg %",
                "volume_M":     "Volume (M)",
                "market_cap_B": "Mkt Cap (B)",
                "pe_ratio":     "P/E",
            }),
            use_container_width=True,
            height=300,
        )

st.caption("FinTech AI Project · XGBoost + LSTM Stock Price Prediction · 2024")
