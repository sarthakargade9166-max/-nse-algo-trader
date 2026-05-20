"""
NSE Algorithmic Trading & Market Regime Filter
================================================
Main Streamlit Application Entry Point
Author: YOUR_NAME
GitHub: YOUR_GITHUB_REPO_LINK
Live App: YOUR_DEPLOYED_APP_LINK
"""

import streamlit as st
st.set_page_config(
    page_title="NSE Algo Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from src.data_fetcher import fetch_nse_stock_data
from src.feature_engineer import engineer_features
from src.regime_filter import classify_market_regime
from src.model_trainer import train_xgboost_model, load_model, predict_signal
from src.database import init_db, log_prediction, get_trade_log
from src.visualizer import plot_price_with_signals, plot_regime, plot_feature_importance


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}
.stApp { background-color: #0a0e1a; }
.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #141828 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; }
.buy-signal  { color: #48bb78; }
.sell-signal { color: #fc8181; }
.hold-signal { color: #f6ad55; }
.regime-bull { color: #48bb78; font-weight: 700; }
.regime-bear { color: #fc8181; font-weight: 700; }
.regime-side { color: #f6ad55; font-weight: 700; }
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    color: #718096;
    text-transform: uppercase;
    margin-bottom: 8px;
}
div[data-testid="stSidebarContent"] {
    background-color: #0d1220;
    border-right: 1px solid #2d3748;
}
</style>
""", unsafe_allow_html=True)



with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Stock selection
    STOCK_OPTIONS = {
        "Reliance Industries": "RELIANCE.NS",
        "TCS":                 "TCS.NS",
        "HDFC Bank":           "HDFCBANK.NS",
        "Infosys":             "INFY.NS",
        "ICICI Bank":          "ICICIBANK.NS",
        "Wipro":               "WIPRO.NS",
        "Bajaj Finance":       "BAJFINANCE.NS",
        "Maruti Suzuki":       "MARUTI.NS",
    }
    selected_stock_name = st.selectbox("Select NSE Stock", list(STOCK_OPTIONS.keys()))
    ticker = STOCK_OPTIONS[selected_stock_name]

    # Date range
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=365 * 3)  # 3 years of data
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start Date", value=start_date)
    with col2:
        end = st.date_input("End Date", value=end_date)

    # Model params
    st.markdown("---")
    st.markdown("### 🤖 Model Parameters")
    n_estimators   = st.slider("XGBoost Trees",        50, 500, 200, 50)
    max_depth      = st.slider("Max Tree Depth",         2,   8,   4,   1)
    test_size      = st.slider("Test Set Size (%)",     10,  40,  20,   5) / 100
    prediction_horizon = st.selectbox("Predict N Days Ahead", [1, 3, 5], index=0)

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")


#header
st.markdown("""
<div style='padding: 10px 0 20px 0;'>
  <span style='font-family: Space Mono, monospace; font-size:1.8rem; font-weight:700; color:#e2e8f0;'>
    📈 NSE ALGO TRADER
  </span>
  <span style='font-family: DM Sans; font-size:0.95rem; color:#718096; margin-left:16px;'>
    Market Regime Filter · XGBoost Signal Engine · NSE Stocks
  </span>
</div>
""", unsafe_allow_html=True)

# Initialize database on startup
init_db()

# ─────────────────────────────────────────────
# MAIN ANALYSIS PIPELINE
# ─────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching data for {selected_stock_name}..."):
        raw_df = fetch_nse_stock_data(ticker, str(start), str(end))

    if raw_df is None or raw_df.empty:
        st.error("❌ Could not fetch data. Check ticker or internet connection.")
        st.stop()

    with st.spinner("Engineering features..."):
        featured_df = engineer_features(raw_df)

    with st.spinner("Classifying market regimes..."):
        regime_df = classify_market_regime(featured_df)

    with st.spinner("Training XGBoost model..."):
        model, metrics, X_test, y_test, y_pred = train_xgboost_model(
            regime_df,
            n_estimators=n_estimators,
            max_depth=max_depth,
            test_size=test_size,
            horizon=prediction_horizon
        )

    # Latest signal
    latest_signal_val = y_pred[-1] if len(y_pred) > 0 else 0
    latest_signal_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
    latest_signal     = latest_signal_map.get(latest_signal_val, "HOLD")
    latest_regime     = regime_df["Regime"].iloc[-1]
    latest_price      = regime_df["Close"].iloc[-1]
    latest_date       = regime_df.index[-1].strftime("%d %b %Y")

    # Log to DB
    log_prediction(
        stock=ticker,
        date=latest_date,
        price=float(latest_price),
        signal=latest_signal,
        regime=latest_regime,
        accuracy=metrics["accuracy"]
    )

    # ─── KPI METRICS ROW ───
    st.markdown("### 📊 Live Signal Dashboard")
    m1, m2, m3, m4, m5 = st.columns(5)

    signal_class = {"BUY": "buy-signal", "SELL": "sell-signal", "HOLD": "hold-signal"}[latest_signal]
    regime_class = {"Bullish": "regime-bull", "Bearish": "regime-bear", "Sideways": "regime-side"}.get(latest_regime, "")

    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="section-header">Latest Signal</div>
            <div class="metric-value {signal_class}">{latest_signal}</div>
            <div style="color:#718096;font-size:0.8rem;">{latest_date}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="section-header">Market Regime</div>
            <div class="metric-value {regime_class}" style="font-size:1.4rem;">{latest_regime}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="section-header">Last Close (₹)</div>
            <div class="metric-value" style="color:#90cdf4;">₹{latest_price:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="section-header">Model Accuracy</div>
            <div class="metric-value" style="color:#b794f4;">{metrics['accuracy']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card">
            <div class="section-header">F1-Score</div>
            <div class="metric-value" style="color:#f6ad55;">{metrics['f1']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── PRICE CHART WITH SIGNALS ───
    st.markdown("### 🕯️ Price Chart with Trading Signals")
    fig_price = plot_price_with_signals(regime_df, selected_stock_name, y_pred, X_test)
    st.plotly_chart(fig_price, use_container_width=True)

    # ─── REGIME ans FEATURES ───
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### 🌡️ Market Regime Classification")
        fig_regime = plot_regime(regime_df)
        st.plotly_chart(fig_regime, use_container_width=True)

    with col_right:
        st.markdown("### 🔬 Feature Importance (XGBoost)")
        fig_feat = plot_feature_importance(model)
        st.plotly_chart(fig_feat, use_container_width=True)

    # ─── METRICS DETAIL ───
    st.markdown("### 📋 Model Performance Report")
    report_df = pd.DataFrame({
        "Metric":    ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Value":     [
            f"{metrics['accuracy']*100:.2f}%",
            f"{metrics['precision']*100:.2f}%",
            f"{metrics['recall']*100:.2f}%",
            f"{metrics['f1']*100:.2f}%",
            f"{metrics['roc_auc']*100:.2f}%"
        ],
        "Benchmark": ["≥ 55%", "≥ 50%", "≥ 50%", "≥ 52%", "≥ 55%"],
        "Status":    [
            "✅" if metrics['accuracy']  >= 0.55 else "⚠️",
            "✅" if metrics['precision'] >= 0.50 else "⚠️",
            "✅" if metrics['recall']    >= 0.50 else "⚠️",
            "✅" if metrics['f1']        >= 0.52 else "⚠️",
            "✅" if metrics['roc_auc']   >= 0.55 else "⚠️",
        ]
    })
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    # ─── TRADE LOG ───
    st.markdown("### 🗃️ Trade Signal Log (Database)")
    trade_log = get_trade_log()
    if not trade_log.empty:
        st.dataframe(trade_log.tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("No trade logs yet. Run more analyses to populate.")

    st.markdown("---")
    st.caption(f"Data Source: Yahoo Finance (NSE) · Model: XGBoost · Last Updated: {datetime.now().strftime('%d %b %Y %H:%M IST')}")

else:
  
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px;'>
        <div style='font-size:4rem;'>📈</div>
        <h2 style='color:#e2e8f0; font-family: Space Mono, monospace;'>NSE Algo Trader</h2>
        <p style='color:#718096; font-size:1.1rem; max-width:600px; margin:0 auto;'>
            Select a stock and configure parameters in the sidebar, then click 
            <strong style='color:#48bb78;'>Run Analysis</strong> to generate ML-powered 
            trading signals with market regime filtering.
        </p>
        <br>
        <div style='display:flex; justify-content:center; gap:40px; flex-wrap:wrap; margin-top:20px;'>
            <div style='color:#718096;'>🤖 XGBoost ML Engine</div>
            <div style='color:#718096;'>🌡️ Volatility Regime Filter</div>
            <div style='color:#718096;'>📊 20+ Technical Indicators</div>
            <div style='color:#718096;'>🗃️ SQL Trade Logging</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
