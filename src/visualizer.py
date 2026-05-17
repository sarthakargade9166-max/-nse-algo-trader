"""
src/visualizer.py
==================
All chart-building functions for the Streamlit dashboard.
Uses Plotly for interactive, professional-looking charts.

Beginner Tip:
    Plotly charts are "interactive" — users can hover, zoom, and pan.
    This is much better than Matplotlib for web apps.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Shared dark theme colors
DARK_BG    = "#0a0e1a"
PANEL_BG   = "#1a2035"
GREEN      = "#48bb78"
RED        = "#fc8181"
YELLOW     = "#f6ad55"
BLUE       = "#90cdf4"
PURPLE     = "#b794f4"
GRID_COLOR = "#2d3748"
TEXT_COLOR = "#e2e8f0"


def plot_price_with_signals(
    df: pd.DataFrame,
    stock_name: str,
    y_pred: np.ndarray,
    X_test: pd.DataFrame
) -> go.Figure:
    """
    Plots candlestick price chart with BUY/SELL signal markers overlaid.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"{stock_name} — Price & Signals", "Volume"]
    )

    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x     = df.index,
        open  = df["Open"],
        high  = df["High"],
        low   = df["Low"],
        close = df["Close"],
        name  = "Price",
        increasing_line_color = GREEN,
        decreasing_line_color = RED,
    ), row=1, col=1)

    # Moving Averages
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_20"], name="SMA 20",
            line=dict(color=BLUE, width=1.5, dash="dot"),
            opacity=0.8
        ), row=1, col=1)

    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_50"], name="SMA 50",
            line=dict(color=PURPLE, width=1.5),
            opacity=0.8
        ), row=1, col=1)

    # Buy/Sell Signal Markers
    test_dates  = X_test.index
    pred_series = pd.Series(y_pred, index=test_dates[:len(y_pred)])

    buy_dates  = pred_series[pred_series ==  1].index
    sell_dates = pred_series[pred_series == -1].index

    if len(buy_dates) > 0:
        fig.add_trace(go.Scatter(
            x    = buy_dates,
            y    = df.loc[df.index.isin(buy_dates), "Low"] * 0.99,
            name = "BUY Signal",
            mode = "markers",
            marker = dict(symbol="triangle-up", size=12, color=GREEN,
                          line=dict(width=1, color="white")),
        ), row=1, col=1)

    if len(sell_dates) > 0:
        fig.add_trace(go.Scatter(
            x    = sell_dates,
            y    = df.loc[df.index.isin(sell_dates), "High"] * 1.01,
            name = "SELL Signal",
            mode = "markers",
            marker = dict(symbol="triangle-down", size=12, color=RED,
                          line=dict(width=1, color="white")),
        ), row=1, col=1)

    # Volume Bar Chart
    vol_colors = [GREEN if row["Close"] >= row["Open"] else RED
                  for _, row in df.iterrows()]

    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=vol_colors, opacity=0.7
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor = DARK_BG,
        plot_bgcolor  = PANEL_BG,
        font          = dict(color=TEXT_COLOR, family="DM Sans"),
        legend        = dict(bgcolor="#1a2035", bordercolor="#2d3748", borderwidth=1),
        margin        = dict(l=40, r=20, t=50, b=40),
        height        = 600,
        showlegend    = True,
        xaxis_rangeslider_visible = False,
        title         = dict(
            text=f"<b>{stock_name}</b> Price Chart with ML Signals",
            font=dict(size=14)
        ),
    )

    fig.update_xaxes(gridcolor=GRID_COLOR, showgrid=True)
    fig.update_yaxes(gridcolor=GRID_COLOR, showgrid=True)

    return fig


def plot_regime(df: pd.DataFrame) -> go.Figure:
    """
    Plots a colored background chart showing market regimes over time.
    Green = Bullish, Red = Bearish, Yellow = Sideways.
    """
    fig = go.Figure()

    # Line of closing price
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close Price",
        line=dict(color=BLUE, width=2),
    ))

    # Shade background by regime
    regime_colors = {
        "Bullish":  "rgba(72,187,120,0.15)",
        "Bearish":  "rgba(252,129,129,0.15)",
        "Sideways": "rgba(246,173,85,0.12)"
    }

    prev_regime = None
    start_date  = df.index[0]

    for date, row in df.iterrows():
        if row["Regime"] != prev_regime:
            if prev_regime is not None:
                fig.add_vrect(
                    x0=start_date, x1=date,
                    fillcolor=regime_colors.get(prev_regime, "rgba(0,0,0,0)"),
                    layer="below", line_width=0,
                )
            start_date  = date
            prev_regime = row["Regime"]

    # Last segment
    if prev_regime:
        fig.add_vrect(
            x0=start_date, x1=df.index[-1],
            fillcolor=regime_colors.get(prev_regime, "rgba(0,0,0,0)"),
            layer="below", line_width=0,
        )

    # ADX line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ADX"],
        name="ADX (Trend Strength)",
        line=dict(color=YELLOW, width=1.5, dash="dot"),
        yaxis="y2", opacity=0.8
    ))

    # ADX=25 threshold line
    fig.add_hline(
        y=25, line_dash="dash", line_color=YELLOW,
        annotation_text="ADX=25 (Trend Threshold)",
        opacity=0.5, yref="y2"
    )

    fig.update_layout(
        paper_bgcolor = DARK_BG,
        plot_bgcolor  = PANEL_BG,
        font          = dict(color=TEXT_COLOR, family="DM Sans"),
        legend        = dict(bgcolor="#1a2035", bordercolor="#2d3748", borderwidth=1),
        margin        = dict(l=40, r=20, t=50, b=40),
        height        = 380,
        xaxis         = dict(gridcolor=GRID_COLOR, showgrid=True),
        yaxis         = dict(gridcolor=GRID_COLOR, showgrid=True, title="Price"),
        yaxis2        = dict(
            title="ADX",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, 60],
            tickfont=dict(color=TEXT_COLOR)
        ),
        title=dict(
            text="<b>Market Regime</b> — Bullish / Bearish / Sideways",
            font=dict(size=13)
        ),
    )

    return fig


def plot_feature_importance(model) -> go.Figure:
    """
    Horizontal bar chart of XGBoost feature importance scores.
    Shows which indicators the model relies on most.
    """
    importances = model.feature_importances_
    features    = model.get_booster().feature_names

    if features is None:
        features = [f"f{i}" for i in range(len(importances))]

    # Sort by importance — top 15
    sorted_idx  = np.argsort(importances)[-15:]
    sorted_feat = [features[i] for i in sorted_idx]
    sorted_imp  = [importances[i] for i in sorted_idx]

    max_imp = max(sorted_imp) if max(sorted_imp) > 0 else 1
    colors  = [
        f"rgba(183,148,244,{0.4 + 0.6 * (imp / max_imp)})"
        for imp in sorted_imp
    ]

    fig = go.Figure(go.Bar(
        x            = sorted_imp,
        y            = sorted_feat,
        orientation  = "h",
        marker_color = colors,
        text         = [f"{v:.3f}" for v in sorted_imp],
        textposition = "outside",
    ))

    fig.update_layout(
        paper_bgcolor = DARK_BG,
        plot_bgcolor  = PANEL_BG,
        font          = dict(color=TEXT_COLOR, family="DM Sans"),
        margin        = dict(l=40, r=60, t=50, b=40),
        height        = 400,
        xaxis         = dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            title="Importance Score"
        ),
        yaxis         = dict(
            gridcolor=GRID_COLOR,
            showgrid=False
        ),
        title=dict(
            text="<b>Feature Importance</b> — Top 15 Predictors",
            font=dict(size=13)
        ),
    )

    return fig