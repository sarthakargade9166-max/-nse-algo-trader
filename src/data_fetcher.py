"""
src/data_fetcher.py
====================
Fetches historical OHLCV data for NSE stocks using yfinance.
Beginner Note: OHLCV = Open, High, Low, Close, Volume — the 5 core price columns
               that every trading system is built on.
"""

import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600)  # Cache data for 1 hour to avoid repeated API calls
def fetch_nse_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Downloads daily OHLCV data for an NSE stock from Yahoo Finance.

    Parameters:
        ticker (str): Yahoo Finance ticker symbol (e.g., "RELIANCE.NS")
        start  (str): Start date in "YYYY-MM-DD" format
        end    (str): End date in "YYYY-MM-DD" format

    Returns:
        pd.DataFrame: DataFrame with Date index and OHLCV columns,
                      or None if the download fails.
    """
    try:
        # Download raw data from Yahoo Finance
        # yfinance uses NSE tickers with ".NS" suffix (e.g., RELIANCE.NS)
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

        if raw.empty:
            return None

        # Flatten MultiIndex columns if present (happens with some yfinance versions)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Keep only the 5 core OHLCV columns
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Drop any rows where all values are NaN
        df.dropna(how="all", inplace=True)

        # Ensure the index is a proper DatetimeIndex
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        return df

    except Exception as e:
        # Print error for debugging; in production you'd log this
        print(f"[ERROR] fetch_nse_stock_data: {e}")
        return None