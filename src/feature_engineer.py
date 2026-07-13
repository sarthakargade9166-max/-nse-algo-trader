
import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw OHLCV DataFrame and adds 20+ technical indicator columns.

    Parameters:
        df (pd.DataFrame): Raw OHLCV data with DatetimeIndex

    Returns:
        pd.DataFrame: Original data + all engineered features (NaNs dropped)
    """
    data = df.copy()

    # 1. TREND INDICATORS
 
    data["SMA_20"]  = data["Close"].rolling(window=20).mean()
    data["SMA_50"]  = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()

    # Exponential Moving Averages — give more weight to recent prices
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

    # Price relative to moving averages 
    data["Price_to_SMA20"] = (data["Close"] - data["SMA_20"]) / data["SMA_20"]
    data["Price_to_SMA50"] = (data["Close"] - data["SMA_50"]) / data["SMA_50"]
    data["SMA20_SMA50_cross"] = data["SMA_20"] - data["SMA_50"]  # Golden/Death cross signal

    # 2. MACD
    
    data["MACD"]        = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"]   = data["MACD"] - data["MACD_Signal"]  # Histogram = momentum of MACD

    #  3. RSI 
    
    delta  = data["Close"].diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=13, adjust=False).mean()
    avg_l  = loss.ewm(com=13, adjust=False).mean()
    rs     = avg_g / (avg_l + 1e-10)            # 1e-10 prevents division by zero
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # 4. BOLLINGER BANDS — Volatility Channel 
 
    rolling_std       = data["Close"].rolling(window=20).std()
    data["BB_Upper"]  = data["SMA_20"] + (2 * rolling_std)
    data["BB_Lower"]  = data["SMA_20"] - (2 * rolling_std)
    data["BB_Width"]  = (data["BB_Upper"] - data["BB_Lower"]) / data["SMA_20"]  # Normalized width
    data["BB_Pct"]    = (data["Close"] - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"] + 1e-10)

    # ─── 5. ATR (Average True Range) 
    high_low   = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close  = (data["Low"]  - data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR_14"] = true_range.ewm(com=13, adjust=False).mean()
    data["ATR_Pct"] = data["ATR_14"] / data["Close"]  #

    # ─── 6. VOLUME INDICATORS 
    data["Volume_SMA20"]    = data["Volume"].rolling(window=20).mean()
    data["Volume_Ratio"]    = data["Volume"] / (data["Volume_SMA20"] + 1e-10)  # >1 = high volume day
    data["OBV"]             = (np.sign(data["Close"].diff()) * data["Volume"]).cumsum()  # On-Balance Volume

    # ─── 7. PRICE ACTION / MOMENTUM 
    data["Return_1d"]  = data["Close"].pct_change(1)   # 1-day return
    data["Return_5d"]  = data["Close"].pct_change(5)   # 5-day return (1 week)
    data["Return_20d"] = data["Close"].pct_change(20)  # 20-day return (1 month)

    # Rate of Change (ROC) — % change from N periods ago
    data["ROC_10"] = ((data["Close"] - data["Close"].shift(10)) / data["Close"].shift(10)) * 100

    # Candlestick body size and direction
    data["Candle_Body"] = (data["Close"] - data["Open"]) / data["Open"]  # +ve = bullish, -ve = bearish
    data["High_Low_Pct"] = (data["High"] - data["Low"]) / data["Close"]  # Daily range as % of close

    # ─── 8. STOCHASTIC OSCILLATOR ─────────────────────────────────────────
    # Compares close to the high-low range over N days; overbought/oversold
    low_14  = data["Low"].rolling(window=14).min()
    high_14 = data["High"].rolling(window=14).max()
    data["Stoch_K"] = 100 * (data["Close"] - low_14) / (high_14 - low_14 + 1e-10)
    data["Stoch_D"] = data["Stoch_K"].rolling(window=3).mean()  # Signal line

    # ─── DROP NaN ROWS ────────────────────────────────────────────────────
    # Many indicators need a "warm-up" period (e.g., SMA_200 needs 200 days).
    # We drop all rows that have NaN values from the warm-up period.
    data.dropna(inplace=True)

    return data


def get_feature_columns() -> list[str]:
    """
    Returns the exact list of feature column names used for model training.
    This must stay consistent between training and prediction.
    """
    return [
        "SMA_20", "SMA_50", "SMA_200",
        "EMA_12", "EMA_26",
        "Price_to_SMA20", "Price_to_SMA50", "SMA20_SMA50_cross",
        "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_14",
        "BB_Width", "BB_Pct",
        "ATR_14", "ATR_Pct",
        "Volume_Ratio", "OBV",
        "Return_1d", "Return_5d", "Return_20d",
        "ROC_10",
        "Candle_Body", "High_Low_Pct",
        "Stoch_K", "Stoch_D",
    ]
