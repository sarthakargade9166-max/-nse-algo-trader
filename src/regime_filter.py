import pandas as pd
import numpy as np


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes the Average Directional Index (ADX).
    ADX > 25 indicates a strong trend (either up or down).
    ADX < 20 indicates a weak trend (ranging/sideways market).

    Parameters:
        df     (pd.DataFrame): OHLCV DataFrame
        period (int):          Lookback period (default 14 days)

    Returns:
        pd.Series: ADX values indexed by date
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    # +DM: Positive Directional Movement 
    plus_dm  = high.diff()
    minus_dm = -low.diff()

    # Only keep the positive directional movement when it's greater
    plus_dm[plus_dm  < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm  = plus_dm.where(plus_dm  > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm,  0)

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    # Smoothed ATR and Directional Movements 
    atr_s      = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di    = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_s + 1e-10)
    minus_di   = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_s + 1e-10)

    # DX = Directional Index, ADX = smoothed DX
    dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx


def classify_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'Regime' column to the DataFrame.
    
    Classification Rules:
        - Bullish:  ADX > 25 AND Close > SMA_50
        - Bearish:  ADX > 25 AND Close < SMA_50
        - Sideways: ADX <= 25 (weak trend, avoid trading)

    Parameters:
        df (pd.DataFrame): Feature-engineered DataFrame (output of feature_engineer.py)

    Returns:
        pd.DataFrame: Same DataFrame with added 'Regime' and 'ADX' columns
    """
    data = df.copy()

    # Compute ADX if not already present
    data["ADX"] = compute_adx(data)

  
    if "SMA_50" not in data.columns:
        data["SMA_50"] = data["Close"].rolling(window=50).mean()

    # ─── Regime Classification Logic ──────────────────────────────────────
    conditions = [
        (data["ADX"] > 25) & (data["Close"] > data["SMA_50"]),  # Strong uptrend
        (data["ADX"] > 25) & (data["Close"] < data["SMA_50"]),  # Strong downtrend
    ]
    choices = ["Bullish", "Bearish"]
    data["Regime"] = np.select(conditions, choices, default="Sideways")

    # Regime Confidence Scoree
    data["Regime_Confidence"] = data["ADX"].clip(upper=60) / 60 * 100

    # regime Numeric (for ML features) 
    regime_map = {"Bullish": 1, "Sideways": 0, "Bearish": -1}
    data["Regime_Num"] = data["Regime"].map(regime_map)

    data.dropna(inplace=True)
    return data


def get_regime_summary(df: pd.DataFrame) -> dict:
    """
    Returns a summary of regime distribution in the dataset.
    Useful for understanding how often each regime occurs.

    Example output:
        {"Bullish": 45.2%, "Bearish": 28.1%, "Sideways": 26.7%}
    """
    counts = df["Regime"].value_counts(normalize=True) * 100
    return counts.to_dict()
