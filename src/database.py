import sqlite3
import pandas as pd
import os
from datetime import datetime


DATABASE_URL = os.getenv("DATABASE_URL", "data/trade_log.db")

# Check if using PostgreSQL (production) or SQLite (local development)
USE_POSTGRES = DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")


def get_connection():
    """
    Returns a database connection.
    Automatically uses SQLite locally or PostgreSQL in production.
    """
    if USE_POSTGRES:
        try:
            import psycopg2
            return psycopg2.connect(DATABASE_URL)
        except ImportError:
            print("[WARN] psycopg2 not installed. Falling back to SQLite.")
            return sqlite3.connect("data/trade_log.db")
    else:
        os.makedirs("data", exist_ok=True)
        return sqlite3.connect(DATABASE_URL)


def init_db():
    """
    Creates the trade_signals table if it doesn't already exist.
    This runs once at startup.

    Table Schema:
        id        — Auto-incrementing primary key
        stock     — Ticker symbol (e.g., RELIANCE.NS)
        date      — Date of signal
        price     — Closing price at signal time
        signal    — BUY / HOLD / SELL
        regime    — Market regime at signal time (Bullish / Bearish / Sideways)
        accuracy  — Model accuracy at the time of prediction
        created_at— Timestamp when the record was inserted
    """
    conn = get_connection()
    cursor = conn.cursor()

    # SQLite-compatible CREATE TABLE statement
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            stock       TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            price       REAL    NOT NULL,
            signal      TEXT    NOT NULL,
            regime      TEXT    NOT NULL,
            accuracy    REAL,
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def log_prediction(
    stock:    str,
    date:     str,
    price:    float,
    signal:   str,
    regime:   str,
    accuracy: float
) -> bool:
    """
    Inserts a new trade signal record into the database.

    Parameters:
        stock    (str):   Ticker symbol
        date     (str):   Signal date as string
        price    (float): Closing price
        signal   (str):   "BUY", "HOLD", or "SELL"
        regime   (str):   "Bullish", "Bearish", or "Sideways"
        accuracy (float): Model accuracy (0.0–1.0)

    Returns:
        bool: True if successful, False if an error occurred
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trade_signals (stock, date, price, signal, regime, accuracy, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (stock, date, price, signal, regime, round(accuracy, 4), datetime.now().isoformat()))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"[ERROR] log_prediction: {e}")
        return False


def get_trade_log(limit: int = 50) -> pd.DataFrame:
    """
    Retrieves the most recent trade signal logs from the database.

    Parameters:
        limit (int): Maximum number of rows to return (default 50)

    Returns:
        pd.DataFrame: Trade log with all columns
    """
    try:
        conn = get_connection()
        query = f"""
            SELECT id, stock, date, price, signal, regime, accuracy, created_at
            FROM trade_signals
            ORDER BY id DESC
            LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Format columns for display
        if not df.empty:
            df["price"]    = df["price"].map("₹{:,.2f}".format)
            df["accuracy"] = df["accuracy"].map("{:.1%}".format)
            df.rename(columns={
                "id": "ID", "stock": "Ticker", "date": "Signal Date",
                "price": "Price (₹)", "signal": "Signal", "regime": "Regime",
                "accuracy": "Accuracy", "created_at": "Logged At"
            }, inplace=True)

        return df

    except Exception as e:
        print(f"[ERROR] get_trade_log: {e}")
        return pd.DataFrame()


def get_signal_statistics() -> dict:
    """
    Returns aggregate statistics from the trade log.
    Useful for building a performance dashboard.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT signal, COUNT(*) FROM trade_signals GROUP BY signal")
        signal_counts = dict(cursor.fetchall())

        cursor.execute("SELECT AVG(accuracy) FROM trade_signals")
        avg_accuracy = cursor.fetchone()[0] or 0

        conn.close()
        return {"signal_counts": signal_counts, "avg_accuracy": avg_accuracy}

    except Exception as e:
        print(f"[ERROR] get_signal_statistics: {e}")
        return {}
