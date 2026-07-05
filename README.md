#  NSE Algorithmic Trading & Market Regime Filter

<div align="center">




##  About The Project

This project builds a **data-driven, ML-powered trading signal system** for Indian equities listed on the NSE. It goes beyond simple technical analysis by using a two-layer decision system:

1. **Market Regime Filter** — Classifies the current market as *Bullish*, *Bearish*, or *Sideways* using ADX (Average Directional Index) and a 50-day SMA. This filters out low-quality signals generated during choppy/ranging markets.

2. **XGBoost Signal Engine** — A gradient-boosted ML classifier trained on **26 engineered features** (momentum, trend, volatility, and volume indicators) to predict the next 1/3/5-day price direction.

>  **Disclaimer**: This project is built for educational and portfolio purposes only. It does not constitute financial advice. Never trade real money based solely on ML model outputs.

---

##  Key Features

| Feature | Description |
|---|---|
|  **XGBoost Classifier** | Gradient boosted trees trained on 26 technical features |
|  **Market Regime Filter** | ADX + SMA-based Bullish/Bearish/Sideways classification |
|  **26 Technical Features** | RSI, MACD, Bollinger Bands, ATR, Stochastic, OBV, and more |
|  **Interactive Charts** | Plotly candlestick + signal overlay + volume charts |
|  **SQL Trade Logging** | Every signal logged to SQLite (upgradeable to PostgreSQL) |
|  **Model Report Card** | Accuracy, Precision, Recall, F1, ROC-AUC benchmarked |
|  **No Data Leakage** | Time-series aware train/test split (never shuffled) |
|  **Streamlit UI** | One-click analysis for any of 8 major NSE stocks |

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web App                       │
│                        (app.py)                             │
└──────────┬──────────────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────┐
    │              Data Pipeline                  │
    │  Yahoo Finance → OHLCV → Feature Engineering│
    │   (yfinance)     (Pandas)  (26 indicators)  │
    └──────┬──────────────────────────────────────┘
           │
    ┌──────▼────────────────────────────────────┐
    │         Market Regime Filter              │
    │  ADX + SMA_50 → Bullish/Bearish/Sideways  │
    └──────┬────────────────────────────────────┘
           │
    ┌──────▼────────────────────────────────────┐
    │         XGBoost Signal Engine             │
    │  Features + Regime → BUY / HOLD / SELL    │
    └──────┬────────────────────────────────────┘
           │
    ┌──────▼────────────────────────────────────┐
    │         SQL Database Logger               │
    │  SQLite (local) / PostgreSQL (production) │
    └───────────────────────────────────────────┘
```

---

##  Tech Stack

- **Language**: Python 3.11+
- **ML Model**: XGBoost (gradient boosted decision trees)
- **Feature Engineering**: Pandas, NumPy
- **Web Frontend**: Streamlit
- **Visualization**: Plotly
- **Data Source**: Yahoo Finance via `yfinance`
- **Database**: SQLite (local) / PostgreSQL (production via Supabase)
- **Model Persistence**: Joblib
- **Version Control**: Git & GitHub

---

##  Project Structure

```
nse-algo-trader/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py       # Yahoo Finance data download
│   ├── feature_engineer.py   # 26 technical indicators
│   ├── regime_filter.py      # ADX-based market regime classifier
│   ├── model_trainer.py      # XGBoost train / evaluate / save
│   ├── database.py           # SQLite / PostgreSQL trade logging
│   └── visualizer.py         # All Plotly chart functions
│
├── models/
│   └── xgboost_nse_model.pkl # Saved trained model (gitignored)
│
├── data/
│   ├── raw/                  # Raw downloaded data (gitignored)
│   └── trade_log.db          # SQLite database (gitignored)
│
└── notebooks/
    └── exploration.ipynb     # EDA and experimentation notebook
```

---

## How to Run Locally

### Prerequisites
- Python 3.11 or higher
- Git

### Step 1 — Clone the Repository
```bash
git clone YOUR_GITHUB_REPO_LINK
cd nse-algo-trader
```

### Step 2 — Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables
```bash
# Copy the template
cp .env.example .env

# Edit .env — leave DATABASE_URL blank to use local SQLite
# For PostgreSQL, set: DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

### Step 5 — Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Deployment

### Streamlit Community Cloud (Free, Recommended)
1. Push code to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub → select `app.py`
4. Add secrets in Streamlit Cloud dashboard (Settings → Secrets):
   ```toml
   DATABASE_URL = "YOUR_SUPABASE_POSTGRESQL_URL"
   ```


### Hugging Face Spaces (Alternative)
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** as the SDK
3. Push your code to the Space repository

---

## 📊 Model Performance

> Results on RELIANCE.NS (Jan 2022 – May 2025, 20% test set)

| Metric | Score | Benchmark |
|---|---|---|
| **Accuracy** | `YOUR_ACCURACY%` | ≥ 55% |
| **Precision** | `YOUR_PRECISION%` | ≥ 50% |
| **Recall** | `YOUR_RECALL%` | ≥ 50% |
| **F1-Score** | `YOUR_F1%` | ≥ 52% |
| **ROC-AUC** | `YOUR_AUC%` | ≥ 55% |

*Run the app and fill in your actual metrics above.*

---

##  Features Engineered (26 Total)

| Category | Indicators |
|---|---|
| **Trend** | SMA 20/50/200, EMA 12/26, Price-to-SMA ratios, Golden Cross |
| **Momentum** | MACD, MACD Signal, MACD Histogram, ROC-10 |
| **Oscillators** | RSI-14, Stochastic %K, Stochastic %D |
| **Volatility** | Bollinger Band Width, BB%, ATR-14, ATR% |
| **Volume** | Volume Ratio, On-Balance Volume (OBV) |
| **Price Action** | 1/5/20-day returns, Candle Body, High-Low % |
| **Regime** | ADX, Regime Numeric, Regime Confidence |

---

## Roadmap

- [ ] Add Backtesting Engine with P&L simulation
- [ ] Add NIFTY 50 Index support
- [ ] Add email/WhatsApp alerts for BUY/SELL signals
- [ ] Implement walk-forward validation
- [ ] Add portfolio-level analysis (multiple stocks simultaneously)
- [ ] Switch to LSTM/Transformer model for sequence modeling

---

##  Author

Sarthak Argade
- LinkedIn: www.linkedin.com/in/sarthak-argade-144966390
- GitHub: https://github.com/sarthakargade9166-max
- Email: sarthakargade9166@gmail.com

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

- [Yahoo Finance](https://finance.yahoo.com) for free market data via `yfinance`
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit](https://streamlit.io) for the amazing web framework
- [Plotly](https://plotly.com) for interactive charting
