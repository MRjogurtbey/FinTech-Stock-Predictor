"""
data_preprocessing.py
---------------------
Step 1 – Data Loading & Preprocessing

• Reads the stock list from yahoo-stocks-data.xlsx
• Downloads 5-year daily OHLCV history via yfinance
• Handles missing values (forward-fill → linear interpolation)
• Target: Close (≡ Adj Close in modern yfinance)
• Chronological 80/20 train-test split
• MinMaxScaler(feature_range=(0,1)) on all features
• 60-day sliding-window sequence generator
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR   = os.path.join(_ROOT, "data")
MODELS_DIR = os.path.join(_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 60
TRAIN_RATIO  = 0.80
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COL   = "Close"           # yfinance ≥0.2 already adjusts Close


# ─────────────────────────────────────────────────────────────────────────────
# 1. Stock list
# ─────────────────────────────────────────────────────────────────────────────
def load_stock_list(xlsx_path: str | None = None) -> pd.DataFrame:
    """Return the 349-row snapshot DataFrame from the Excel file."""
    if xlsx_path is None:
        xlsx_path = os.path.join(DATA_DIR, "yahoo-stocks-data.xlsx")
    df = pd.read_excel(xlsx_path)
    return df


def get_symbols(xlsx_path: str | None = None) -> list[str]:
    """Return the list of ticker symbols from the Excel file."""
    return load_stock_list(xlsx_path)["symbol"].dropna().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Historical download
# ─────────────────────────────────────────────────────────────────────────────
def download_historical_data(
    symbol: str = "AAPL",
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV history from Yahoo Finance via yfinance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)  # strip timezone
    df.index.name = "Date"

    # Keep only the columns we need
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available].copy()

    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'. "
                         "Check the ticker or your internet connection.")
    print(f"[download] {symbol}: {len(df)} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then linear-interpolate any remaining NaNs."""
    df = df.ffill()
    df = df.interpolate(method="linear")
    df = df.dropna()
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Clean, scale, and return:
        scaled_df    – DataFrame with same index, values in [0,1]
        scaler       – fitted MinMaxScaler
        feature_cols – column list (TARGET_COL is last)
        original_df  – unscaled numeric DataFrame
    """
    df = _handle_missing(df)

    # Re-order so target is always the last column (makes sequence labels easy)
    cols = [c for c in FEATURE_COLS if c in df.columns and c != TARGET_COL]
    if TARGET_COL in df.columns:
        cols = cols + [TARGET_COL]
    else:
        # Fallback: use last column as target
        cols = df.columns.tolist()

    original_df = df[cols].copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(original_df.values)
    scaled_df = pd.DataFrame(scaled_values, columns=cols, index=df.index)

    return scaled_df, scaler, cols, original_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train / test split
# ─────────────────────────────────────────────────────────────────────────────
def chronological_split(
    scaled_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple:
    """
    Chronological (no shuffle) split.
    Returns (train_df, test_df, train_size_int).
    """
    n = len(scaled_df)
    train_size = int(n * train_ratio)
    return scaled_df.iloc[:train_size], scaled_df.iloc[train_size:], train_size


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sliding-window sequence generator
# ─────────────────────────────────────────────────────────────────────────────
def create_sequences(
    data: pd.DataFrame | np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> tuple:
    """
    Build (X, y) from a 2-D array/DataFrame using a sliding window.

    X shape : (n_samples, window_size, n_features)
    y shape : (n_samples,)   ← last column value at position i
    """
    arr = data.values if hasattr(data, "values") else np.asarray(data)
    X, y = [], []
    for i in range(window_size, len(arr)):
        X.append(arr[i - window_size : i])   # shape: (window, features)
        y.append(arr[i, -1])                  # target = last column
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full pipeline entry-point
# ─────────────────────────────────────────────────────────────────────────────
def prepare_data(
    symbol: str = "AAPL",
    xlsx_path: str | None = None,
    window_size: int = WINDOW_SIZE,
    period: str = "5y",
) -> dict:
    """
    End-to-end data preparation.

    Returns a dict with keys:
        X_train, y_train, X_test, y_test   – numpy arrays
        scaler                              – fitted MinMaxScaler
        feature_cols                        – list[str]
        train_size                          – int
        original_df                         – unscaled DataFrame
        raw_df                              – raw downloaded DataFrame
        scaled_df                           – scaled DataFrame
        window_size                         – int
        target_col                          – str
    """
    # Download
    raw_df = download_historical_data(symbol=symbol, period=period)

    # Preprocess
    scaled_df, scaler, feature_cols, original_df = preprocess_data(raw_df)

    # Split
    train_df, test_df, train_size = chronological_split(scaled_df)

    # Sequences
    X_train, y_train = create_sequences(train_df, window_size)
    X_test,  y_test  = create_sequences(test_df,  window_size)

    print(f"[prepare_data] features={feature_cols}  target='{feature_cols[-1]}'")
    print(f"[prepare_data] train sequences: {len(X_train)}  "
          f"test sequences: {len(X_test)}")

    data_dict = {
        "X_train":      X_train,
        "y_train":      y_train,
        "X_test":       X_test,
        "y_test":       y_test,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "train_size":   train_size,
        "original_df":  original_df,
        "raw_df":       raw_df,
        "scaled_df":    scaled_df,
        "window_size":  window_size,
        "target_col":   feature_cols[-1],
    }

    # Cache to disk so app.py can reload without re-downloading
    cache_path = os.path.join(MODELS_DIR, f"{symbol}_data.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"[prepare_data] cached → {cache_path}")

    return data_dict


def load_cached_data(symbol: str = "AAPL") -> dict | None:
    """Load cached data_dict from disk (returns None if not found)."""
    path = os.path.join(MODELS_DIR, f"{symbol}_data.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
