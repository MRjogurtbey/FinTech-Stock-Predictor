"""
main.py
-------
Orchestration pipeline – runs Steps 1 through 4 in sequence.

Usage
-----
    python main.py              # trains AAPL (default)
    python main.py --symbol TSLA
    python main.py --symbol NVDA --retrain
"""

import argparse
import os
import sys
import time

# ── Make 'src' importable regardless of working directory ─────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.data_preprocessing import (
    prepare_data,
    load_cached_data,
    get_symbols,
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
from src.evaluation import evaluate_all


# ─────────────────────────────────────────────────────────────────────────────
def _banner(text: str) -> None:
    width = 62
    print(f"\n{'═'*width}")
    print(f"  {text}")
    print(f"{'═'*width}\n")


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(symbol: str = "AAPL", retrain: bool = False) -> dict:
    """
    Execute the full ML pipeline.

    Steps
    -----
    1. Data loading & preprocessing  (data_preprocessing.py)
    2. XGBoost baseline training      (baseline_model.py)
    3. LSTM deep-learning training    (lstm_model.py)
    4. Evaluation & visualisation     (evaluation.py)

    Parameters
    ----------
    symbol  : Yahoo Finance ticker (e.g. 'AAPL', 'TSLA')
    retrain : if True, force retraining even if saved models exist

    Returns
    -------
    dict from evaluate_all()
    """
    t0 = time.time()
    _banner(f"FinTech Stock Prediction Pipeline  ·  {symbol}")

    # ── Step 1: Data ───────────────────────────────────────────────────────
    print("STEP 1 / 4  ─  Data Loading & Preprocessing")
    print("-" * 44)

    cached = load_cached_data(symbol) if not retrain else None
    if cached is not None:
        print(f"  ✓ Using cached data for {symbol}.")
        data_dict = cached
    else:
        data_dict = prepare_data(symbol=symbol)

    print(f"  Training sequences : {len(data_dict['X_train']):,}")
    print(f"  Testing  sequences : {len(data_dict['X_test']):,}")
    print(f"  Features           : {data_dict['feature_cols']}")
    print(f"  Window size        : {data_dict['window_size']} days")

    # ── Step 2: XGBoost ────────────────────────────────────────────────────
    print("\nSTEP 2 / 4  ─  XGBoost Baseline Model")
    print("-" * 44)

    if not retrain and xgboost_model_exists(symbol):
        print("  ✓ Loading saved XGBoost model …")
        xgb_model = load_xgboost_model(symbol)
    else:
        xgb_model = train_xgboost(
            data_dict["X_train"],
            data_dict["y_train"],
            symbol=symbol,
        )

    xgb_preds = predict_xgboost(xgb_model, data_dict["X_test"])
    print(f"  Predictions generated: {len(xgb_preds):,}")

    # ── Step 3: LSTM ───────────────────────────────────────────────────────
    print("\nSTEP 3 / 4  ─  LSTM Deep Learning Model")
    print("-" * 44)

    if not retrain and lstm_model_exists(symbol):
        print("  ✓ Loading saved LSTM model …")
        lstm_model = load_lstm_model(symbol)
    else:
        lstm_model, history = train_lstm(
            data_dict["X_train"],
            data_dict["y_train"],
            symbol=symbol,
        )

    lstm_preds = predict_lstm(lstm_model, data_dict["X_test"])
    print(f"  Predictions generated: {len(lstm_preds):,}")

    # ── Step 4: Evaluation ─────────────────────────────────────────────────
    print("\nSTEP 4 / 4  ─  Evaluation & Visualisation")
    print("-" * 44)

    results = evaluate_all(data_dict, xgb_preds, lstm_preds, symbol=symbol)

    elapsed = time.time() - t0
    _banner(f"Pipeline complete  ({elapsed:.1f}s)  ·  charts saved to results/")

    return results


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost + LSTM stock-price predictors."
    )
    parser.add_argument(
        "--symbol", type=str, default="AAPL",
        help="Yahoo Finance ticker symbol (default: AAPL)",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retraining even if saved models exist",
    )
    args = parser.parse_args()

    run_pipeline(symbol=args.symbol.upper(), retrain=args.retrain)


if __name__ == "__main__":
    main()
