"""
evaluation.py
-------------
Step 4 – Evaluation & Visualization

• Inverse-transforms scaled predictions back to USD
• Computes RMSE, MAE, MAPE
• Produces two publication-quality charts:
    1. Actual vs XGBoost vs LSTM (+ absolute error panel)
    2. Metrics comparison bar chart
• All figures saved to  results/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
COLORS = {
    "actual": "#2196F3",
    "xgb":    "#FF9800",
    "lstm":   "#4CAF50",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Inverse transform
# ─────────────────────────────────────────────────────────────────────────────
def inverse_transform_1d(
    predictions: np.ndarray,
    scaler,
    n_features: int,
    target_col_idx: int = -1,
) -> np.ndarray:
    """
    Reconstruct real-price values from scaled predictions.

    The scaler was fit on ALL features; we put predictions into a zero
    dummy array at the target column position, then inverse-transform.
    """
    if target_col_idx < 0:
        target_col_idx = n_features + target_col_idx

    dummy = np.zeros((len(predictions), n_features), dtype=np.float32)
    dummy[:, target_col_idx] = predictions
    return scaler.inverse_transform(dummy)[:, target_col_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metrics
# ─────────────────────────────────────────────────────────────────────────────
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """Return a dict with RMSE, MAE, MAPE (all in real USD / %)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    # Guard against zero prices
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)

    metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
    print(f"\n{'─'*40}")
    print(f"  {model_name} Performance")
    print(f"{'─'*40}")
    print(f"  RMSE : ${rmse:>10.4f}")
    print(f"  MAE  : ${mae:>10.4f}")
    print(f"  MAPE :  {mape:>9.4f} %")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prediction chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_predictions(
    dates: pd.DatetimeIndex,
    y_actual: np.ndarray,
    y_xgb: np.ndarray,
    y_lstm: np.ndarray,
    symbol: str = "AAPL",
    save: bool = True,
) -> plt.Figure:
    """
    Two-panel figure:
        Top    – price curves (Actual / XGBoost / LSTM)
        Bottom – absolute errors per day
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(
        f"{symbol}  ·  Stock Price Prediction (Test Period)",
        fontsize=16, fontweight="bold", y=1.01,
    )

    # ── Top panel: prices ──────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, y_actual, label="Actual",   color=COLORS["actual"],
             linewidth=2.0, zorder=3)
    ax1.plot(dates, y_xgb,   label="XGBoost",  color=COLORS["xgb"],
             linewidth=1.5, linestyle="--",  alpha=0.9, zorder=2)
    ax1.plot(dates, y_lstm,  label="LSTM",      color=COLORS["lstm"],
             linewidth=1.5, linestyle="-.",  alpha=0.9, zorder=2)

    ax1.set_ylabel("Price (USD)", fontsize=12)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.set_title("Predicted vs Actual Close Prices", fontsize=13)

    # ── Bottom panel: |errors| ─────────────────────────────────────────────
    ax2 = axes[1]
    err_xgb  = np.abs(y_actual - y_xgb)
    err_lstm = np.abs(y_actual - y_lstm)

    ax2.fill_between(dates, err_xgb,  alpha=0.35, color=COLORS["xgb"])
    ax2.fill_between(dates, err_lstm, alpha=0.35, color=COLORS["lstm"])
    ax2.plot(dates, err_xgb,  label="XGBoost |error|", color=COLORS["xgb"],
             linewidth=1.2)
    ax2.plot(dates, err_lstm, label="LSTM |error|",    color=COLORS["lstm"],
             linewidth=1.2)

    ax2.set_ylabel("Absolute Error (USD)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(fontsize=11, loc="upper left")
    ax2.set_title("Absolute Prediction Errors", fontsize=13)

    # Date formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, f"{symbol}_predictions.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[eval] Prediction chart saved → {path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Metrics comparison chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_comparison(
    metrics_xgb: dict,
    metrics_lstm: dict,
    symbol: str = "AAPL",
    save: bool = True,
) -> plt.Figure:
    """Grouped bar chart: XGBoost vs LSTM for RMSE, MAE, MAPE."""
    metric_names = ["RMSE", "MAE", "MAPE"]
    units        = ["USD",  "USD", "%"]
    xgb_vals     = [metrics_xgb[m]  for m in metric_names]
    lstm_vals    = [metrics_lstm[m] for m in metric_names]

    x      = np.arange(len(metric_names))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_xgb  = ax.bar(x - width/2, xgb_vals,  width, label="XGBoost",
                        color=COLORS["xgb"],  edgecolor="black", alpha=0.85)
    bars_lstm = ax.bar(x + width/2, lstm_vals, width, label="LSTM",
                        color=COLORS["lstm"], edgecolor="black", alpha=0.85)

    # Value annotations
    for bar, val, unit in zip(bars_xgb, xgb_vals, units):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * max(max(xgb_vals), max(lstm_vals)),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    for bar, val, unit in zip(bars_lstm, lstm_vals, units):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * max(max(xgb_vals), max(lstm_vals)),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{m}\n({u})" for m, u in zip(metric_names, units)], fontsize=12
    )
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"{symbol}  ·  Model Performance Comparison (XGBoost vs LSTM)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=12)
    sns.despine()
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, f"{symbol}_metrics_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[eval] Metrics chart saved → {path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full evaluation entry-point
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_all(
    data_dict: dict,
    xgb_preds_scaled: np.ndarray,
    lstm_preds_scaled: np.ndarray,
    symbol: str = "AAPL",
) -> dict:
    """
    Inverse-transform, compute metrics, and produce both charts.

    Returns
    -------
    dict with keys:
        y_actual, y_xgb, y_lstm, metrics_xgb, metrics_lstm, test_dates
    """
    scaler       = data_dict["scaler"]
    n_features   = len(data_dict["feature_cols"])
    y_test_sc    = data_dict["y_test"]

    # Inverse transform back to USD
    y_actual = inverse_transform_1d(y_test_sc,          scaler, n_features)
    y_xgb    = inverse_transform_1d(xgb_preds_scaled,   scaler, n_features)
    y_lstm   = inverse_transform_1d(lstm_preds_scaled,  scaler, n_features)

    # Metrics
    metrics_xgb  = calculate_metrics(y_actual, y_xgb,  "XGBoost")
    metrics_lstm = calculate_metrics(y_actual, y_lstm,  "LSTM")

    # Recover test-period dates
    raw_df       = data_dict["raw_df"]
    window_size  = data_dict["window_size"]
    train_size   = data_dict["train_size"]
    start_idx    = train_size + window_size
    end_idx      = start_idx + len(y_actual)
    test_dates   = raw_df.index[start_idx:end_idx]

    # Charts
    plot_predictions(test_dates, y_actual, y_xgb, y_lstm, symbol=symbol)
    plot_metrics_comparison(metrics_xgb, metrics_lstm, symbol=symbol)

    print(f"\n[eval] {'='*50}")
    print(f"[eval]  Winner: "
          f"{'LSTM' if metrics_lstm['RMSE'] < metrics_xgb['RMSE'] else 'XGBoost'} "
          f"(lower RMSE)")
    print(f"[eval] {'='*50}")

    return {
        "y_actual":    y_actual,
        "y_xgb":       y_xgb,
        "y_lstm":      y_lstm,
        "metrics_xgb": metrics_xgb,
        "metrics_lstm":metrics_lstm,
        "test_dates":  test_dates,
    }
