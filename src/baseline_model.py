"""
baseline_model.py
-----------------
Step 2 – XGBoost Baseline Model

• Flattens the 3-D windowed sequences → 2-D for XGBoost
• Trains an XGBRegressor
• Saves / loads the fitted model with joblib
"""

import os
import numpy as np
import joblib
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_HERE)
MODELS_DIR = os.path.join(_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILENAME = "xgboost_model.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Reshape 3-D sequence array to 2-D for tree-based models.

    Input  : (n_samples, window_size, n_features)
    Output : (n_samples, window_size * n_features)
    """
    n_samples, window_size, n_features = X.shape
    return X.reshape(n_samples, window_size * n_features)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_PARAMS = dict(
    n_estimators      = 300,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.80,
    colsample_bytree  = 0.80,
    min_child_weight  = 3,
    gamma             = 0.1,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict | None = None,
    symbol: str = "AAPL",
) -> xgb.XGBRegressor:
    """
    Train and persist an XGBoost regressor.

    Parameters
    ----------
    X_train : shape (n, window, features)
    y_train : shape (n,)
    params  : optional dict to override _DEFAULT_PARAMS
    symbol  : used for the saved filename

    Returns
    -------
    Fitted XGBRegressor
    """
    hp = {**_DEFAULT_PARAMS, **(params or {})}

    X_flat = flatten_sequences(X_train)
    print(f"[XGBoost] Training on {X_flat.shape[0]} samples "
          f"({X_flat.shape[1]} features after flattening) …")

    model = xgb.XGBRegressor(**hp)
    model.fit(
        X_flat, y_train,
        eval_set=[(X_flat, y_train)],
        verbose=False,
    )

    path = os.path.join(MODELS_DIR, f"{symbol}_{MODEL_FILENAME}")
    joblib.dump(model, path)
    print(f"[XGBoost] Model saved → {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict_xgboost(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """Return scaled predictions for X (shape: n_samples, window, features)."""
    return model.predict(flatten_sequences(X)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def load_xgboost_model(symbol: str = "AAPL") -> xgb.XGBRegressor:
    """Load a previously saved XGBoost model."""
    path = os.path.join(MODELS_DIR, f"{symbol}_{MODEL_FILENAME}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved XGBoost model found at '{path}'. "
            "Run main.py first to train."
        )
    return joblib.load(path)


def xgboost_model_exists(symbol: str = "AAPL") -> bool:
    path = os.path.join(MODELS_DIR, f"{symbol}_{MODEL_FILENAME}")
    return os.path.exists(path)
