"""
lstm_model.py
-------------
Step 3 – LSTM Deep Learning Model

Architecture
    Input  → LSTM(128, return_sequences=True)
           → Dropout(0.2)
           → LSTM(64, return_sequences=False)
           → Dropout(0.2)
           → Dense(1)

• Compiled with Adam + MSE
• EarlyStopping on val_loss (patience=5, restore_best_weights=True)
• Model saved as  models/<symbol>_lstm_model.keras
"""

import os
import warnings
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_HERE)
MODELS_DIR = os.path.join(_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILENAME = "lstm_model.keras"


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm_model(
    input_shape: tuple,
    units_layer1: int = 128,
    units_layer2: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build and compile the 2-LSTM model.

    Parameters
    ----------
    input_shape   : (window_size, n_features)
    units_layer1  : LSTM units in the first (return_sequences) layer
    units_layer2  : LSTM units in the second layer
    dropout_rate  : fraction of units to drop
    learning_rate : Adam learning rate

    Returns
    -------
    Compiled Keras Sequential model
    """
    model = Sequential(
        [
            # Layer 1 – stacked LSTM (returns full sequence)
            LSTM(units=units_layer1, return_sequences=True,
                 input_shape=input_shape,
                 name="lstm_1"),
            Dropout(rate=dropout_rate, name="dropout_1"),

            # Layer 2 – LSTM (returns only last step)
            LSTM(units=units_layer2, return_sequences=False,
                 name="lstm_2"),
            Dropout(rate=dropout_rate, name="dropout_2"),

            # Output
            Dense(units=1, name="output"),
        ],
        name="LSTM_StockPredictor",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.10,
    symbol: str = "AAPL",
) -> tuple:
    """
    Train the LSTM model with early stopping.

    Parameters
    ----------
    X_train          : shape (n, window_size, n_features)
    y_train          : shape (n,)
    epochs           : maximum training epochs
    batch_size       : mini-batch size
    validation_split : fraction of training data used for val_loss
    symbol           : used in saved filename

    Returns
    -------
    (model, history)
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"[LSTM] input_shape={input_shape}  "
          f"train_samples={len(X_train)}  "
          f"val_split={validation_split}")

    model = build_lstm_model(input_shape)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        shuffle=False,       # keep temporal order within each epoch
        verbose=1,
    )

    path = os.path.join(MODELS_DIR, f"{symbol}_{MODEL_FILENAME}")
    model.save(path)
    print(f"[LSTM] Model saved → {path}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict_lstm(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """Return scaled predictions for X (shape: n_samples, window, features)."""
    return model.predict(X, verbose=0).flatten().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def load_lstm_model(symbol: str = "AAPL") -> tf.keras.Model:
    """Load a previously saved LSTM model."""
    path = os.path.join(MODELS_DIR, f"{symbol}_{MODEL_FILENAME}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved LSTM model found at '{path}'. "
            "Run main.py first to train."
        )
    return load_model(path)


def lstm_model_exists(symbol: str = "AAPL") -> bool:
    path = os.path.join(MODELS_DIR, f"{symbol}_{MODEL_FILENAME}")
    return os.path.exists(path)
