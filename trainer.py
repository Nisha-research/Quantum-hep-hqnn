"""
trainer.py
----------
Training module for the HQNN and its classical CNN benchmark.

Provides:
  - StreamlitProgressCallback : Keras callback that updates Streamlit UI live
  - train_model               : generic training routine
  - run_experiment            : trains both models and returns comparison metrics
"""

import time
from typing import Callable

import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

from data_generator import generate_dataset
from quantum_model import build_classical_cnn, build_hqnn


# ── Live Streamlit training callback ─────────────────────────────────────────

class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that reports per-epoch metrics directly into the
    Streamlit UI via placeholder containers.

    Parameters
    ----------
    n_epochs        : int
        Total epochs to train (used for progress bar scaling).
    progress_bar    : st.delta_generator.DeltaGenerator
        A `st.progress()` handle to update.
    chart_container : st.delta_generator.DeltaGenerator
        A `st.empty()` placeholder in which live loss / accuracy charts are
        rendered each epoch using Matplotlib.
    status_text     : st.delta_generator.DeltaGenerator
        A `st.empty()` placeholder for a short text status line.
    """

    def __init__(
        self,
        n_epochs: int,
        progress_bar,
        chart_container,
        status_text,
    ):
        super().__init__()
        self.n_epochs       = n_epochs
        self.progress_bar   = progress_bar
        self.chart_container = chart_container
        self.status_text    = status_text

        # Accumulate per-epoch metric history
        self.history: dict[str, list[float]] = {
            "loss": [], "val_loss": [],
            "accuracy": [], "val_accuracy": [],
        }

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        # Store metrics
        for key in self.history:
            self.history[key].append(float(logs.get(key, 0.0)))

        # ── Progress bar ──────────────────────────────────────────────────
        progress = (epoch + 1) / self.n_epochs
        self.progress_bar.progress(progress)

        # ── Status text ───────────────────────────────────────────────────
        self.status_text.markdown(
            f"**Epoch {epoch + 1}/{self.n_epochs}** — "
            f"loss: `{logs.get('loss', 0):.4f}` | "
            f"val_loss: `{logs.get('val_loss', 0):.4f}` | "
            f"acc: `{logs.get('accuracy', 0):.3f}` | "
            f"val_acc: `{logs.get('val_accuracy', 0):.3f}`"
        )

        # ── Live charts (rendered only after first epoch) ─────────────────
        if len(self.history["loss"]) > 0:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
            epochs_range = range(1, len(self.history["loss"]) + 1)

            # Loss curve
            ax1.plot(epochs_range, self.history["loss"],     "b-o", ms=4, label="Train loss")
            ax1.plot(epochs_range, self.history["val_loss"], "r--s", ms=4, label="Val loss")
            ax1.set_title("Loss", fontweight="bold")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Binary cross-entropy")
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)

            # Accuracy curve
            ax2.plot(epochs_range, self.history["accuracy"],     "b-o", ms=4, label="Train acc")
            ax2.plot(epochs_range, self.history["val_accuracy"], "r--s", ms=4, label="Val acc")
            ax2.set_title("Accuracy", fontweight="bold")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_ylim(0, 1.05)
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)

            fig.tight_layout()
            self.chart_container.pyplot(fig)
            plt.close(fig)


# ── Core training routine ─────────────────────────────────────────────────────

def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int,
    batch_size: int = 16,
    extra_callbacks: list | None = None,
) -> tuple[tf.keras.callbacks.History, float]:
    """
    Train *model* and measure wall-clock time.

    Parameters
    ----------
    model         : compiled tf.keras.Model
    X_train, y_train : training data and labels
    X_val, y_val     : validation data and labels
    n_epochs      : int
    batch_size    : int
    extra_callbacks : list of additional Keras callbacks

    Returns
    -------
    history      : tf.keras.callbacks.History
    elapsed_time : float  (seconds)
    """
    callbacks = extra_callbacks or []

    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,          # suppress Keras stdout; UI is updated via callback
    )
    elapsed = time.perf_counter() - t0
    return history, elapsed


# ── Full experiment runner ────────────────────────────────────────────────────

def run_experiment(
    n_samples: int,
    n_epochs: int,
    learning_rate: float,
    hqnn_ui_callbacks: list | None = None,
    cnn_ui_callbacks: list | None = None,
    seed: int = 42,
) -> dict:
    """
    Generate data, train both models, and return comparison metrics.

    Parameters
    ----------
    n_samples         : int   — total dataset size
    n_epochs          : int   — training epochs for both models
    learning_rate     : float — Adam learning rate
    hqnn_ui_callbacks : list  — extra callbacks for HQNN (e.g. Streamlit progress)
    cnn_ui_callbacks  : list  — extra callbacks for Classical CNN
    seed              : int   — RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        "hqnn_history", "hqnn_time", "hqnn_accuracy", "hqnn_precision",
        "cnn_history",  "cnn_time",  "cnn_accuracy",  "cnn_precision",
        "X_val", "y_val"   ← held-out validation arrays for extra analysis
    """
    # ── Data ─────────────────────────────────────────────────────────────
    X, y = generate_dataset(n_samples=n_samples, seed=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # ── HQNN ─────────────────────────────────────────────────────────────
    hqnn = build_hqnn(learning_rate=learning_rate)
    hqnn_hist, hqnn_time = train_model(
        hqnn, X_train, y_train, X_val, y_val,
        n_epochs=n_epochs,
        extra_callbacks=hqnn_ui_callbacks,
    )

    y_pred_hqnn = (hqnn.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    hqnn_acc  = accuracy_score(y_val, y_pred_hqnn)
    hqnn_prec = precision_score(y_val, y_pred_hqnn, zero_division=0)

    # ── Classical CNN ─────────────────────────────────────────────────────
    cnn = build_classical_cnn(learning_rate=learning_rate)
    cnn_hist, cnn_time = train_model(
        cnn, X_train, y_train, X_val, y_val,
        n_epochs=n_epochs,
        extra_callbacks=cnn_ui_callbacks,
    )

    y_pred_cnn = (cnn.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    cnn_acc  = accuracy_score(y_val, y_pred_cnn)
    cnn_prec = precision_score(y_val, y_pred_cnn, zero_division=0)

    return {
        # HQNN results
        "hqnn_history":  hqnn_hist.history,
        "hqnn_time":     hqnn_time,
        "hqnn_accuracy": hqnn_acc,
        "hqnn_precision": hqnn_prec,
        # Classical CNN results
        "cnn_history":   cnn_hist.history,
        "cnn_time":      cnn_time,
        "cnn_accuracy":  cnn_acc,
        "cnn_precision": cnn_prec,
        # Validation set (for downstream analysis)
        "X_val": X_val,
        "y_val": y_val,
    }
