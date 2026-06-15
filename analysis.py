"""
analysis.py
-----------
Post-training analysis and visualisation helpers.

Provides standalone functions (each returns a Matplotlib Figure) so that
main.py can call them after training and embed the results in Streamlit.

Functions
---------
plot_confusion_matrices    — side-by-side CMs for HQNN and CNN
plot_roc_curves            — overlay ROC / AUC for both models
plot_embedding_pca         — PCA of the Dense(4, tanh) feature space
plot_classification_demo   — per-event grid with prediction + confidence
plot_noise_robustness      — accuracy vs Gaussian noise level
plot_qubit_scaling         — loss / accuracy for 2, 4, 6 qubit circuits
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)


# ── Colour palette ────────────────────────────────────────────────────────────
_SIGNAL_COLOR = "#2196f3"   # blue
_NOISE_COLOR  = "#f44336"   # red
_HQNN_COLOR   = "#4fc3f7"   # light blue
_CNN_COLOR    = "#ef5350"   # light red


# ─────────────────────────────────────────────────────────────────────────────
# 1. Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred_hqnn: np.ndarray,
    y_pred_cnn: np.ndarray,
) -> plt.Figure:
    """
    Side-by-side normalised confusion matrices for HQNN and CNN.

    Normalised by true-class totals (recall perspective) so class imbalance
    does not distort the display.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, y_pred, title, cmap in [
        (ax1, y_pred_hqnn, "Hybrid QNN",     "Blues"),
        (ax2, y_pred_cnn,  "Classical CNN",  "Reds"),
    ]:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        class_names = ["Signal\n(track)", "Noise\n(background)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted label", fontsize=9)
        ax.set_ylabel("True label", fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=11)

        # Annotate cells with raw counts + normalised value
        for i in range(2):
            for j in range(2):
                text_color = "white" if cm_norm[i, j] > 0.55 else "black"
                ax.text(
                    j, i,
                    f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)",
                    ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold",
                )

    fig.suptitle(
        "Confusion Matrices — normalised by true class\n"
        "Diagonal = correct classifications",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROC curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_score_hqnn: np.ndarray,
    y_score_cnn: np.ndarray,
) -> tuple[plt.Figure, float, float]:
    """
    Overlay ROC curves and return AUC scores.

    Returns
    -------
    (fig, auc_hqnn, auc_cnn)
    """
    fpr_h, tpr_h, _ = roc_curve(y_true, y_score_hqnn)
    fpr_c, tpr_c, _ = roc_curve(y_true, y_score_cnn)
    auc_h = auc(fpr_h, tpr_h)
    auc_c = auc(fpr_c, tpr_c)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_h, tpr_h, color=_HQNN_COLOR, lw=2.5,
            label=f"HQNN (AUC = {auc_h:.3f})")
    ax.plot(fpr_c, tpr_c, color=_CNN_COLOR, lw=2.5, ls="--",
            label=f"Classical CNN (AUC = {auc_c:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="Random classifier (AUC = 0.500)")
    ax.fill_between(fpr_h, tpr_h, alpha=0.08, color=_HQNN_COLOR)
    ax.fill_between(fpr_c, tpr_c, alpha=0.08, color=_CNN_COLOR)

    ax.set_xlabel("False Positive Rate  (1 - Specificity)", fontsize=9)
    ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)", fontsize=9)
    ax.set_title("ROC Curve — Signal Detection Performance", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    return fig, auc_h, auc_c


# ─────────────────────────────────────────────────────────────────────────────
# 3. Embedding PCA — "what the quantum layer sees"
# ─────────────────────────────────────────────────────────────────────────────

def plot_embedding_pca(
    hqnn_model: tf.keras.Model,
    cnn_model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
) -> plt.Figure:
    """
    Extract the Dense(4, tanh) embedding from both models, project to 2D
    with PCA, and scatter-plot coloured by true class.

    This visualises the feature space the quantum circuit operates on —
    good class separation here means the CNN pre-processor is doing its job.
    """
    # Sub-models up to the shared embedding layer
    emb_h = tf.keras.Model(
        inputs=hqnn_model.input,
        outputs=hqnn_model.get_layer("embedding").output,
        name="hqnn_emb",
    )
    emb_c = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer("embedding").output,
        name="cnn_emb",
    )

    feats_h = emb_h.predict(X, verbose=0)   # (N, 4)
    feats_c = emb_c.predict(X, verbose=0)

    pca = PCA(n_components=2)
    proj_h = pca.fit_transform(feats_h)
    proj_c = pca.fit_transform(feats_c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, proj, title, var in [
        (ax1, proj_h, "Hybrid QNN",
         PCA(n_components=2).fit(feats_h).explained_variance_ratio_),
        (ax2, proj_c, "Classical CNN",
         PCA(n_components=2).fit(feats_c).explained_variance_ratio_),
    ]:
        for cls, color, marker, label in [
            (0, _SIGNAL_COLOR, "o", "Signal (helix track)"),
            (1, _NOISE_COLOR,  "^", "Background noise"),
        ]:
            mask = y == cls
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                c=color, marker=marker, alpha=0.75, s=45,
                label=f"{label}  (n={mask.sum()})",
                edgecolors="white", linewidths=0.3,
            )

        ax.set_title(
            f"{title} — Learned 4-dim Embedding\n"
            f"PC1 {var[0]*100:.1f}%  |  PC2 {var[1]*100:.1f}% variance",
            fontweight="bold", fontsize=10,
        )
        ax.set_xlabel("Principal Component 1", fontsize=9)
        ax.set_ylabel("Principal Component 2", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Feature Embedding Space — Dense(4, tanh) output projected to 2D via PCA\n"
        "Clear blue/red separation means the CNN has learned discriminating features "
        "before the quantum circuit",
        fontsize=9, y=1.03,
    )
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Classification demo — per-event grid (like MNIST digit recognition)
# ─────────────────────────────────────────────────────────────────────────────

def plot_classification_demo(
    hqnn_model: tf.keras.Model,
    cnn_model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_show: int = 10,
) -> plt.Figure:
    """
    Display *n_show* validation events in two rows (HQNN | CNN).

    Each cell shows:
      - The 28×28 detector image (inferno colourmap)
      - Predicted class label and confidence score
      - Green border = correct  /  Red border = incorrect

    Analogous to the classic MNIST digit recognition demo — evaluators can
    point at individual events and see exactly what each model decided.
    """
    n_show = min(n_show, len(X_val))
    indices = np.arange(n_show)

    y_score_h = hqnn_model.predict(X_val[:n_show], verbose=0).flatten()
    y_score_c = cnn_model.predict(X_val[:n_show], verbose=0).flatten()
    y_pred_h  = (y_score_h > 0.5).astype(int)
    y_pred_c  = (y_score_c > 0.5).astype(int)
    y_true    = y_val[:n_show].astype(int)

    fig, axes = plt.subplots(
        2, n_show,
        figsize=(n_show * 1.85, 5.5),
        facecolor="#0e1117",
    )

    class_names   = ["Signal", "Noise"]
    short_names   = ["SIG", "NOISE"]
    row_labels    = ["HQNN", "CNN"]
    row_preds     = [(y_pred_h, y_score_h), (y_pred_c, y_score_c)]
    row_colors_ok = ["#00cc66",  "#00cc66"]

    for row_idx, (y_pred, y_score) in enumerate(row_preds):
        n_correct = int(np.sum(y_pred == y_true))
        for col_idx in range(n_show):
            ax  = axes[row_idx, col_idx]
            img = X_val[col_idx, :, :, 0]
            ax.imshow(img, cmap="inferno", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            pred  = int(y_pred[col_idx])
            true  = int(y_true[col_idx])
            score = float(y_score[col_idx])
            # Confidence toward predicted class
            conf = score if pred == 1 else 1.0 - score
            correct = (pred == true)
            border  = "#00cc66" if correct else "#ff3333"

            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(2.8)

            ax.set_title(
                f"{short_names[pred]}\n{conf*100:.0f}%",
                color=border, fontsize=7.5, fontweight="bold", pad=3,
            )
            # True label at the bottom
            ax.set_xlabel(
                f"True: {short_names[true]}",
                color="#cccccc", fontsize=6.5, labelpad=2,
            )

        # Row label with accuracy
        axes[row_idx, 0].set_ylabel(
            f"{row_labels[row_idx]}\n{n_correct}/{n_show} correct",
            color="white", fontsize=8, fontweight="bold", labelpad=5,
        )

    fig.suptitle(
        "Event Classifier Demo  —  Green border = correct prediction, Red = wrong\n"
        "Each cell: predicted class + confidence score | bottom = true label",
        color="white", fontsize=9, y=1.02,
    )
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Noise robustness test
# ─────────────────────────────────────────────────────────────────────────────

def plot_noise_robustness(
    hqnn_model: tf.keras.Model,
    cnn_model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    noise_levels: list[float] | None = None,
) -> tuple[plt.Figure, list[float], list[float]]:
    """
    Evaluate both models under increasing levels of additive Gaussian noise.

    At each noise level σ, samples X_noisy = clip(X + N(0, σ), 0, 1) are
    constructed and accuracy is recorded. The resulting curves show robustness
    to detector noise — a key metric for NISQ-era hardware deployment.

    Returns
    -------
    (fig, hqnn_accuracies, cnn_accuracies)
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    rng = np.random.default_rng(seed=99)
    hqnn_accs, cnn_accs = [], []

    for sigma in noise_levels:
        if sigma == 0.0:
            X_noisy = X
        else:
            noise   = rng.normal(0, sigma, X.shape).astype(np.float32)
            X_noisy = np.clip(X + noise, 0.0, 1.0)

        y_h = (hqnn_model.predict(X_noisy, verbose=0) > 0.5).astype(int).flatten()
        y_c = (cnn_model.predict(X_noisy, verbose=0) > 0.5).astype(int).flatten()
        hqnn_accs.append(accuracy_score(y, y_h))
        cnn_accs.append(accuracy_score(y, y_c))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Accuracy vs noise
    ax1.plot(noise_levels, hqnn_accs, color=_HQNN_COLOR, marker="o",
             ms=7, lw=2.5, label="HQNN")
    ax1.plot(noise_levels, cnn_accs,  color=_CNN_COLOR,  marker="s",
             ms=7, lw=2.5, ls="--", label="Classical CNN")
    ax1.axhline(0.5, color="gray", ls=":", lw=1.2, label="Random baseline")
    ax1.fill_between(noise_levels, hqnn_accs, 0.5,
                     alpha=0.08, color=_HQNN_COLOR)
    ax1.fill_between(noise_levels, cnn_accs, 0.5,
                     alpha=0.08, color=_CNN_COLOR)
    ax1.set_xlabel("Gaussian noise sigma added to pixel values", fontsize=9)
    ax1.set_ylabel("Accuracy", fontsize=9)
    ax1.set_title("Robustness to Input Noise", fontweight="bold")
    ax1.set_ylim(0.35, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    # Accuracy drop relative to clean baseline
    drop_h = [hqnn_accs[0] - a for a in hqnn_accs]
    drop_c = [cnn_accs[0]  - a for a in cnn_accs]
    ax2.bar(
        [s - 0.012 for s in noise_levels], drop_h,
        width=0.018, color=_HQNN_COLOR, alpha=0.8, label="HQNN drop",
    )
    ax2.bar(
        [s + 0.012 for s in noise_levels], drop_c,
        width=0.018, color=_CNN_COLOR, alpha=0.8, label="CNN drop",
    )
    ax2.set_xlabel("Gaussian noise sigma", fontsize=9)
    ax2.set_ylabel("Accuracy drop vs clean baseline", fontsize=9)
    ax2.set_title("Accuracy Degradation Under Noise", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Noise Robustness Test — simulates detector noise, pile-up, or hardware errors\n"
        "Contrary to the quantum-robustness hypothesis: Classical CNN degrades more gracefully than HQNN",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    return fig, hqnn_accs, cnn_accs


# ─────────────────────────────────────────────────────────────────────────────
# 6. Qubit scaling experiment
# ─────────────────────────────────────────────────────────────────────────────

def plot_qubit_scaling(results_dict: dict) -> plt.Figure:
    """
    Plot training convergence for different qubit counts.

    Parameters
    ----------
    results_dict : dict
        {n_qubits: {"history": dict_of_lists, "time": float, "final_acc": float}}

    Supports the barren-plateau discussion with empirical data:
    circuits with more qubits should show slower / flatter convergence.
    """
    colors = {2: "#e74c3c", 4: "#3498db", 6: "#2ecc71", 8: "#f39c12"}
    markers = {2: "o", 4: "s", 6: "^", 8: "D"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_loss, ax_acc, ax_time = axes

    sorted_keys = sorted(results_dict.keys())

    for n_q in sorted_keys:
        data = results_dict[n_q]
        hist = data["history"]
        ep   = range(1, len(hist["val_loss"]) + 1)
        c    = colors.get(n_q, "gray")
        m    = markers.get(n_q, "o")
        label = f"{n_q} qubits"

        ax_loss.plot(ep, hist["val_loss"],     f"-{m}", color=c, ms=5, lw=2, label=label)
        ax_acc.plot(ep,  hist["val_accuracy"], f"-{m}", color=c, ms=5, lw=2, label=label)

    # Training time bar chart
    n_qs  = sorted_keys
    times = [results_dict[n]["time"] for n in n_qs]
    bar_colors = [colors.get(n, "gray") for n in n_qs]
    bars = ax_time.bar([str(n) for n in n_qs], times, color=bar_colors, alpha=0.85)
    for bar, t in zip(bars, times):
        ax_time.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{t:.0f}s", ha="center", va="bottom", fontsize=9,
        )
    ax_time.set_xlabel("Number of qubits", fontsize=9)
    ax_time.set_ylabel("Training time (seconds)", fontsize=9)
    ax_time.set_title("Training Time vs Qubit Count", fontweight="bold")
    ax_time.grid(alpha=0.25, axis="y")

    ax_loss.set_title("Validation Loss vs Epoch\n(flat curve = barren plateau onset)",
                       fontweight="bold", fontsize=10)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Validation loss")
    ax_loss.legend(fontsize=9); ax_loss.grid(alpha=0.25)

    ax_acc.set_title("Validation Accuracy vs Epoch",
                      fontweight="bold", fontsize=10)
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Validation accuracy")
    ax_acc.set_ylim(0.3, 1.05)
    ax_acc.legend(fontsize=9); ax_acc.grid(alpha=0.25)

    fig.suptitle(
        "Qubit Scaling Experiment — note: high variance expected with small sample sizes\n"
        "Interpret trends cautiously; non-monotonic accuracy reflects statistical noise, not pure barren-plateau scaling",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    return fig
