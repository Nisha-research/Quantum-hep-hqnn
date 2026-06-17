"""
analysis.py
-----------
Post-training analysis and visualisation helpers.

Provides standalone functions (each returns a Matplotlib Figure) so that
main.py can call them after training and embed the results in Streamlit.

Functions
---------
plot_confusion_matrices        — side-by-side CMs for HQNN and CNN
plot_roc_curves                — overlay ROC / AUC for both models
plot_embedding_pca             — PCA of the Dense(4, tanh) feature space
plot_classification_demo       — per-event grid with prediction + confidence
plot_noise_robustness          — accuracy vs Gaussian pixel noise level
plot_qubit_scaling             — loss / accuracy for 2, 4, 6 qubit circuits
plot_bloch_spheres             — 3D Bloch sphere: signal vs noise qubit states
plot_entanglement_map          — 4×4 quantum mutual information heatmap
plot_hardware_noise_robustness — Qiskit Aer hardware-noise accuracy vs baselines
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

        class_names = ["⚛️ Signal\n(Class 0)", "❌ Noise\n(Class 1)"]
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
            (0, _SIGNAL_COLOR, "o", "⚛️ Signal (Particle Track)"),
            (1, _NOISE_COLOR,  "^", "❌ Background Noise (Detector Hits)"),
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

    class_names   = ["⚛️ Signal", "❌ Noise"]
    short_names   = ["⚛️SIG", "❌NOISE"]
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
        "Noise Robustness Test — accuracy under increasing additive Gaussian noise",
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


# ─────────────────────────────────────────────────────────────────────────────
# 7. Bloch Sphere Visualisation — "Visualise the Quantum Brain"
# ─────────────────────────────────────────────────────────────────────────────

def plot_bloch_spheres(
    hqnn_model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_samples_per_class: int = 12,
) -> plt.Figure:
    """
    For each qubit, plot a 3D Bloch sphere showing where signal-event and
    background-event quantum states land after the full PQC.

    The Bloch vector (⟨X⟩, ⟨Y⟩, ⟨Z⟩) of each qubit is measured via a
    separate diagnostic QNode that runs the same encoding + ansatz but
    measures all three Pauli expectation values.  No TF gradients are
    needed — this is a pure forward-pass analysis circuit.

    Parameters
    ----------
    hqnn_model           : trained HQNN (must contain a "quantum_layer"
                           with a `q_weights` tf.Variable)
    X_val                : validation images, shape (N, 28, 28, 1)
    y_val                : true labels, shape (N,)
    max_samples_per_class: how many events per class to plot (keeps it fast)

    Returns
    -------
    plt.Figure
    """
    import pennylane as qml
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # ── Extract trained weights ───────────────────────────────────────────
    q_weights_tf = hqnn_model.get_layer("quantum_layer").q_weights
    q_weights = q_weights_tf.numpy()          # (N_LAYERS, N_QUBITS, 3)
    n_layers, n_qubits, _ = q_weights.shape

    # ── Build embedding sub-model ─────────────────────────────────────────
    embed_model = tf.keras.Model(
        inputs=hqnn_model.input,
        outputs=hqnn_model.get_layer("embedding").output,
    )

    # Sample up to max_samples_per_class events of each class
    sig_idx   = np.where(y_val == 0)[0][:max_samples_per_class]
    noise_idx = np.where(y_val == 1)[0][:max_samples_per_class]
    sel_idx   = np.concatenate([sig_idx, noise_idx])
    sel_labels = y_val[sel_idx]

    embeddings = embed_model.predict(X_val[sel_idx], verbose=0)  # (N, 4)

    # ── Diagnostic circuit — measures Bloch vector for every qubit ────────
    dev_diag = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev_diag)
    def _bloch_circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RX(np.pi * inputs[i], wires=i)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.Rot(
                    weights[layer, q, 0],
                    weights[layer, q, 1],
                    weights[layer, q, 2],
                    wires=q,
                )
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        # Return all 3 Paulis for every qubit in one shot
        return (
            qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1)),
            qml.expval(qml.PauliX(2)), qml.expval(qml.PauliX(3)),
            qml.expval(qml.PauliY(0)), qml.expval(qml.PauliY(1)),
            qml.expval(qml.PauliY(2)), qml.expval(qml.PauliY(3)),
            qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3)),
        )

    # Compute Bloch vectors for all selected samples
    bloch = []
    for emb in embeddings:
        raw = np.array(_bloch_circuit(emb.astype(float), q_weights), dtype=float)
        # raw: [X0..X3, Y0..Y3, Z0..Z3] → shape (3, n_qubits)
        bloch.append(raw.reshape(3, n_qubits))
    bloch = np.array(bloch)   # (N_sel, 3, n_qubits)

    sig_mask   = sel_labels == 0
    noise_mask = sel_labels == 1

    # ── Draw ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5 * n_qubits, 5), facecolor="#0e1117")
    qubit_labels = [f"Qubit {q}" for q in range(n_qubits)]

    for q in range(n_qubits):
        ax = fig.add_subplot(1, n_qubits, q + 1, projection="3d")
        ax.set_facecolor("#0e1117")

        # Wire-frame sphere
        u = np.linspace(0, 2 * np.pi, 36)
        v = np.linspace(0, np.pi, 18)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(u.shape), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color="#444444", lw=0.4, alpha=0.5)

        # Axes arrows
        for start, end, label, pos in [
            ([0, 0, -1.4], [0, 0, 1.4], "|0⟩", (0, 0, 1.5)),
            ([0, 0, 1.4],  [0, 0, -1.4], "|1⟩", (0, 0, -1.7)),
            ([-1.4, 0, 0], [1.4, 0, 0], "|+⟩", (1.6, 0, 0)),
            ([0, -1.4, 0], [0, 1.4, 0], "|i⟩", (0, 1.6, 0)),
        ]:
            ax.quiver(*start[:3], *(np.array(end) - np.array(start)),
                      color="#666666", alpha=0.5, arrow_length_ratio=0.12,
                      lw=0.8, length=1)
            ax.text(*pos, label, color="#888888", fontsize=7, ha="center")

        # Scatter: signal
        ax.scatter(
            bloch[sig_mask,   0, q],
            bloch[sig_mask,   1, q],
            bloch[sig_mask,   2, q],
            c="#00ff88", s=40, alpha=0.85, edgecolors="white",
            linewidths=0.3, label="⚛️ Signal (Class 0)", zorder=6,
        )
        # Scatter: noise
        ax.scatter(
            bloch[noise_mask, 0, q],
            bloch[noise_mask, 1, q],
            bloch[noise_mask, 2, q],
            c="#ff4444", s=40, alpha=0.85, edgecolors="white",
            linewidths=0.3, label="❌ Noise (Class 1)", zorder=6,
        )

        ax.set_title(qubit_labels[q], color="white", fontsize=10,
                     fontweight="bold", pad=6)
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_zlim([-1.3, 1.3])
        ax.set_xlabel("⟨X⟩", color="#888888", fontsize=7, labelpad=1)
        ax.set_ylabel("⟨Y⟩", color="#888888", fontsize=7, labelpad=1)
        ax.set_zlabel("⟨Z⟩", color="#888888", fontsize=7, labelpad=1)
        ax.tick_params(colors="#555555", labelsize=5)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#222222")
        ax.yaxis.pane.set_edgecolor("#222222")
        ax.zaxis.pane.set_edgecolor("#222222")

        if q == 0:
            ax.legend(fontsize=8, loc="upper left",
                      labelcolor="white", facecolor="#111111",
                      edgecolor="#444444")

    fig.suptitle(
        "Bloch Sphere Visualisation — Qubit State Distribution After Full PQC\n"
        "Green = Signal event  |  Red = Background noise  |  Each sphere = one qubit",
        color="white", fontsize=10, y=1.01,
    )
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Entanglement Strength Map — Quantum Mutual Information Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _partial_trace(state_vec: np.ndarray, keep: list, n: int) -> np.ndarray:
    """
    Compute the reduced density matrix by tracing out all qubits NOT in `keep`.

    Parameters
    ----------
    state_vec : complex ndarray, shape (2**n,)
    keep      : list of qubit indices to retain (0-indexed)
    n         : total number of qubits

    Returns
    -------
    rho : complex ndarray, shape (2**len(keep), 2**len(keep))
    """
    state = state_vec.reshape([2] * n)
    other = [q for q in range(n) if q not in keep]
    perm = list(keep) + other
    state = np.transpose(state, perm)
    dk = 2 ** len(keep)
    do = 2 ** (n - len(keep))
    sf = state.reshape(dk, do)
    return sf @ sf.conj().T


def _von_neumann_entropy(rho: np.ndarray) -> float:
    """Von Neumann entropy S = −Tr(ρ log₂ ρ) in bits."""
    ev = np.linalg.eigvalsh(rho).real
    ev = ev[ev > 1e-14]
    if len(ev) == 0:
        return 0.0
    return float(-np.sum(ev * np.log2(ev)))


def plot_entanglement_map(
    hqnn_model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_samples: int = 16,
) -> plt.Figure:
    """
    Compute and visualise the Quantum Mutual Information (QMI) matrix.

    For every pair of qubits (i, j), QMI is defined as:
        I(i:j) = S(ρᵢ) + S(ρⱼ) − S(ρᵢⱼ)
    where S is the Von Neumann entropy and ρᵢⱼ is the two-qubit reduced
    density matrix obtained by tracing out the other qubits.

    Diagonal entries show the single-qubit entanglement entropy (how
    entangled qubit i is with the rest of the register).

    Values are averaged over `max_samples` validation events.  Brighter
    cells mean stronger quantum correlations — direct evidence that the
    CNOT ring has created genuine entanglement between the qubits.

    Parameters
    ----------
    hqnn_model  : trained HQNN
    X_val       : validation images
    y_val       : true labels
    max_samples : number of samples to average over (keeps runtime fast)

    Returns
    -------
    (fig, mi_matrix_signal, mi_matrix_noise) — Figure + raw data for both classes
    """
    import pennylane as qml

    q_weights = hqnn_model.get_layer("quantum_layer").q_weights.numpy()
    n_layers, n_qubits, _ = q_weights.shape

    embed_model = tf.keras.Model(
        inputs=hqnn_model.input,
        outputs=hqnn_model.get_layer("embedding").output,
    )

    dev_sv = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev_sv)
    def _state_circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RX(np.pi * inputs[i], wires=i)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.Rot(
                    weights[layer, q, 0],
                    weights[layer, q, 1],
                    weights[layer, q, 2],
                    wires=q,
                )
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return qml.state()

    def _compute_mi_matrix(embeddings):
        """Average QMI matrix over a set of embeddings."""
        mi_acc = np.zeros((n_qubits, n_qubits))
        for emb in embeddings:
            sv = np.array(_state_circuit(emb.astype(float), q_weights), dtype=complex)
            # Single-qubit entropies
            s_single = [_von_neumann_entropy(_partial_trace(sv, [q], n_qubits))
                        for q in range(n_qubits)]
            # Build MI matrix
            for i in range(n_qubits):
                mi_acc[i, i] += s_single[i]
                for j in range(i + 1, n_qubits):
                    rho_ij = _partial_trace(sv, [i, j], n_qubits)
                    s_ij   = _von_neumann_entropy(rho_ij)
                    mi = s_single[i] + s_single[j] - s_ij
                    mi_acc[i, j] += mi
                    mi_acc[j, i] += mi
        return mi_acc / max(len(embeddings), 1)

    # Compute QMI separately for signal and noise events
    n_each = max_samples // 2
    sig_idx   = np.where(y_val == 0)[0][:n_each]
    noise_idx = np.where(y_val == 1)[0][:n_each]

    emb_sig   = embed_model.predict(X_val[sig_idx],   verbose=0)
    emb_noise = embed_model.predict(X_val[noise_idx], verbose=0)

    mi_sig   = _compute_mi_matrix(emb_sig)
    mi_noise = _compute_mi_matrix(emb_noise)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    qnames = [f"Q{i}" for i in range(n_qubits)]

    for ax, mi_mat, title, cmap in [
        (axes[0], mi_sig,   "Signal Events\n(Helicoidal Tracks)",  "Blues"),
        (axes[1], mi_noise, "Background Events\n(Random Deposits)", "Reds"),
    ]:
        im = ax.imshow(mi_mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_qubits)); ax.set_xticklabels(qnames, fontsize=11)
        ax.set_yticks(range(n_qubits)); ax.set_yticklabels(qnames, fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=11, pad=10)

        for i in range(n_qubits):
            for j in range(n_qubits):
                val = mi_mat[i, j]
                label = f"S={val:.2f}" if i == j else f"I={val:.2f}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val > 0.4 else "black")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(
            "Diagonal: Von Neumann entropy S  |  Off-diag: Mutual information I",
            fontsize=7,
        )
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Qubit Correlation Map — Quantum Mutual Information I(i:j) = S(ρᵢ) + S(ρⱼ) − S(ρᵢⱼ)\n"
        "Diagonal = entanglement entropy per qubit  |  Off-diagonal = inter-qubit quantum correlation\n"
        "Brighter cells = stronger quantum entanglement created by the CNOT ring ansatz",
        fontsize=9, y=1.03,
    )
    fig.tight_layout()
    return fig, mi_sig, mi_noise


# ─────────────────────────────────────────────────────────────────────────────
# 9. Hardware Noise Robustness — Qiskit Aer simulation
# ─────────────────────────────────────────────────────────────────────────────

def plot_hardware_noise_robustness(
    qiskit_results: dict,
    hqnn_clean_acc: float,
    cnn_clean_acc: float,
    hqnn_pixel_noise_accs: list[float] | None = None,
    pixel_noise_levels: list[float] | None = None,
) -> plt.Figure:
    """
    Three-panel comparison of quantum hardware noise vs clean baselines.

    Panel 1 — Hardware noise accuracy degradation
        Bar chart: HQNN accuracy at each Qiskit Aer noise level.
        Horizontal dashed lines: clean HQNN (PennyLane) and clean CNN.
        Shows how the trained PQC degrades as gate errors increase.

    Panel 2 — AUC under hardware noise
        Line plot of AUC-ROC at each noise level.
        AUC < 0.5 means the noisy circuit is worse than random.

    Panel 3 — Three-axis comparison table
        Side-by-side bar for: clean HQNN, clean CNN, pixel-noisy HQNN
        (σ=0.2), and Qiskit HQNN at Manila-calibrated level.
        Answers the central question: "quantum noise vs input noise vs clean".

    Parameters
    ----------
    qiskit_results         : output of qiskit_noise.run_hardware_noise_sweep
    hqnn_clean_acc         : PennyLane HQNN accuracy on clean validation set
    cnn_clean_acc          : Classical CNN accuracy on clean validation set
    hqnn_pixel_noise_accs  : list of HQNN accuracies from plot_noise_robustness
                             (indices match pixel_noise_levels)
    pixel_noise_levels     : list of sigma values matching hqnn_pixel_noise_accs
    """
    from qiskit_noise import NOISE_LEVELS, _LEVEL_COLORS

    labels   = list(qiskit_results.keys())
    accs     = [qiskit_results[l]["accuracy"] for l in labels]
    aucs     = [qiskit_results[l]["auc"]      for l in labels]
    colors   = [_LEVEL_COLORS[l]              for l in labels]
    x_pos    = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax_acc, ax_auc, ax_cmp = axes

    # ── Panel 1: Accuracy vs noise level ──────────────────────────────────
    bars = ax_acc.bar(x_pos, accs, color=colors, alpha=0.85, width=0.6, zorder=3)
    for bar, acc in zip(bars, accs):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{acc*100:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax_acc.axhline(hqnn_clean_acc, color=_HQNN_COLOR, ls="--", lw=2.0,
                   label=f"HQNN clean (PL)  {hqnn_clean_acc*100:.1f}%", zorder=4)
    ax_acc.axhline(cnn_clean_acc,  color=_CNN_COLOR,  ls=":",  lw=2.0,
                   label=f"CNN  clean        {cnn_clean_acc*100:.1f}%",  zorder=4)
    ax_acc.axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.6,
                   label="Random baseline 50%", zorder=4)

    ax_acc.set_xticks(x_pos)
    ax_acc.set_xticklabels(labels, fontsize=9)
    ax_acc.set_ylabel("Accuracy", fontsize=10)
    ax_acc.set_ylim(0.3, 1.12)
    ax_acc.set_title(
        "Hardware Noise Accuracy Degradation\n(Qiskit Aer, 2048 shots per circuit)",
        fontweight="bold", fontsize=10,
    )
    ax_acc.legend(fontsize=8, loc="lower left")
    ax_acc.grid(alpha=0.25, axis="y", zorder=0)

    # Shade the accuracy-drop region
    ideal_acc = qiskit_results["Ideal"]["accuracy"]
    ax_acc.fill_between(
        [-0.4, len(labels) - 0.6],
        [ideal_acc, ideal_acc],
        [min(accs), min(accs)],
        alpha=0.05, color="red", zorder=1,
    )

    # ── Panel 2: AUC vs noise level ───────────────────────────────────────
    ax_auc.plot(x_pos, aucs, "o-", color="#9b59b6", lw=2.5, ms=9, zorder=3)
    for xi, (lbl, auc_v) in enumerate(zip(labels, aucs)):
        ax_auc.annotate(
            f"{auc_v:.3f}",
            (xi, auc_v),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=9, fontweight="bold", color="#9b59b6",
        )
    ax_auc.axhline(0.5, color="gray", ls=":", lw=1.2, label="Random (AUC=0.5)")
    ax_auc.set_xticks(x_pos)
    ax_auc.set_xticklabels(labels, fontsize=9)
    ax_auc.set_ylabel("AUC-ROC", fontsize=10)
    ax_auc.set_ylim(0.3, 1.12)
    ax_auc.set_title(
        "AUC-ROC Under Hardware Noise\n(threshold-independent signal detection)",
        fontweight="bold", fontsize=10,
    )
    ax_auc.legend(fontsize=8)
    ax_auc.grid(alpha=0.25, zorder=0)

    # ── Panel 3: Three-axis comparison ────────────────────────────────────
    # Comparison points: clean HQNN, clean CNN, pixel-noise HQNN (σ=0.20),
    # and Qiskit HQNN at Manila-calibrated level
    cmp_labels, cmp_vals, cmp_cols = [], [], []

    cmp_labels.append("HQNN\nclean (PL)")
    cmp_vals.append(hqnn_clean_acc)
    cmp_cols.append(_HQNN_COLOR)

    cmp_labels.append("CNN\nclean")
    cmp_vals.append(cnn_clean_acc)
    cmp_cols.append(_CNN_COLOR)

    if hqnn_pixel_noise_accs is not None and pixel_noise_levels is not None:
        # Pick σ ≈ 0.20 as a representative "moderate" pixel noise point
        target = 0.20
        idx = int(np.argmin(np.abs(np.array(pixel_noise_levels) - target)))
        cmp_labels.append(f"HQNN pixel\nnoise (σ={pixel_noise_levels[idx]:.2f})")
        cmp_vals.append(hqnn_pixel_noise_accs[idx])
        cmp_cols.append("#f39c12")

    manila_acc = qiskit_results.get("Manila-calibr.", {}).get("accuracy", float("nan"))
    cmp_labels.append("HQNN hw\n(Manila-cal.)")
    cmp_vals.append(manila_acc)
    cmp_cols.append(_LEVEL_COLORS["Manila-calibr."])

    cmp_x = np.arange(len(cmp_labels))
    cmp_bars = ax_cmp.bar(cmp_x, cmp_vals, color=cmp_cols, alpha=0.88, width=0.55, zorder=3)
    for bar, val in zip(cmp_bars, cmp_vals):
        if not np.isnan(val):
            ax_cmp.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
    ax_cmp.axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax_cmp.set_xticks(cmp_x)
    ax_cmp.set_xticklabels(cmp_labels, fontsize=9)
    ax_cmp.set_ylabel("Accuracy", fontsize=10)
    ax_cmp.set_ylim(0.3, 1.12)
    ax_cmp.set_title(
        "Three-Axis Comparison\n"
        "Clean QNN vs CNN  |  Pixel noise  |  Hardware noise",
        fontweight="bold", fontsize=10,
    )
    ax_cmp.grid(alpha=0.25, axis="y", zorder=0)

    fig.suptitle(
        "Qiskit Aer Hardware Noise Study — Trained PQC (no retraining) under IBM-calibrated gate errors\n"
        "Left: accuracy degrades with noise  |  Centre: AUC degradation  |  "
        "Right: three-axis comparison (clean / pixel-noise / hardware-noise)",
        fontsize=9, y=1.03,
    )
    fig.tight_layout()
    return fig
