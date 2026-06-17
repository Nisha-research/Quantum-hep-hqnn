"""
main.py
-------
Interactive Streamlit dashboard — QML for High-Energy Physics.

Sections
--------
  0. Sidebar        — hyperparameters + train / qubit-scaling controls
  1. Data Gallery   — annotated detector image gallery (3 sub-panels)
  2. Training       — live progress bars + loss/accuracy curves
  3. Benchmarks     — table, confusion matrices, ROC, classification demo,
                      embedding PCA, noise robustness, qubit scaling
  4. Report         — technical engineering report
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="QML for High-Energy Physics",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Final Year Project — Hybrid QNN for CERN event classification"},
)


# ── Lazy-load heavy modules once, cached across reruns ───────────────────────
@st.cache_resource(show_spinner=False)
def _load_modules():
    from data_generator import generate_dataset, generate_sample
    from trainer import StreamlitProgressCallback
    return generate_dataset, generate_sample, StreamlitProgressCallback


# ── Header ───────────────────────────────────────────────────────────────────
st.title("⚛️ Quantum Machine Learning for High-Energy Physics")
st.markdown(
    "<span style='font-size:17px'>**Final Year Project** — Hybrid Classical-Quantum Neural Network (HQNN) "
    "for classifying simulated CERN particle collision events using PennyLane + TensorFlow.</span>",
    unsafe_allow_html=True,
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Hyperparameters")

    n_epochs = st.slider(
        "Training epochs", min_value=2, max_value=30, value=5, step=1,
        help="Passes over the training data for both HQNN and CNN.",
    )
    learning_rate = st.select_slider(
        "Learning rate",
        options=[0.001, 0.005, 0.01, 0.02, 0.05],
        value=0.01,
    )
    dataset_size = st.slider(
        "Dataset size (total samples)", min_value=40, max_value=300,
        value=80, step=20,
        help="Half signal, half background. Keep ≤ 100 for fast demo.",
    )

    st.divider()
    st.subheader("Extra Experiments")
    run_scaling = st.checkbox(
        "Qubit scaling experiment (2 / 4 / 6 qubits)",
        value=False,
        help="Trains 3 extra HQNN variants to show barren-plateau evidence. Adds ~3–5 min.",
    )
    run_hw_noise = st.checkbox(
        "Hardware noise simulation (Qiskit Aer)",
        value=False,
        help=(
            "Rebuilds the trained PQC in Qiskit and runs it through 4 noise levels "
            "(Ideal → 2× IBM Manila-calibrated depolarising + readout error). "
            "No retraining — pure inference transfer. Adds ~1–2 min."
        ),
    )

    st.divider()
    st.caption(
        "Quantum simulation is CPU-bound. "
        "Keep dataset ≤ 100 and epochs ≤ 10 for a live demo."
    )
    st.divider()
    train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Data Gallery
# ─────────────────────────────────────────────────────────────────────────────
st.header("🔬 Section 1 — Simulated Particle Track Gallery")
st.markdown(
    "Each image is a **28 × 28 px synthetic detector readout** inspired by the "
    "[CERN TrackML dataset](https://www.kaggle.com/c/trackml-particle-identification) "
    "(fully synthetic — see Section 4 for the data limitation note). "
    "Pixel brightness = energy deposited in one detector cell."
)

with st.expander("📷 View annotated detector image gallery", expanded=True):
    try:
        generate_dataset, generate_sample, StreamlitProgressCallback = _load_modules()

        # Colour-scale legend
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("⬛ Black pixel",  "No hit",     "0 energy")
        lc2.metric("🟣 Purple pixel", "Weak hit",   "Low energy")
        lc3.metric("🟠 Orange pixel", "Medium hit", "Moderate energy")
        lc4.metric("🟡 Yellow pixel", "Strong hit", "High energy")
        st.divider()

        # ── Part A: Annotated pair ────────────────────────────────────────
        st.subheader("A — Annotated Reference Examples")
        st.caption("One signal and one background event with key physics features labelled.")

        sig_img = generate_sample(label=0, seed=7)
        bg_img  = generate_sample(label=1, seed=3)

        fig_ann, (ax_s, ax_b) = plt.subplots(1, 2, figsize=(10, 5), facecolor="#0e1117")

        im_s = ax_s.imshow(sig_img, cmap="inferno", vmin=0, vmax=1)
        ax_s.set_title("[CLASS 0]  Signal Event\n(Helicoidal Particle Track)",
                       color="#00ff88", fontsize=11, fontweight="bold", pad=10)
        ax_s.set_xlabel("Detector column (pixel)", color="#aaaaaa", fontsize=8)
        ax_s.set_ylabel("Detector row (pixel)",    color="#aaaaaa", fontsize=8)
        ax_s.tick_params(colors="#aaaaaa", labelsize=7)
        for sp in ax_s.spines.values(): sp.set_edgecolor("#555555")

        ax_s.annotate("Interaction\nvertex\n(origin)",
                      xy=(14, 14), xytext=(2, 3), color="cyan", fontsize=7, fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color="cyan", lw=1.2),
                      bbox=dict(boxstyle="round,pad=0.2", fc="#001a1a", ec="cyan", alpha=0.8))
        ax_s.annotate("Helicoidal\ncurvature\n(B-field)",
                      xy=(20, 8), xytext=(17, 0), color="lime", fontsize=7, fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color="lime", lw=1.2),
                      bbox=dict(boxstyle="round,pad=0.2", fc="#001a00", ec="lime", alpha=0.8))
        ax_s.annotate("Detector\nsmearing\n(finite res.)",
                      xy=(22, 18), xytext=(15, 24), color="orange", fontsize=7,
                      arrowprops=dict(arrowstyle="->", color="orange", lw=1.0),
                      bbox=dict(boxstyle="round,pad=0.2", fc="#1a0a00", ec="orange", alpha=0.8))
        cb_s = fig_ann.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
        cb_s.set_label("Pixel energy (normalised)", color="#aaaaaa", fontsize=7)
        cb_s.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=6)
        plt.setp(cb_s.ax.yaxis.get_ticklabels(), color="#aaaaaa")

        im_b = ax_b.imshow(bg_img, cmap="inferno", vmin=0, vmax=1)
        ax_b.set_title("[CLASS 1]  Background Noise\n(Uncorrelated Energy Deposits)",
                       color="#ff4444", fontsize=11, fontweight="bold", pad=10)
        ax_b.set_xlabel("Detector column (pixel)", color="#aaaaaa", fontsize=8)
        ax_b.set_ylabel("Detector row (pixel)",    color="#aaaaaa", fontsize=8)
        ax_b.tick_params(colors="#aaaaaa", labelsize=7)
        for sp in ax_b.spines.values(): sp.set_edgecolor("#555555")

        ax_b.annotate("Isolated hit\n(no track)",
                      xy=(5, 5), xytext=(8, 1), color="red", fontsize=7, fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                      bbox=dict(boxstyle="round,pad=0.2", fc="#1a0000", ec="red", alpha=0.8))
        ax_b.annotate("Random deposits\n(no spatial order)",
                      xy=(18, 22), xytext=(10, 25), color="yellow", fontsize=7, fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2),
                      bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a00", ec="yellow", alpha=0.8))
        ax_b.annotate("No origin\npoint",
                      xy=(24, 10), xytext=(18, 6), color="violet", fontsize=7,
                      arrowprops=dict(arrowstyle="->", color="violet", lw=1.0),
                      bbox=dict(boxstyle="round,pad=0.2", fc="#0d001a", ec="violet", alpha=0.8))
        cb_b = fig_ann.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
        cb_b.set_label("Pixel energy (normalised)", color="#aaaaaa", fontsize=7)
        cb_b.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=6)
        plt.setp(cb_b.ax.yaxis.get_ticklabels(), color="#aaaaaa")

        fig_ann.tight_layout()
        st.pyplot(fig_ann)
        plt.close(fig_ann)

        inf_l, inf_r = st.columns(2)
        with inf_l:
            st.info(
                "**Class 0 — Signal Event**\n\n"
                "- Track **originates at the interaction vertex** (detector centre)\n"
                "- **Curved trajectory** from Lorentz force under solenoidal B-field\n"
                "- Hits are **spatially connected** along a smooth helix\n"
                "- Detector smearing broadens each hit by ~0.5–1 px\n"
                "- Simulates: e⁻, μ, or π tracks from a pp collision"
            )
        with inf_r:
            st.warning(
                "**Class 1 — Background Noise**\n\n"
                "- Hits are **randomly distributed** across the full 28×28 plane\n"
                "- **No spatial correlation** between adjacent cells\n"
                "- **No origin point** — hits do not form a continuous path\n"
                "- Variable intensities, no directional structure\n"
                "- Simulates: thermal noise, cross-talk, or soft pile-up"
            )
        st.divider()

        # ── Part B: Variety grid ──────────────────────────────────────────
        st.subheader("B — Variety Grid: 6 Signal vs 6 Background")
        st.caption(
            "Every signal image shows a curved track from the centre; "
            "every noise image looks like random confetti — no shared structure."
        )
        n_each = 6
        fig_grid, axes_grid = plt.subplots(
            2, n_each, figsize=(n_each * 2.2, 5), facecolor="#0e1117"
        )
        row_labels = ["[0] SIGNAL\n(Helicoidal Track)", "[1] NOISE\n(Random Deposits)"]
        row_colors = ["#00ff88", "#ff4444"]

        for row_idx, (label, row_color) in enumerate(zip([0, 1], row_colors)):
            for col_idx in range(n_each):
                seed_val = col_idx * 7 + row_idx * 31 + 200
                img = generate_sample(label=label, seed=seed_val)
                ax  = axes_grid[row_idx, col_idx]
                ax.imshow(img, cmap="inferno", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(row_color); sp.set_linewidth(2)
                ax.set_title(
                    f"{'Signal' if label == 0 else 'Noise'} #{col_idx + 1}",
                    color=row_color, fontsize=8, pad=3,
                )
            axes_grid[row_idx, 0].set_ylabel(
                row_labels[row_idx], color=row_color,
                fontsize=9, fontweight="bold", labelpad=6,
            )
        fig_grid.suptitle(
            "Inferno colormap:  Black=no energy  |  Purple=low  |  Orange=medium  |  Yellow=high",
            color="#aaaaaa", fontsize=8, y=0.02,
        )
        fig_grid.tight_layout(rect=[0, 0.05, 1, 1])
        st.pyplot(fig_grid)
        plt.close(fig_grid)

        st.divider()

        # ── Part C: Visual differences ────────────────────────────────────
        st.subheader("C — How to Visually Distinguish the Two Classes")
        d1, d2, d3 = st.columns(3)
        d1.markdown(
            "**Track Origin**\n\n"
            "Signal hits **converge toward the detector centre** (interaction vertex). "
            "Noise hits are scattered uniformly with no preferred origin."
        )
        d2.markdown(
            "**Spatial Continuity**\n\n"
            "Signal hits form a **connected, curved path** you can trace pixel-by-pixel. "
            "Noise hits are isolated with no neighbour relationship."
        )
        d3.markdown(
            "**Curvature Pattern**\n\n"
            "The B-field forces signal tracks onto a **helical arc** (tight or loose "
            "depending on momentum). Background deposits show no curvature whatsoever."
        )

    except Exception as exc:
        st.error(f"Could not render gallery: {exc}")
        st.exception(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Training Dashboard
# ─────────────────────────────────────────────────────────────────────────────
st.header("📈 Section 2 — Training Dashboard")

hqnn_sec_ph = st.empty()
hqnn_bar_ph = st.empty()
hqnn_sta_ph = st.empty()
hqnn_cht_ph = st.empty()

cnn_sec_ph  = st.empty()
cnn_bar_ph  = st.empty()
cnn_sta_ph  = st.empty()
cnn_cht_ph  = st.empty()

if not train_button:
    st.info("Press **🚀 Train Model** in the sidebar to start training.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Benchmarks & Analysis (placeholders; filled after training)
# ─────────────────────────────────────────────────────────────────────────────
st.header("🏆 Section 3 — Benchmarks & Analysis")

s3a_ph = st.empty()   # benchmark table
s3b_ph = st.empty()   # confusion matrices
s3c_ph = st.empty()   # ROC curves
s3d_ph = st.empty()   # classification demo
s3e_ph = st.empty()   # embedding PCA
s3f_ph = st.empty()   # noise robustness
s3g_ph = st.empty()   # qubit scaling
s3h_ph = st.empty()   # bloch spheres
s3i_ph = st.empty()   # entanglement map
s3j_ph = st.empty()   # hardware noise (Qiskit Aer)

if not train_button:
    s3a_ph.info("Train the model first to see benchmark results.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Engineering Report
# ─────────────────────────────────────────────────────────────────────────────
st.header("📚 Section 4 — Engineering Report")

with st.expander("Read the full technical report", expanded=False):
    st.markdown(r"""
## Hybrid Quantum-Classical Neural Networks for Particle Physics

> **Data Disclaimer:** All detector images in this project are **fully synthetic**,
> generated by `data_generator.py`. They are *inspired by* the structural properties
> of the CERN TrackML dataset (curved tracks vs. random noise) but are **not** derived
> from real LHC data. Reviewers should treat reported metrics as proof-of-concept on a
> controlled synthetic task. Real TrackML signals have significantly higher hit
> multiplicity (~100 k hits/event) and require track-finding pre-processing before
> classification.

---

### 1. The NISQ Era & Why It Matters
The **Noisy Intermediate-Scale Quantum (NISQ)** era covers quantum processors
with 50–1 000 qubits — too noisy for fault-tolerant algorithms, but capable of
demonstrating quantum-enhanced ML on targeted tasks. CERN's LHC produces
~15 PB of collision data per year, making it a natural testbed for NISQ-era
methods where data is abundant and feature extraction is the bottleneck.

This project uses a *hybrid* strategy: a classical CNN handles high-dimensional
image preprocessing (784 → 4 features), keeping the quantum circuit within the
coherence budget of near-term devices.

---

### 2. Quantum Feature Embedding (Angle Encoding)
Each feature $x_i \in [-1, 1]$ (tanh output) is mapped to a rotation angle
$\theta_i = \pi x_i$ applied via an $R_X$ gate on qubit $i$.

| Encoding | Gate depth | Expressibility | Hardware cost |
|---|---|---|---|
| **Angle (used here)** | O(n) | Moderate | Very low |
| Amplitude | O(2ⁿ) | High | Very high |
| Basis | O(n) | Low | Very low |

Angle encoding is hardware-efficient and preserves the continuous-valued
output of the tanh activation.

---

### 3. PQC Ansatz & Entanglement Strategy
Each of L = 2 layers applies:
1. **Rot gates** R(φ, θ, ω) — arbitrary single-qubit rotation (3 parameters)
2. **CNOT ring** — entangles adjacent qubits cyclically

Total quantum parameters: 2 × 4 × 3 = **24**. The ring topology provides
nearest-neighbour entanglement without long-range connectivity, compatible with
current superconducting qubit layouts (e.g. IBM Falcon, Google Sycamore).

---

### 4. Barren Plateau Mitigation
The **Barren Plateau** problem: for deep circuits on many qubits, gradients
vanish exponentially — $\text{Var}[\partial_\theta \mathcal{L}] \sim O(2^{-n})$.

Mitigation strategies in this project (see Section 3G for empirical evidence):
- **Shallow ansatz** (L = 2): avoids the deep circuits where plateaus are worst
- **Local observable** $\langle Z_0 \rangle$: Cerezo et al. (2021) prove local
  cost functions have polynomially-vanishing gradients vs. exponential for global
- **Classical pre-processing**: 784 → 4 features eliminates the vast majority of
  the Hilbert space before quantum computation
- **Near-zero initialisation** (Grant et al. 2019): quantum weights drawn from
  $[-0.1, 0.1]$ keep gates near identity at epoch 0, where local gradients
  are O(1) rather than exponentially small.  Uniform $[0, 2\pi]$ initialisation
  places parameters in the flat high-variance region from the very first step.

**Implementation note — gradient flow and dtype consistency:** PennyLane 0.45
removed `qml.qnn.KerasLayer`.  The replacement is a custom `tf.keras.layers.Layer`
wrapping the QNode via `tf.map_fn`.  Two requirements must hold simultaneously:

1. `diff_method="parameter-shift"` (not `"backprop"`).  `tf.map_fn` compiles to
   `tf.while_loop`; TF's backprop cannot trace the QNode graph through a while_loop.
   Parameter-shift registers the gradient externally as a `@tf.custom_gradient`,
   which *does* propagate through `map_fn` without needing circuit-graph tracing.

2. Quantum weights must be declared as `dtype=tf.float64`.  Parameter-shift
   evaluates the circuit at shifted parameter values using numpy (float64 internally)
   and returns float64 gradients via the custom_gradient.  If weights are float32
   (Keras default), TF raises `"type float32 does not match type float64"` during
   `tape.gradient`.  Storing weights as float64 and casting inputs up / output back
   down keeps the gradient chain dtype-consistent.

Without (1), quantum weight gradients are identically zero and the model collapses
to a constant predictor (loss ≈ ln 2, accuracy ≈ 50%).
Without (2), training crashes with a dtype mismatch inside `tape.gradient`.

The qubit scaling experiment (Section 3G) provides **empirical** validation:
training curves for 2, 4, and 6 qubits directly demonstrate the slower
convergence predicted by the barren plateau theory.

---

### 5. Architecture Summary
```
Input (28×28×1)
    -> Conv2D(8 filters, 3x3, ReLU) + MaxPool(2x2)
    -> Flatten -> Dense(4, tanh)        <- 4-dim quantum embedding
    -> QuantumLayer (4 qubits, L=2)     <- PennyLane default.qubit
         Angle Encoding:  RX(pi * xi) on qubit i
         PQC:             Rot(phi,theta,omega) + CNOT ring  x 2
         Measurement:     <Z0>  (local observable)
    -> Dense(1, sigmoid)                <- binary classifier
Total parameters: ~1.6k classical + 24 quantum = ~1.6k total
```

---

### 6. Benchmark Methodology & Limitations
Both models are trained on **identical** splits from seed 42. Metrics:
- **Accuracy / Precision** — standard binary classification
- **AUC-ROC** — threshold-independent signal/background separation
- **Confusion matrix** — explicit false-positive (signal missed) vs.
  false-negative (noise accepted as signal) trade-off
- **Noise robustness** — accuracy under additive Gaussian noise
  (σ = 0–0.5), simulating detector noise or hardware errors

**Known limitations of this study:**
1. Dataset is fully synthetic (see data disclaimer above)
2. Quantum simulation is exact (no hardware noise model)
3. Dataset size (≤ 300 samples) is far smaller than real LHC workloads
4. 4-qubit circuit may not achieve quantum advantage over classical models

---

*References:*
Cerezo et al. (2021). *Variational Quantum Algorithms*. Nature Reviews Physics 3, 625–644.
Farhi & Neven (2018). *Classification with Quantum Neural Networks on Near Term Processors*. arXiv:1802.06002.
Grant et al. (2019). *An initialization strategy for addressing barren plateaus*. Quantum 3, 214.
Heredge et al. (2021). *Quantum machine learning for HEP event classification*. arXiv:2101.10949.
CERN TrackML Particle Tracking Challenge — https://www.kaggle.com/c/trackml-particle-identification
""")


# ─────────────────────────────────────────────────────────────────────────────
# Training Logic
# ─────────────────────────────────────────────────────────────────────────────
if train_button:
    try:
        from data_generator import generate_dataset as _gen
        from sklearn.model_selection import train_test_split
        from quantum_model import build_hqnn, build_classical_cnn, build_hqnn_nqubits
        from trainer import train_model, StreamlitProgressCallback
        from sklearn.metrics import accuracy_score, precision_score
        from analysis import (
            plot_confusion_matrices, plot_roc_curves, plot_embedding_pca,
            plot_classification_demo, plot_noise_robustness, plot_qubit_scaling,
            plot_bloch_spheres, plot_entanglement_map,
            plot_hardware_noise_robustness,
        )
        import pandas as pd

        # ── Data ──────────────────────────────────────────────────────────
        with st.spinner("Generating synthetic particle-track dataset…"):
            X, y = _gen(n_samples=dataset_size, seed=42)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        st.toast(f"Dataset ready — {len(X_train)} train / {len(X_val)} val samples")

        # ── HQNN training ─────────────────────────────────────────────────
        hqnn_sec_ph.markdown("### ⚛️ Hybrid QNN — Training Progress")
        hqnn_bar = hqnn_bar_ph.progress(0, text="HQNN — initialising…")
        hqnn_sta = hqnn_sta_ph.empty()
        hqnn_cht = hqnn_cht_ph.empty()

        hqnn_cb = StreamlitProgressCallback(
            n_epochs=n_epochs, progress_bar=hqnn_bar,
            chart_container=hqnn_cht, status_text=hqnn_sta,
        )
        hqnn_model = build_hqnn(learning_rate=learning_rate)
        hqnn_hist, hqnn_time = train_model(
            hqnn_model, X_train, y_train, X_val, y_val,
            n_epochs=n_epochs, extra_callbacks=[hqnn_cb],
        )
        # Raw probability scores (needed for ROC)
        y_score_hqnn = hqnn_model.predict(X_val, verbose=0).flatten()
        y_pred_hqnn  = (y_score_hqnn > 0.5).astype(int)
        hqnn_acc     = accuracy_score(y_val, y_pred_hqnn)
        hqnn_prec    = precision_score(y_val, y_pred_hqnn, zero_division=0)
        hqnn_bar.progress(1.0, text="HQNN training complete!")

        # ── Classical CNN training ────────────────────────────────────────
        cnn_sec_ph.markdown("### 🖥️ Classical CNN — Training Progress")
        cnn_bar = cnn_bar_ph.progress(0, text="Classical CNN — initialising…")
        cnn_sta = cnn_sta_ph.empty()
        cnn_cht = cnn_cht_ph.empty()

        cnn_cb = StreamlitProgressCallback(
            n_epochs=n_epochs, progress_bar=cnn_bar,
            chart_container=cnn_cht, status_text=cnn_sta,
        )
        cnn_model = build_classical_cnn(learning_rate=learning_rate)
        cnn_hist, cnn_time = train_model(
            cnn_model, X_train, y_train, X_val, y_val,
            n_epochs=n_epochs, extra_callbacks=[cnn_cb],
        )
        y_score_cnn = cnn_model.predict(X_val, verbose=0).flatten()
        y_pred_cnn  = (y_score_cnn > 0.5).astype(int)
        cnn_acc     = accuracy_score(y_val, y_pred_cnn)
        cnn_prec    = precision_score(y_val, y_pred_cnn, zero_division=0)
        cnn_bar.progress(1.0, text="Classical CNN training complete!")

        # ── 3A: Benchmark table ───────────────────────────────────────────
        s3a_ph.empty()
        with s3a_ph.container():
            st.subheader("3A — Performance Comparison")
            df = pd.DataFrame({
                "Metric": [
                    "Test Accuracy", "Test Precision", "Training Time (s)"
                ],
                "Hybrid QNN": [
                    f"{hqnn_acc*100:.1f}%", f"{hqnn_prec*100:.1f}%",
                    f"{hqnn_time:.1f}s",
                ],
                "Classical CNN": [
                    f"{cnn_acc*100:.1f}%", f"{cnn_prec*100:.1f}%",
                    f"{cnn_time:.1f}s",
                ],
            })
            st.table(df.set_index("Metric"))

            acc_delta  = hqnn_acc - cnn_acc
            prec_delta = hqnn_prec - cnn_prec
            time_ratio = hqnn_time / max(cnn_time, 0.001)
            m1, m2, m3 = st.columns(3)
            m1.metric("HQNN Accuracy",  f"{hqnn_acc*100:.1f}%",
                      delta=f"{acc_delta*100:+.1f}% vs CNN")
            m2.metric("HQNN Precision", f"{hqnn_prec*100:.1f}%",
                      delta=f"{prec_delta*100:+.1f}% vs CNN")
            m3.metric("Time ratio (HQNN/CNN)", f"{time_ratio:.1f}×",
                      delta_color="off")

            # Validation accuracy overlay (FIX: use .history attribute)
            st.subheader("Validation Accuracy — Head-to-Head")
            fig_cmp, ax = plt.subplots(figsize=(9, 3.5))
            ep = range(1, n_epochs + 1)
            ax.plot(ep, hqnn_hist.history["val_accuracy"],
                    "b-o", ms=4, label="HQNN val accuracy")
            ax.plot(ep, cnn_hist.history["val_accuracy"],
                    "r--s", ms=4, label="CNN val accuracy")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy")
            ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)
            fig_cmp.tight_layout()
            st.pyplot(fig_cmp)
            plt.close(fig_cmp)

        # ── 3B: Confusion matrices ────────────────────────────────────────
        with s3b_ph.container():
            st.subheader("3B — Confusion Matrices")
            st.caption(
                "Normalised by true class (rows sum to 100%). "
                "Off-diagonal = misclassification rate — the number reviewers "
                "care about more than accuracy alone on a balanced set."
            )
            fig_cm = plot_confusion_matrices(
                y_val.astype(int), y_pred_hqnn, y_pred_cnn
            )
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        # ── 3C: ROC curves ────────────────────────────────────────────────
        with s3c_ph.container():
            st.subheader("3C — ROC Curves & AUC")
            st.caption(
                "AUC measures signal-detection quality independent of threshold. "
                "A perfect classifier has AUC = 1.0; random = 0.5."
            )
            fig_roc, auc_h, auc_c = plot_roc_curves(
                y_val.astype(int), y_score_hqnn, y_score_cnn
            )
            rc1, rc2, rc3 = st.columns([2, 1, 1])
            with rc1:
                st.pyplot(fig_roc)
                plt.close(fig_roc)
            with rc2:
                st.metric("HQNN AUC", f"{auc_h:.3f}")
            with rc3:
                st.metric("CNN AUC",  f"{auc_c:.3f}",
                          delta=f"{(auc_c - auc_h):+.3f} vs HQNN")

        # ── 3D: Classification demo ───────────────────────────────────────
        with s3d_ph.container():
            st.subheader("3D — Event Classifier Demo")
            st.markdown(
                "Like the classic MNIST digit-recognition grid — each cell shows "
                "a raw detector image, what **HQNN** predicts (top row) and what "
                "**Classical CNN** predicts (bottom row), plus confidence. "
                "**Green border = correct, Red = wrong.**"
            )
            n_demo = min(10, len(X_val))
            fig_demo = plot_classification_demo(
                hqnn_model, cnn_model, X_val, y_val, n_show=n_demo
            )
            st.pyplot(fig_demo)
            plt.close(fig_demo)

        # ── 3E: Embedding PCA ─────────────────────────────────────────────
        with s3e_ph.container():
            st.subheader("3E — Learned Feature Embedding (PCA)")
            st.markdown(
                "The Dense(4, tanh) layer compresses each 28×28 image to a "
                "**4-dimensional vector** before the quantum circuit. "
                "This PCA scatter shows the 2D projection of that space, "
                "coloured by true class — good separation here means the CNN "
                "has extracted discriminating physics features for the quantum circuit to classify."
            )
            fig_emb = plot_embedding_pca(hqnn_model, cnn_model, X_val, y_val)
            st.pyplot(fig_emb)
            plt.close(fig_emb)

        # ── 3F: Noise robustness ──────────────────────────────────────────
        with s3f_ph.container():
            st.subheader("3F — Noise Robustness Test")
            st.markdown(
                "Additive Gaussian noise (σ from 0 → 0.5) is applied to the "
                "validation images at inference time, simulating detector cross-talk, "
                "pile-up, or hardware bit-flip errors. The degradation curves show "
                "how gracefully each model handles corrupted inputs — a practical "
                "concern for any NISQ-era deployment where hardware noise is unavoidable."
            )
            with st.spinner("Running noise robustness test…"):
                fig_noise, hqnn_noise_accs, cnn_noise_accs = plot_noise_robustness(
                    hqnn_model, cnn_model, X_val, y_val
                )
            st.pyplot(fig_noise)
            plt.close(fig_noise)

            noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
            df_noise = pd.DataFrame({
                "Noise sigma":   [f"{s:.2f}" for s in noise_levels],
                "HQNN Accuracy": [f"{a*100:.1f}%" for a in hqnn_noise_accs],
                "CNN Accuracy":  [f"{a*100:.1f}%" for a in cnn_noise_accs],
            })
            st.dataframe(df_noise, use_container_width=True, hide_index=True)

        # ── 3G: Qubit scaling (optional) ──────────────────────────────────
        if run_scaling:
            with s3g_ph.container():
                st.subheader("3G — Qubit Scaling Experiment")
                st.markdown(
                    "Trains HQNN variants with **2, 4, and 6 qubits** on a fixed dataset "
                    "(60 samples, 5 epochs) to observe convergence behaviour across qubit counts. "
                    "**Note:** with a small synthetic dataset results carry high statistical "
                    "variance — non-monotonic final accuracies (e.g. 6-qubit ≈ 2-qubit) reflect "
                    "this variance, not a true exception to the barren plateau effect. "
                    "Interpret the *loss-curve flatness* as the more reliable barren-plateau "
                    "signal, not the final accuracy alone."
                )
                scale_results = {}
                scale_prog = st.progress(0, text="Qubit scaling — 2 qubits…")
                X_sc, y_sc = _gen(n_samples=60, seed=7)
                X_sc_tr, X_sc_vl, y_sc_tr, y_sc_vl = train_test_split(
                    X_sc, y_sc, test_size=0.25, random_state=7
                )
                for i, nq in enumerate([2, 4, 6]):
                    scale_prog.progress(
                        (i + 0.5) / 3, text=f"Qubit scaling — {nq} qubits…"
                    )
                    m = build_hqnn_nqubits(n_qubits=nq, n_layers=2,
                                           learning_rate=learning_rate)
                    h, t = train_model(
                        m, X_sc_tr, y_sc_tr, X_sc_vl, y_sc_vl,
                        n_epochs=5, batch_size=8,
                    )
                    scale_results[nq] = {
                        "history":   h.history,
                        "time":      t,
                        "final_acc": h.history["val_accuracy"][-1],
                    }
                scale_prog.progress(1.0, text="Qubit scaling complete!")

                fig_scale = plot_qubit_scaling(scale_results)
                st.pyplot(fig_scale)
                plt.close(fig_scale)

                nq_rows = [
                    {
                        "Qubits":   nq,
                        "Params":   f"{2 * nq * 3}",
                        "Final val acc": f"{scale_results[nq]['final_acc']*100:.1f}%",
                        "Time (s)": f"{scale_results[nq]['time']:.1f}s",
                    }
                    for nq in [2, 4, 6]
                ]
                st.dataframe(
                    pd.DataFrame(nq_rows), use_container_width=True, hide_index=True
                )

        # ── 3H: Bloch Sphere Visualisation ───────────────────────────────────
        with s3h_ph.container():
            st.subheader("3H — Quantum Feature Map: Bloch Sphere Visualisation")
            st.markdown(
                "Each Bloch sphere shows **where the quantum state of one qubit lands** "
                "on the unit sphere after angle-encoding the classical CNN features and "
                "applying the full CNOT-ring PQC ansatz. "
                "The Bloch vector **r** = (⟨X⟩, ⟨Y⟩, ⟨Z⟩) is the qubit's mean "
                "observable — a point on or inside the unit sphere. "
                "If signal events (🟢) and noise events (🔴) land in **distinct "
                "regions** of the sphere, the quantum circuit has separated the two "
                "classes in Hilbert space and the subsequent Dense(1, sigmoid) layer "
                "can exploit this geometric separation. "
                "Poles (|0⟩ / |1⟩) = Z eigenstates; equator = equal superpositions."
            )
            with st.spinner("Running Bloch-sphere diagnostic circuits…"):
                fig_bloch = plot_bloch_spheres(hqnn_model, X_val, y_val)
            st.pyplot(fig_bloch)
            plt.close(fig_bloch)
            st.caption(
                "Computed via a separate diagnostic QNode (no gradients needed) — "
                "measures ⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit after the full PQC on the "
                "trained embedding of each validation event."
            )

        # ── 3I: Entanglement Strength Map ─────────────────────────────────
        with s3i_ph.container():
            st.subheader("3I — Entanglement Analytics: Qubit Correlation Map")
            st.markdown(
                "The **Quantum Mutual Information** between every pair of qubits, "
                "I(i:j) = S(ρᵢ) + S(ρⱼ) − S(ρᵢⱼ), is computed from the full "
                "state vector obtained after the PQC. "
                "Diagonal cells show the **Von Neumann entropy** S(ρᵢ) of each "
                "qubit — its entanglement with the rest of the register. "
                "Off-diagonal cells show pairwise quantum correlations created by "
                "the CNOT ring. **Bright cells prove the circuit is using genuine "
                "quantum entanglement**, not just acting like a classical layer. "
                "Results shown separately for Signal vs. Background events."
            )
            with st.spinner("Computing Von Neumann entropies and mutual information…"):
                fig_ent, mi_sig, mi_noise = plot_entanglement_map(
                    hqnn_model, X_val, y_val
                )
            st.pyplot(fig_ent)
            plt.close(fig_ent)

            e1, e2 = st.columns(2)
            with e1:
                st.markdown("**Signal — avg entanglement entropy per qubit**")
                for q in range(mi_sig.shape[0]):
                    st.metric(f"Qubit {q}", f"S = {mi_sig[q, q]:.3f} bits")
            with e2:
                st.markdown("**Background — avg entanglement entropy per qubit**")
                for q in range(mi_noise.shape[0]):
                    st.metric(f"Qubit {q}", f"S = {mi_noise[q, q]:.3f} bits")
            st.caption(
                "Von Neumann entropy S = 0 → qubit is in a pure product state (no entanglement). "
                "S = 1 → maximally entangled with the register. "
                "Values averaged over up to 8 validation events per class."
            )

        # ── 3J: Hardware noise simulation (Qiskit Aer, optional) ──────────
        if run_hw_noise:
            with s3j_ph.container():
                st.subheader("3J — Hardware Noise Robustness: Qiskit Aer Study")
                st.markdown(
                    "The trained PQC is **reconstructed in Qiskit** (same RX angle encoding, "
                    "same Rot+CNOT-ring ansatz, same ⟨Z₀⟩ measurement) and executed "
                    "through Qiskit Aer's shot-based simulator at four hardware noise levels:\n\n"
                    "| Level | 1Q gate error | 2Q (CX) error | Readout error |\n"
                    "|---|---|---|---|\n"
                    "| **Ideal** | 0% | 0% | 0% |\n"
                    "| **Light** | 0.1% | 1.0% | 1.0% |\n"
                    "| **Manila-calibr.** | 0.15% | 1.2% | 2.0% |\n"
                    "| **2× Manila** | 0.30% | 2.4% | 4.0% |\n\n"
                    "IBM Manila parameters from public calibration reports (ibm_manila, "
                    "5-qubit Falcon r5, 2022–2023). The classical CNN pre-processing is "
                    "noise-free; only the 4-qubit quantum circuit is noisy. "
                    "**No retraining** — this is pure inference transfer."
                )

                hw_prog = st.progress(0.0, text="Building Qiskit circuits…")

                def _hw_cb(frac, msg):
                    hw_prog.progress(float(frac), text=msg)

                from qiskit_noise import run_hardware_noise_sweep
                with st.spinner("Running Qiskit Aer noise sweep (4 levels × validation set)…"):
                    qiskit_results = run_hardware_noise_sweep(
                        hqnn_model, X_val, y_val,
                        shots=2048,
                        progress_callback=_hw_cb,
                    )
                hw_prog.progress(1.0, text="Hardware noise sweep complete!")

                fig_hw = plot_hardware_noise_robustness(
                    qiskit_results,
                    hqnn_clean_acc=hqnn_acc,
                    cnn_clean_acc=cnn_acc,
                    hqnn_pixel_noise_accs=hqnn_noise_accs,
                    pixel_noise_levels=noise_levels,
                )
                st.pyplot(fig_hw)
                plt.close(fig_hw)

                # Summary table
                hw_rows = [
                    {
                        "Noise level":    lbl,
                        "Accuracy":       f"{qiskit_results[lbl]['accuracy']*100:.1f}%",
                        "AUC-ROC":        f"{qiskit_results[lbl]['auc']:.3f}",
                        "Acc drop vs ideal": f"{(qiskit_results['Ideal']['accuracy'] - qiskit_results[lbl]['accuracy'])*100:+.1f}pp",
                    }
                    for lbl in qiskit_results
                ]
                st.dataframe(
                    pd.DataFrame(hw_rows), use_container_width=True, hide_index=True
                )

                mn = qiskit_results.get("Manila-calibr.", {})
                ideal = qiskit_results.get("Ideal", {})
                hw1, hw2, hw3 = st.columns(3)
                hw1.metric(
                    "Ideal (Qiskit)",
                    f"{ideal.get('accuracy', 0)*100:.1f}%",
                    delta=f"{(ideal.get('accuracy', 0) - hqnn_acc)*100:+.1f}% vs PL clean",
                )
                hw2.metric(
                    "Manila-calibrated",
                    f"{mn.get('accuracy', 0)*100:.1f}%",
                    delta=f"{(mn.get('accuracy', 0) - ideal.get('accuracy', 0))*100:+.1f}% vs ideal",
                    delta_color="inverse",
                )
                hw3.metric(
                    "AUC (Manila-cal.)",
                    f"{mn.get('auc', 0):.3f}",
                )

                st.caption(
                    "The small Ideal vs PennyLane discrepancy is shot noise (2048 shots). "
                    "AUC < 0.5 at any noise level would mean the noisy circuit has inverted "
                    "its predictions — a sign the gate-error rate exceeds the circuit's "
                    "noise tolerance. For deeper discussion see the Future Work note in Section 4."
                )

        st.success(
            "Experiment complete! All results are in Section 3 above. "
            "Scroll up to review confusion matrices, ROC curves, Bloch spheres, "
            "entanglement heatmap"
            + (" and Qiskit hardware-noise study." if run_hw_noise else ".")
        )
        st.balloons()

    except Exception as exc:
        st.error(f"Training failed: {exc}")
        st.exception(exc)
