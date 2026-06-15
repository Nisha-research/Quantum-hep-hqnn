"""
main.py
-------
Interactive Streamlit dashboard for the
"Quantum Machine Learning for High-Energy Physics" Final Year Project.

Launch:
    streamlit run main.py --server.port 5000

Sections:
    0. Sidebar        – hyperparameter controls + Train button
    1. Data Preview   – gallery of synthetic particle-track images
    2. Training Dashboard – live progress + accuracy / loss curves
    3. Quantum Benchmarks – comparative metrics table (HQNN vs Classical CNN)
    4. Engineering Report – NISQ era, quantum embedding, barren plateaus
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QML for High-Energy Physics",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (heavy; deferred so the UI renders immediately) ─────────────
@st.cache_resource(show_spinner=False)
def _import_modules():
    """Import ML modules once and cache; avoids reloading on every rerun."""
    from data_generator import generate_dataset, generate_sample
    from trainer import StreamlitProgressCallback, run_experiment
    return generate_dataset, generate_sample, StreamlitProgressCallback, run_experiment


# ── Header ───────────────────────────────────────────────────────────────────
st.title("⚛️ Quantum Machine Learning for High-Energy Physics")
st.markdown(
    "**Final Year Engineering Project** — Hybrid Classical-Quantum Neural Network (HQNN) "
    "for classifying simulated CERN particle collision events."
)
st.divider()

# ── Sidebar: hyperparameters ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Hyperparameters")

    n_epochs = st.slider(
        "Training epochs", min_value=2, max_value=30, value=5, step=1,
        help="Number of full passes through the training data."
    )
    learning_rate = st.select_slider(
        "Learning rate",
        options=[0.001, 0.005, 0.01, 0.02, 0.05],
        value=0.01,
        help="Adam optimiser step size."
    )
    dataset_size = st.slider(
        "Dataset size (total samples)", min_value=40, max_value=300,
        value=80, step=20,
        help=(
            "Total number of images (half signal, half background). "
            "Larger sizes increase accuracy but significantly slow quantum training."
        ),
    )

    st.divider()
    st.markdown(
        "**Note:** Quantum simulation is CPU-intensive. "
        "Keep dataset ≤ 100 and epochs ≤ 10 for fast runs."
    )
    st.divider()

    train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Data Preview
# ─────────────────────────────────────────────────────────────────────────────
st.header("🔬 Section 1 — Simulated Particle Track Gallery")

st.markdown(
    "Each image is a **28 × 28 pixel synthetic detector readout** modelled after the "
    "[CERN TrackML dataset](https://www.kaggle.com/c/trackml-particle-identification). "
    "Pixel brightness represents the **energy deposited** in a detector cell — bright "
    "yellow/white = high energy hit, dark purple/black = no hit. "
    "The HQNN must learn to separate the two classes below."
)

with st.expander("📷 View annotated detector image gallery", expanded=True):
    try:
        generate_dataset, generate_sample, StreamlitProgressCallback, run_experiment = (
            _import_modules()
        )

        # ── Legend / colour-scale explanation ────────────────────────────────
        leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
        leg_col1.metric("⬛ Black pixel", "No hit", "0 energy deposited")
        leg_col2.metric("🟣 Purple pixel", "Weak hit", "Low energy deposit")
        leg_col3.metric("🟠 Orange pixel", "Medium hit", "Moderate energy")
        leg_col4.metric("🟡 Yellow pixel", "Strong hit", "High energy deposit")

        st.divider()

        # ── Part A: Large annotated side-by-side comparison ──────────────────
        st.subheader("A — Annotated Reference Examples")
        st.caption(
            "One canonical signal event and one canonical background event "
            "with key features labelled."
        )

        sig_img  = generate_sample(label=0, seed=7)
        bg_img   = generate_sample(label=1, seed=3)

        fig_ann, (ax_s, ax_b) = plt.subplots(
            1, 2, figsize=(10, 5), facecolor="#0e1117"
        )

        # — Signal panel —
        im_s = ax_s.imshow(sig_img, cmap="inferno", vmin=0, vmax=1)
        ax_s.set_title("[CLASS 0]  Signal Event\n(Helicoidal Particle Track)",
                        color="#00ff88", fontsize=11, fontweight="bold", pad=10)
        ax_s.set_xlabel("Detector column (pixel)", color="#aaaaaa", fontsize=8)
        ax_s.set_ylabel("Detector row (pixel)", color="#aaaaaa", fontsize=8)
        ax_s.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax_s.spines.values():
            spine.set_edgecolor("#555555")

        # Annotations on signal image
        cx, cy = 14, 14   # approximate centre
        ax_s.annotate(
            "Interaction\nvertex\n(origin)",
            xy=(cx, cy), xytext=(2, 3),
            color="cyan", fontsize=7, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="cyan", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc="#001a1a", ec="cyan", alpha=0.8),
        )
        ax_s.annotate(
            "Helicoidal\ncurvature\n(B-field effect)",
            xy=(20, 8), xytext=(17, 0),
            color="lime", fontsize=7, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="lime", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc="#001a00", ec="lime", alpha=0.8),
        )
        ax_s.annotate(
            "Detector\nsmearing\n(finite resolution)",
            xy=(22, 18), xytext=(15, 24),
            color="orange", fontsize=7,
            arrowprops=dict(arrowstyle="->", color="orange", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.2", fc="#1a0a00", ec="orange", alpha=0.8),
        )

        # Colorbar for signal
        cbar_s = fig_ann.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
        cbar_s.set_label("Pixel energy (normalised)", color="#aaaaaa", fontsize=7)
        cbar_s.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=6)
        plt.setp(cbar_s.ax.yaxis.get_ticklabels(), color="#aaaaaa")

        # — Background panel —
        im_b = ax_b.imshow(bg_img, cmap="inferno", vmin=0, vmax=1)
        ax_b.set_title("[CLASS 1]  Background Noise\n(Uncorrelated Energy Deposits)",
                        color="#ff4444", fontsize=11, fontweight="bold", pad=10)
        ax_b.set_xlabel("Detector column (pixel)", color="#aaaaaa", fontsize=8)
        ax_b.set_ylabel("Detector row (pixel)", color="#aaaaaa", fontsize=8)
        ax_b.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax_b.spines.values():
            spine.set_edgecolor("#555555")

        # Annotations on background image
        ax_b.annotate(
            "Isolated hit\n(no track structure)",
            xy=(5, 5), xytext=(8, 1),
            color="red", fontsize=7, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc="#1a0000", ec="red", alpha=0.8),
        )
        ax_b.annotate(
            "Random\ndeposits\n(no spatial order)",
            xy=(18, 22), xytext=(10, 25),
            color="yellow", fontsize=7, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a00", ec="yellow", alpha=0.8),
        )
        ax_b.annotate(
            "No origin\npoint",
            xy=(24, 10), xytext=(18, 6),
            color="violet", fontsize=7,
            arrowprops=dict(arrowstyle="->", color="violet", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.2", fc="#0d001a", ec="violet", alpha=0.8),
        )

        cbar_b = fig_ann.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
        cbar_b.set_label("Pixel energy (normalised)", color="#aaaaaa", fontsize=7)
        cbar_b.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=6)
        plt.setp(cbar_b.ax.yaxis.get_ticklabels(), color="#aaaaaa")

        fig_ann.tight_layout()
        st.pyplot(fig_ann)
        plt.close(fig_ann)

        # — Explanation columns under annotated images —
        info_l, info_r = st.columns(2)
        with info_l:
            st.info(
                "**Class 0 — Signal Event ✅**\n\n"
                "- Track **originates at the interaction vertex** (detector centre)\n"
                "- **Curved trajectory** caused by the solenoidal magnetic field "
                "(Lorentz force on a charged particle)\n"
                "- Hits are **spatially connected** and follow a smooth helix\n"
                "- Detector smearing broadens each hit by ~0.5–1 pixel\n"
                "- Simulates: electron, muon, or pion tracks from a proton–proton collision"
            )
        with info_r:
            st.warning(
                "**Class 1 — Background Noise ❌**\n\n"
                "- Hits are **randomly distributed** across the full detector plane\n"
                "- **No spatial correlation** between deposits\n"
                "- **No origin point** — hits do not trace a continuous path\n"
                "- Variable intensities with no structure\n"
                "- Simulates: thermal noise, detector cross-talk, or soft pile-up "
                "from secondary interactions"
            )

        st.divider()

        # ── Part B: Variety grid — 6 signal + 6 noise ────────────────────────
        st.subheader("B — Variety Grid: 6 Signal vs 6 Background Examples")
        st.caption(
            "Each column shows a distinct generated event. "
            "Notice that **every signal image** has a curved track from the centre, "
            "while **every noise image** looks like random confetti — no shared structure."
        )

        n_each = 6
        fig_grid, axes_grid = plt.subplots(
            2, n_each,
            figsize=(n_each * 2.2, 5),
            facecolor="#0e1117",
        )

        row_labels = ["[0] SIGNAL\n(Helicoidal Track)", "[1] NOISE\n(Random Deposits)"]
        row_colors = ["#00ff88", "#ff4444"]

        for row_idx, (label, row_color) in enumerate(zip([0, 1], row_colors)):
            for col_idx in range(n_each):
                seed_val = col_idx * 7 + row_idx * 31 + 200
                img = generate_sample(label=label, seed=seed_val)
                ax  = axes_grid[row_idx, col_idx]
                ax.imshow(img, cmap="inferno", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(row_color)
                    spine.set_linewidth(2)
                ax.set_title(
                    f"{'Signal' if label == 0 else 'Noise'} #{col_idx + 1}",
                    color=row_color, fontsize=8, pad=3,
                )
            # Row label on the left
            axes_grid[row_idx, 0].set_ylabel(
                row_labels[row_idx], color=row_color,
                fontsize=9, fontweight="bold", labelpad=6,
            )

        fig_grid.suptitle(
            "Inferno colormap:  ■ Black = no energy   ■ Purple = low   "
            "■ Orange = medium   ■ Yellow = high energy",
            color="#aaaaaa", fontsize=8, y=0.02,
        )
        fig_grid.tight_layout(rect=[0, 0.05, 1, 1])
        st.pyplot(fig_grid)
        plt.close(fig_grid)

        # ── Part C: Key visual differences summary ────────────────────────────
        st.subheader("C — How to Visually Distinguish the Two Classes")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(
                "**🎯 Track Origin**\n\n"
                "Signal events always show hits **converging toward the centre** of the "
                "detector (the interaction vertex). Noise events have hits scattered "
                "uniformly across the entire 28×28 grid with no preferred origin."
            )
        with d2:
            st.markdown(
                "**〰️ Spatial Continuity**\n\n"
                "Signal hits form a **connected, curved path** — you can trace the "
                "particle's trajectory from pixel to pixel. Noise hits are isolated "
                "deposits with no neighbour relationship."
            )
        with d3:
            st.markdown(
                "**🌀 Curvature Pattern**\n\n"
                "The magnetic field causes signal tracks to follow a **helical arc** "
                "(tight or loose depending on momentum). Background deposits show no "
                "curvature — they are statistically independent random events."
            )

    except Exception as exc:
        st.error(f"Could not render gallery: {exc}")
        st.exception(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Training Dashboard
# ─────────────────────────────────────────────────────────────────────────────
st.header("📈 Section 2 — Training Dashboard")

hqnn_section_ph  = st.empty()
hqnn_bar_ph      = st.empty()
hqnn_status_ph   = st.empty()
hqnn_chart_ph    = st.empty()

cnn_section_ph   = st.empty()
cnn_bar_ph       = st.empty()
cnn_status_ph    = st.empty()
cnn_chart_ph     = st.empty()

if not train_button:
    st.info("Press **🚀 Train Model** in the sidebar to start training.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Quantum Benchmarks
# ─────────────────────────────────────────────────────────────────────────────
st.header("🏆 Section 3 — Quantum Benchmarks")
benchmark_ph = st.empty()
if not train_button:
    benchmark_ph.info("Train the model first to see benchmark results.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Engineering Report
# ─────────────────────────────────────────────────────────────────────────────
st.header("📚 Section 4 — Engineering Report")

with st.expander("Read the full technical report", expanded=False):
    st.markdown(r"""
## Hybrid Quantum-Classical Neural Networks for Particle Physics

### 1. The NISQ Era & Why It Matters
The **Noisy Intermediate-Scale Quantum (NISQ)** era refers to the current
generation of quantum processors with 50–1 000 qubits that are too noisy to
run fault-tolerant algorithms but sophisticated enough to demonstrate
quantum advantage on targeted problems. High-Energy Physics (HEP) is one of
the most data-intensive sciences on Earth — CERN's LHC produces ~15 PB of
raw collision data per year — making it a natural testbed for quantum
ML methods.

This project targets NISQ-era devices by using a *hybrid* architecture: a
classical GPU-accelerated CNN handles the high-dimensional image preprocessing
(28×28 = 784 features), while a lightweight 4-qubit quantum circuit learns
the final classification boundary. This keeps the quantum workload within the
coherence time budget of near-term devices.

---

### 2. Quantum Feature Embedding (Angle Encoding)
Encoding classical data into quantum states is a non-trivial design decision.
This project uses **Angle Encoding**:

Each feature $x_i \in [-1, 1]$ (from the `tanh` layer) is mapped to a
rotation angle $\theta_i = \pi x_i$ applied via an $R_X$ gate on qubit $i$.
This preserves the continuous, real-valued nature of the CNN embedding while
remaining hardware-efficient (only $n$ single-qubit gates for $n$ features).

| Encoding | Gate depth | Expressibility | Hardware cost |
|---|---|---|---|
| **Angle (used here)** | O(n) | Moderate | Very low |
| Amplitude | O(2ⁿ) | High | Very high |
| Basis | O(n) | Low | Very low |

---

### 3. Parameterised Quantum Circuit (PQC) Ansatz
The variational ansatz consists of L = 2 layers, each comprising:
1. **Rot gates** R(φ, θ, ω) — the most general single-qubit rotation,
   parameterised by 3 Euler angles.
2. **CNOT ring** — entangles adjacent qubits in a ring topology, enabling
   the circuit to learn correlations between features.

Total trainable quantum parameters: L × n × 3 = 2 × 4 × 3 = **24 parameters**.

---

### 4. Mitigating Barren Plateaus
The **Barren Plateau** problem is the central training challenge in quantum
ML: as the number of qubits and circuit depth grows, the gradient of the
cost function with respect to any parameter vanishes *exponentially*.

Mitigation strategies employed here:
- **Shallow ansatz** (L = 2 layers): keeps circuit depth low.
- **Local cost function**: measuring only qubit 0 (PauliZ) avoids global
  observables, which exhibit more severe plateau behaviour (Cerezo et al., 2021).
- **Classical pre-processing**: the CNN reduces 784-dim input to 4 features,
  dramatically reducing the quantum Hilbert space.
- **Structured initialisation**: weights initialised near zero where
  gradients are known to be larger (Grant et al., 2019).

---

### 5. Model Architecture Summary
```
Input (28×28×1)
    → Conv2D(8, 3×3, ReLU) + MaxPool(2×2)
    → Flatten → Dense(4, tanh)          ← 4-dim quantum embedding
    → QuantumLayer(4 qubits, L=2)       ← PennyLane default.qubit
         Angle Encoding: RX(π·xᵢ)
         PQC: Rot + CNOT ring × 2 layers
         Measurement: ⟨Z₀⟩
    → Dense(1, sigmoid)                 ← Binary classification
```

---

### 6. Benchmark Methodology
Both models are trained on identical data splits from the same random seed.
The classical CNN replaces the quantum layer with `Dense(8, relu)` sized to
match the approximate quantum parameter budget. Metrics:
- **Accuracy**: fraction of correctly classified validation samples
- **Precision**: TP / (TP + FP) — penalises false signal identifications
- **Training time**: wall-clock seconds (includes TF graph compilation)

---

*References:*
Cerezo et al. (2021). *Variational Quantum Algorithms*. Nature Reviews Physics.
Farhi & Neven (2018). *Classification with Quantum Neural Networks*. arXiv:1802.06002.
Grant et al. (2019). *An initialization strategy for addressing barren plateaus*. Quantum.
CERN TrackML Challenge — https://www.kaggle.com/c/trackml-particle-identification
""")


# ─────────────────────────────────────────────────────────────────────────────
# Training Logic — triggered by the sidebar button
# ─────────────────────────────────────────────────────────────────────────────
if train_button:
    try:
        from data_generator import generate_dataset as _gen
        from sklearn.model_selection import train_test_split
        from quantum_model import build_hqnn, build_classical_cnn
        from trainer import train_model, StreamlitProgressCallback
        from sklearn.metrics import accuracy_score, precision_score
        import pandas as pd

        # ── Generate data ─────────────────────────────────────────────────
        with st.spinner("Generating synthetic particle-track dataset…"):
            X, y = _gen(n_samples=dataset_size, seed=42)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        st.toast(f"Dataset ready — {len(X_train)} train / {len(X_val)} val samples")

        # ── HQNN training ─────────────────────────────────────────────────
        hqnn_section_ph.markdown("### ⚛️ Hybrid QNN — Training Progress")
        hqnn_bar    = hqnn_bar_ph.progress(0, text="HQNN — initialising…")
        hqnn_status = hqnn_status_ph.empty()
        hqnn_chart  = hqnn_chart_ph.empty()

        hqnn_cb = StreamlitProgressCallback(
            n_epochs=n_epochs,
            progress_bar=hqnn_bar,
            chart_container=hqnn_chart,
            status_text=hqnn_status,
        )

        hqnn_model = build_hqnn(learning_rate=learning_rate)
        hqnn_hist, hqnn_time = train_model(
            hqnn_model, X_train, y_train, X_val, y_val,
            n_epochs=n_epochs,
            extra_callbacks=[hqnn_cb],
        )

        y_pred_hqnn = (hqnn_model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
        hqnn_acc    = accuracy_score(y_val, y_pred_hqnn)
        hqnn_prec   = precision_score(y_val, y_pred_hqnn, zero_division=0)
        hqnn_bar.progress(1.0, text="✅ HQNN training complete!")

        # ── Classical CNN training ────────────────────────────────────────
        cnn_section_ph.markdown("### 🖥️ Classical CNN — Training Progress")
        cnn_bar    = cnn_bar_ph.progress(0, text="Classical CNN — initialising…")
        cnn_status = cnn_status_ph.empty()
        cnn_chart  = cnn_chart_ph.empty()

        cnn_cb = StreamlitProgressCallback(
            n_epochs=n_epochs,
            progress_bar=cnn_bar,
            chart_container=cnn_chart,
            status_text=cnn_status,
        )

        cnn_model = build_classical_cnn(learning_rate=learning_rate)
        cnn_hist, cnn_time = train_model(
            cnn_model, X_train, y_train, X_val, y_val,
            n_epochs=n_epochs,
            extra_callbacks=[cnn_cb],
        )

        y_pred_cnn = (cnn_model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
        cnn_acc    = accuracy_score(y_val, y_pred_cnn)
        cnn_prec   = precision_score(y_val, y_pred_cnn, zero_division=0)
        cnn_bar.progress(1.0, text="✅ Classical CNN training complete!")

        # ── Section 3: Benchmark table ────────────────────────────────────
        benchmark_ph.empty()
        with benchmark_ph.container():
            st.subheader("Model Performance Comparison")

            df = pd.DataFrame(
                {
                    "Metric": ["Test Accuracy", "Test Precision", "Training Time (s)"],
                    "⚛️ Hybrid QNN": [
                        f"{hqnn_acc * 100:.1f}%",
                        f"{hqnn_prec * 100:.1f}%",
                        f"{hqnn_time:.1f}s",
                    ],
                    "🖥️ Classical CNN": [
                        f"{cnn_acc * 100:.1f}%",
                        f"{cnn_prec * 100:.1f}%",
                        f"{cnn_time:.1f}s",
                    ],
                }
            )
            st.table(df.set_index("Metric"))

            # Delta metrics
            acc_delta  = hqnn_acc - cnn_acc
            prec_delta = hqnn_prec - cnn_prec
            time_ratio = hqnn_time / max(cnn_time, 0.001)

            m1, m2, m3 = st.columns(3)
            m1.metric("HQNN Accuracy",  f"{hqnn_acc * 100:.1f}%",
                      delta=f"{acc_delta * 100:+.1f}% vs CNN")
            m2.metric("HQNN Precision", f"{hqnn_prec * 100:.1f}%",
                      delta=f"{prec_delta * 100:+.1f}% vs CNN")
            m3.metric("Training Time Ratio", f"{time_ratio:.1f}×",
                      delta="HQNN / Classical", delta_color="off")

            # Overlay accuracy comparison chart
            st.subheader("Validation Accuracy — Head-to-Head")
            fig_cmp, ax = plt.subplots(figsize=(9, 3.5))
            ep = range(1, n_epochs + 1)
            ax.plot(ep, hqnn_hist["val_accuracy"],
                    "b-o", ms=4, label="HQNN val accuracy")
            ax.plot(ep, cnn_hist["val_accuracy"],
                    "r--s", ms=4, label="CNN val accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Accuracy")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(alpha=0.3)
            fig_cmp.tight_layout()
            st.pyplot(fig_cmp)
            plt.close(fig_cmp)

        st.success(
            "✅ Experiment complete! Scroll up to Section 3 to review the benchmark table."
        )
        st.balloons()

    except Exception as exc:
        st.error(f"Training failed: {exc}")
        st.exception(exc)
