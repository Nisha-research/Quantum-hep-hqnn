# QML for High-Energy Physics

Interactive Final Year Project: Hybrid Classical-Quantum Neural Network (HQNN) for classifying simulated CERN particle collision events.

## Run & Operate

- `streamlit run main.py --server.port 5000` — launch the dashboard
- The workflow "Start application" is already configured and auto-starts.

## Stack

- Python 3.11
- PennyLane 0.45 — quantum circuit simulation (`default.qubit`)
- TensorFlow 2.21 — Keras model training
- Streamlit 1.58 — interactive web dashboard
- Matplotlib, NumPy, scikit-learn

## Where things live

- `data_generator.py` — synthetic CERN particle track image generator (28×28 grayscale)
- `quantum_model.py` — HQNN architecture (CNN + 4-qubit PQC) and classical CNN benchmark
- `trainer.py`        — training loop, Streamlit live-callback, experiment runner
- `main.py`           — Streamlit dashboard (4 sections: gallery, training, benchmarks, report)
- `.streamlit/config.toml` — Streamlit server config (port 5000, headless)

## Architecture decisions

- Angle Encoding (RX gates) maps CNN tanh output [−1,1] to qubit rotations: hardware-efficient and preserves feature range.
- Rot + CNOT ring ansatz (L=2 layers) balances expressibility vs. barren-plateau risk.
- Only PauliZ(qubit 0) is measured (local observable) to further mitigate barren plateaus.
- Classical CNN pre-processes 784→4 dimensions before the quantum circuit, keeping QPU load minimal.
- `default.qubit` CPU simulator requires no real QPU — fully self-contained on Replit.

## Product

- **Section 1** — Visual gallery of signal (helicoidal tracks) vs. noise (random deposits)
- **Section 2** — Live training progress bar + loss/accuracy curves updated per epoch
- **Section 3** — Benchmark table: HQNN vs Classical CNN (accuracy, precision, time)
- **Section 4** — Technical engineering report (NISQ era, embedding, barren plateaus)

## User preferences

_Populate as needed._

## Gotchas

- Keep dataset ≤ 100 and epochs ≤ 10 for fast runs; quantum simulation is CPU-bound.
- TF graph compilation on first training run adds ~10–20 s overhead.
- `st.rerun()` must be used instead of the deprecated `experimental_rerun`.

## Pointers

- PennyLane docs: https://pennylane.ai/
- CERN TrackML dataset: https://www.kaggle.com/c/trackml-particle-identification
