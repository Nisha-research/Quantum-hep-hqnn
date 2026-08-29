# Hybrid Quantum-Classical Neural Network for High-Energy Physics

⚛️ **HQCNN** — an interactive Streamlit demo of a hybrid quantum-classical neural network for classifying simulated CERN particle collision events.

[![GitHub](https://img.shields.io/badge/GitHub-Nisha--research%2FQuantum--hep--hqnn-blue)](https://github.com/Nisha-research/Quantum-hep-hqnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**🌐 Live demo:** https://quantum-hep-hqnn-jkzzfkvnwtepws4fxum9wa.streamlit.app/

## Overview

This final-year BSc Computer Science project demonstrates:

- ⚛️ **Quantum Computing**: 4-qubit variational circuit with PennyLane
- 🔬 **High-Energy Physics**: CERN TrackML-inspired particle classification
- 🤖 **Hybrid Architecture**: Classical CNN + quantum classifier
- 📊 **Benchmarking**: Confusion matrices, ROC curves, noise robustness, Bloch spheres, entanglement maps
- 🌐 **Real Hardware (future scope)**: IBM Quantum integration — offline script, not part of the live demo
- 🎯 **Deployment**: Interactive web demo on Streamlit Community Cloud

## Key Results

*(from one representative run — HQNN/CNN accuracy and timing vary run to run, since the model retrains live each session rather than loading fixed weights)*

| Metric        | HQNN   | CNN    |
| ------------- | ------ | ------ |
| Test Accuracy | 100.0% | 100.0% |
| AUC-ROC       | 1.000  | 1.000  |
| Training Time | 758.7s | 11.3s  |

**Note:** the 250-sample / 20-epoch configuration above is heavier than the app's live-demo defaults (80 samples / 5 epochs) — see Quick Start below. 100% accuracy on both models reflects that the synthetic dataset is easily separable, not a claim of proven quantum advantage (see Known Limitations, in-app Section 4).

## Features

✅ Interactive web demo (Streamlit)
✅ Live in-session training (no pre-saved model weights — see note below)
✅ Synthetic data generation with annotated examples
✅ Confusion matrices, ROC curves, PCA embeddings, noise robustness sweep
✅ Quantum analytics: Bloch spheres, entanglement (mutual information) maps
✅ Qubit-scaling experiment (2/4/6 qubits) with barren-plateau discussion

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the app locally

```bash
streamlit run main.py
# Visit: http://localhost:8501
```

There is no separate `train.py`, `benchmark.py`, or `app.py` — all training, benchmarking, and inference happen interactively inside `main.py` via the Streamlit UI (sidebar controls → "🚀 Train Model" button). Nothing is saved to disk between sessions; each run trains fresh in-browser.

## Architecture

**Classical:** Conv2D(8, 3×3) → MaxPool → Flatten → Dense(4, tanh)
**Quantum:** Angle Encoding (RX) → 2 layers Rot + CNOT ring → Measurement ⟨Z₀⟩
**Output:** Dense(1, sigmoid)

**Frontend/Backend:** single-process Streamlit app — no separate REST API or Flask layer.

## Results Summary

- Signal events: near-zero entanglement (S < 0.025 bits) in one representative run
- Background events: higher entanglement (S ≈ 0.15 bits) in the same run
- Qubit scaling: 2q ≈ 53.3%, 4q ≈ 100%, 6q ≈ 53.3% — **suggestive**, not conclusive, evidence for the barren-plateau effect; based on a single run per qubit count on a small dataset, so treat as indicative rather than statistically confirmed (see in-app Section 3G caveat, and Future Scope below)
- Noise robustness: accuracy holds up to σ=0.2 additive Gaussian pixel noise in the same representative run

Exact numbers vary by training run since nothing is cached/pre-trained — rerun the live demo to reproduce your own instance of these results.

## Hardware Integration (future scope, not part of the live app)

✅ IBM Quantum authentication (circuit prep and backend discovery logic complete)
🔭 Real-hardware job submission is scoped for future work, not currently in progress. Two known items would need addressing before picking this back up:
- Re-authenticating via IBM's free **Open Plan** (separate from the original 30-day Trial, which expired)
- Updating `submit_to_ibm.py`'s `backend.run()` call to the current Sampler/Estimator primitives API, since Qiskit Runtime 0.47 deprecated the call path this project originally targeted

Deprioritized for now in favor of finishing the core classification pipeline and analysis suite, which are complete and reproducible on simulator. See `submit_to_ibm.py` for the existing (currently unused) implementation. This script is **not** imported or triggered by the deployed Streamlit app.

## Reproducibility

- Python 3.11 (pinned via `runtime.txt` for Streamlit Cloud)
- PennyLane ≥ 0.45
- TensorFlow-CPU ≥ 2.21 (CPU-only; no GPU code path in this project)
- pandas, scikit-learn, matplotlib, streamlit — see `requirements.txt` for exact pins
- Qiskit ≥ 1.0 is **only** needed if you run `submit_to_ibm.py` or `qiskit_noise.py` directly — it is not required to run the live app
- Default in-app hyperparameters: 80 samples, 5 epochs, lr=0.01 (adjustable via sidebar sliders up to 300 samples / 30 epochs — the app itself recommends staying ≤100/≤10 for a responsive live demo)

## Future Scope

**Statistical rigor**
- Replace the single train/test split with k-fold cross-validation and report accuracy as mean ± std, not a single point estimate
- Run the qubit-scaling experiment (2/4/6 qubits) across multiple random seeds instead of one run each, to distinguish real barren-plateau signal from run-to-run variance
- Measure gradient variance directly across qubit counts (PennyLane supports this) as a more rigorous barren-plateau diagnostic than final accuracy alone

**Task difficulty**
- Add a harder synthetic dataset variant (overlapping signal/background distributions, added noise) where accuracy doesn't saturate at 100% — this would make any future HQNN vs. CNN comparison actually meaningful, since the current task is too easy to show a difference between them

**Circuit design**
- Amplitude encoding to use the full 2ⁿ-dimensional Hilbert space, vs. the current angle encoding
- Deeper ansatz (L > 2) with entanglement beyond nearest-neighbour ring, paired with error mitigation (zero-noise extrapolation) to keep it viable on NISQ hardware

**Data**
- Real TrackML data with hit-level pre-processing, replacing the fully synthetic dataset used throughout this phase

**Hardware validation**
- Real IBM Quantum job submission (see Hardware Integration section above) — scoped but deprioritized

**Code health**
- Sync `pyproject.toml` with `requirements.txt` (currently `requirements.txt` is the source of truth for the deployed app; `pyproject.toml` still lists plain `tensorflow` and is missing `pandas`)
- Make the Section 3I "circuit discovery" insight text in `main.py` dynamic — derive it from the actual computed `mi_sig`/`mi_noise` values instead of a fixed claim

## References

- Cerezo et al. (2021). Variational Quantum Algorithms. *Nature Reviews Physics* 3, 625–644.
- Di Meglio et al. (2023). Quantum Computing for HEP. *PRX Quantum* 5, 037001.
- Farhi & Neven (2018). Classification with Quantum Neural Networks on Near Term Processors. arXiv:1802.06002.
- Grant et al. (2019). An initialization strategy for addressing barren plateaus. *Quantum* 3, 214.
- Heredge et al. (2021). Quantum machine learning for HEP event classification. arXiv:2101.10949.
- CERN TrackML: https://www.kaggle.com/c/trackml-particle-identification

## License

MIT License

## Author

**Nisha Choudhary** — Final-Year BSc Computer Science Project (2026)
GitHub: [@Nisha-research](https://github.com/Nisha-research)
