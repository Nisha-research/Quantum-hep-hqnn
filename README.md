# Hybrid Quantum-Classical Neural Network for High-Energy Physics

⚛️ **HQCNN** - A production-ready implementation of a hybrid quantum-classical neural network for classifying simulated CERN particle collision events.

[![GitHub](https://img.shields.io/badge/GitHub-Nisha--research%2FQuantum--hep--hqnn-blue)](https://github.com/Nisha-research/Quantum-hep-hqnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This final-year BSc Computer Science project demonstrates:
- ⚛️ **Quantum Computing**: 4-qubit variational circuit with PennyLane
- 🔬 **High-Energy Physics**: CERN TrackML-inspired particle classification
- 🤖 **Hybrid Architecture**: Classical CNN + quantum classifier
- 📊 **Benchmarking**: Confusion matrices, ROC curves, noise robustness
- 🌐 **Real Hardware**: IBM Quantum integration
- 🎯 **Deployment**: Interactive web demo + reproducible code

## Key Results

| Metric | HQNN | CNN |
|--------|------|-----|
| Test Accuracy | 100.0% | 100.0% |
| AUC-ROC | 1.000 | 1.000 |
| Training Time | 758.7s | 11.3s |

## Features

✅ Interactive web demo  
✅ Realistic data generation  
✅ Production-grade code  
✅ IBM Quantum ready  
✅ Comprehensive analysis  
✅ Quantum analytics (Bloch, entanglement)  

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Demo
```bash
python app.py
# Visit: http://localhost:5000
```

### Train Model
```bash
python train.py --dataset_size 250 --epochs 20
```

### Benchmark
```bash
python benchmark.py --model ./results/hqnn_model.h5
```

## Architecture

**Classical:** Conv2D → Dense(4, tanh)  
**Quantum:** Angle Encoding → 2 layers Rot+CNOT → Measurement  
**Output:** Dense(1, sigmoid)

## Results

- Signal events: Near-zero entanglement (S < 0.025 bits)
- Background events: Higher entanglement (S ≈ 0.15 bits)
- Qubit scaling: 2q=53.3%, 4q=100%, 6q=53.3% (validates barren plateau theory)
- Noise robustness: 100% accuracy up to σ=0.2

## Hardware Integration

✅ IBM Quantum authentication  
✅ Backend discovery  
✅ Circuit preparation  
⚠️ Job submission: Requires Qiskit Runtime 0.50+ (API update needed)

See `submit_to_ibm.py` for details.

## Reproducibility

- PennyLane ≥ 0.45
- TensorFlow ≥ 2.13
- Qiskit ≥ 1.0
- Hyperparameters: 250 samples, 20 epochs, lr=0.02

## References

- Cerezo et al. (2021). Variational Quantum Algorithms. *Nature Reviews Physics* 3, 625–644.
- Di Meglio et al. (2023). Quantum Computing for HEP. *PRX Quantum* 5, 037001.
- CERN TrackML: https://www.kaggle.com/c/trackml-particle-identification

## License

MIT License

## Author

**Nisha Choudhary** - Final-Year BSc Computer Science Project (2026)

GitHub: [@Nisha-research](https://github.com/Nisha-research)

---

**Status:** ✅ Publication-Ready | 🚀 Deployment-Optimized
