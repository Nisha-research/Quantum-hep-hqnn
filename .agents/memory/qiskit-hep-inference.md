---
name: Qiskit hardware-noise inference for PennyLane-trained PQC
description: How to reconstruct a PennyLane PQC in Qiskit Aer and run noisy inference (no retraining)
---

## Rule
PennyLane Rot(φ,θ,ω) maps to Qiskit as `qc.rz(φ,q); qc.ry(θ,q); qc.rz(ω,q)`.
This is the ZYZ decomposition: Rot = RZ(ω) @ RY(θ) @ RZ(φ) applied left-to-right
in circuit gate order.

## ⟨Z₀⟩ from shot-based counts
With only qubit 0 measured into classical bit 0, counts keys are "0" or "1":
  ⟨Z₀⟩ = 1 - 2 * counts.get("1", 0) / shots   (= P(|0⟩) − P(|1⟩))

Map through trained Dense(1,sigmoid) output layer:
  prob = sigmoid(w * z0 + b)   where w,b = model.get_layer("output").get_weights()

## Qiskit 2.x requirements
- `transpile(circuits, backend, optimization_level=0, num_processes=1)` is required before `backend.run()`
- **`num_processes=1` is mandatory on Replit**: Qiskit's default parallel transpiler spawns a
  `ProcessPoolExecutor`; this crashes with `BrokenProcessPool` inside Streamlit's subprocess
  environment. Single-process transpilation is fast enough for small circuits.
- Noise model applied via `backend.run(tqcs, noise_model=nm, shots=N)`
- `NoiseModel`, `depolarizing_error`, `ReadoutError` all in `qiskit_aer.noise`
- For RX/RY/RZ single-qubit gates: `nm.add_all_qubit_quantum_error(err1, ["rx","ry","rz"])`
- For CX two-qubit gates: `nm.add_all_qubit_quantum_error(err2, ["cx"])`

## IBM Manila calibration reference (public, 2022-2023)
Used as the "device-calibrated" noise level:
  1Q gate error: p1 = 0.0015
  CX gate error: p2 = 0.012
  Readout error: ro = 0.020
Four levels: Ideal (0/0/0), Light (0.001/0.01/0.01), Manila-cal, 2×Manila

## Architectural boundary
Classical CNN preprocessing runs in TF (no noise). Only the 4-qubit quantum
circuit is simulated with Aer noise. Output Dense(1,sigmoid) applied in numpy.
This matches the real deployment scenario: only the QPU sees hardware noise.

## Why not retrain under noise
- Parameter-shift over a noisy Aer backend would require 2×24 circuit evals
  per sample per gradient step — extremely slow and not the user's goal.
- The reviewer-credible result is: "trained on ideal sim, tested on noisy sim
  to characterise deployment tolerance" — this is the standard NISQ benchmarking protocol.
