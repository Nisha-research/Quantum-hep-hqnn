"""
qiskit_noise.py
---------------
Hardware noise robustness analysis via Qiskit Aer.

Step-by-step:
  1. Extract the trained Dense(4,tanh) CNN embedding + 24 PQC Rot-params from the HQNN.
  2. Rebuild the identical 4-qubit circuit in Qiskit (same RX angle encoding,
     same Rot+CNOT-ring ansatz, same qubit-0 Z-measurement).
  3. Construct four Qiskit Aer NoiseModels of increasing severity:
       Ideal       — no errors (noiseless Aer simulation)
       Light       — depolarizing p=0.1% (1Q) / 1% (2Q), readout 1%
       Manila-cal. — depolarizing p=0.15% (1Q) / 1.2% (2Q), readout 2%
                     (IBM Manila public calibration data, ibm_manila 5-qubit Falcon r5)
       2× Manila   — 2× the above; stress-test
  4. Run the full validation set through each noisy circuit.
  5. Map ⟨Z₀⟩ expectation values through the trained Dense(1,sigmoid) output
     layer to get final predictions — no retraining, pure inference transfer.

IBM Manila reference (public calibration reports, 2022–2023):
  1Q gate error:  ~0.001–0.002   (avg ≈ 0.0015)
  CX gate error:  ~0.008–0.015   (avg ≈ 0.012)
  Readout error:  ~0.015–0.030   (avg ≈ 0.020)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from sklearn.metrics import accuracy_score, roc_auc_score

N_QUBITS = 4
N_LAYERS = 2

# ── Noise level catalogue ─────────────────────────────────────────────────────
NOISE_LEVELS: dict[str, dict[str, float]] = {
    "Ideal":          {"p1": 0.000,  "p2": 0.000,  "ro": 0.000},
    "Light":          {"p1": 0.001,  "p2": 0.010,  "ro": 0.010},
    "Manila-calibr.": {"p1": 0.0015, "p2": 0.012,  "ro": 0.020},
    "2× Manila":      {"p1": 0.003,  "p2": 0.024,  "ro": 0.040},
}

_LEVEL_COLORS = {
    "Ideal":          "#2ecc71",
    "Light":          "#3498db",
    "Manila-calibr.": "#e67e22",
    "2× Manila":      "#e74c3c",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_noise_model(p1: float, p2: float, ro: float) -> NoiseModel | None:
    """
    Build a Qiskit Aer NoiseModel.

    Parameters
    ----------
    p1 : depolarizing probability for single-qubit gates (rx, ry, rz)
    p2 : depolarizing probability for two-qubit gates (cx)
    ro : symmetric readout flip probability per qubit
    """
    if p1 == 0.0 and p2 == 0.0 and ro == 0.0:
        return None                             # ideal — skip noise model

    nm = NoiseModel()
    if p1 > 0.0:
        err1 = depolarizing_error(p1, 1)
        nm.add_all_qubit_quantum_error(err1, ["rx", "ry", "rz"])
    if p2 > 0.0:
        err2 = depolarizing_error(p2, 2)
        nm.add_all_qubit_quantum_error(err2, ["cx"])
    if ro > 0.0:
        ro_err = ReadoutError([[1.0 - ro, ro], [ro, 1.0 - ro]])
        nm.add_all_qubit_readout_error(ro_err)
    return nm


def _build_pqc_circuit(embedding: np.ndarray, pqc_weights: np.ndarray) -> QuantumCircuit:
    """
    Reconstruct the PennyLane PQC as a Qiskit QuantumCircuit.

    PennyLane → Qiskit mapping
    --------------------------
    RX(π·xᵢ)            →  qc.rx(π·xᵢ, i)
    Rot(φ,θ,ω)           →  qc.rz(φ, q); qc.ry(θ, q); qc.rz(ω, q)
      (PL Rot = RZ(ω)@RY(θ)@RZ(φ), applied left→right in Qiskit gate order)
    CNOT(q, q+1 mod N)   →  qc.cx(q, (q+1)%N)
    ⟨Z₀⟩ measurement     →  qc.measure(0, 0)

    Parameters
    ----------
    embedding   : shape (N_QUBITS,) — tanh output of the CNN embedding layer
    pqc_weights : shape (N_LAYERS, N_QUBITS, 3) — Rot parameters [φ,θ,ω]
    """
    qc = QuantumCircuit(N_QUBITS, 1)   # 4 qubits, 1 classical bit (qubit 0)

    # Angle encoding
    for i in range(N_QUBITS):
        qc.rx(float(np.pi * embedding[i]), i)

    # Variational layers
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            phi   = float(pqc_weights[layer, q, 0])   # RZ first
            theta = float(pqc_weights[layer, q, 1])   # RY second
            omega = float(pqc_weights[layer, q, 2])   # RZ third
            qc.rz(phi, q)
            qc.ry(theta, q)
            qc.rz(omega, q)
        for q in range(N_QUBITS):
            qc.cx(q, (q + 1) % N_QUBITS)

    qc.measure(0, 0)
    return qc


def _counts_to_expval(counts: dict, shots: int) -> float:
    """
    Convert Qiskit measurement counts to ⟨Z₀⟩ = P(|0⟩) − P(|1⟩).

    With a single classical bit (qubit 0 only), keys are "0" and "1".
    """
    n_one = int(counts.get("1", 0))
    return 1.0 - 2.0 * n_one / shots


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


# ── Public API ────────────────────────────────────────────────────────────────

def export_pqc_params(hqnn_model: tf.keras.Model) -> dict:
    """
    Extract trained PQC + output-layer parameters from the HQNN.

    Returns
    -------
    {
      "pqc_weights"   : np.ndarray  (N_LAYERS, N_QUBITS, 3)
      "output_weight" : float        Dense(1, sigmoid) kernel w
      "output_bias"   : float        Dense(1, sigmoid) bias b
    }
    """
    pqc_w    = hqnn_model.get_layer("quantum_layer").q_weights.numpy()
    out_w, out_b = hqnn_model.get_layer("output").get_weights()
    return {
        "pqc_weights":   pqc_w.astype(float),
        "output_weight": float(out_w[0, 0]),
        "output_bias":   float(out_b[0]),
    }


def run_hardware_noise_sweep(
    hqnn_model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    shots: int = 2048,
    progress_callback=None,
) -> dict[str, dict]:
    """
    Evaluate the trained PQC under each noise level with Qiskit Aer.

    The CNN embedding sub-model runs unchanged in TF (classical pre-processing
    is assumed noise-free).  Only the 4-qubit quantum circuit is simulated with
    hardware noise.  The trained Dense(1, sigmoid) output layer is applied in
    numpy to the recovered ⟨Z₀⟩ expectation values.

    Parameters
    ----------
    hqnn_model       : trained HQNN (needs layers: "embedding", "quantum_layer", "output")
    X_val            : validation images (N, 28, 28, 1), float32
    y_val            : integer labels (N,)
    shots            : Aer measurement shots per circuit
    progress_callback: optional callable(fraction: float, msg: str)

    Returns
    -------
    dict keyed by noise label; each value:
      {"accuracy": float, "auc": float, "probs": np.ndarray, "expvals": np.ndarray}
    """
    # Extract trained parameters
    params      = export_pqc_params(hqnn_model)
    pqc_weights = params["pqc_weights"]
    out_w       = params["output_weight"]
    out_b       = params["output_bias"]

    # CNN embedding (classical, no noise)
    embed_model = tf.keras.Model(
        inputs=hqnn_model.input,
        outputs=hqnn_model.get_layer("embedding").output,
        name="hw_noise_embed",
    )
    embeddings = embed_model.predict(X_val, verbose=0).astype(float)   # (N, 4)

    # Pre-build all circuits (shared across noise levels)
    circuits = [
        _build_pqc_circuit(embeddings[i], pqc_weights)
        for i in range(len(embeddings))
    ]

    backend = AerSimulator()
    # Transpile once — Qiskit 2.x requires transpilation before backend.run().
    # num_processes=1 is mandatory in Replit: Qiskit's default parallel transpiler
    # spawns a ProcessPoolExecutor, which crashes inside Streamlit's subprocess
    # environment ("BrokenProcessPool").  Single-process transpilation is fast
    # enough for circuits this size.
    tqcs = transpile(circuits, backend, optimization_level=0, num_processes=1)

    results: dict[str, dict] = {}
    n_levels = len(NOISE_LEVELS)
    y_int = y_val.astype(int)

    for lvl_i, (label, nm_params) in enumerate(NOISE_LEVELS.items()):
        if progress_callback:
            progress_callback(
                lvl_i / n_levels,
                f"Qiskit Aer [{label}]  ({lvl_i + 1}/{n_levels})…",
            )

        nm = _make_noise_model(**nm_params)
        run_kwargs: dict = {"shots": shots}
        if nm is not None:
            run_kwargs["noise_model"] = nm

        job    = backend.run(tqcs, **run_kwargs)
        result = job.result()

        expvals, probs = [], []
        for i in range(len(circuits)):
            counts  = result.get_counts(i)
            z0      = _counts_to_expval(counts, shots)
            prob    = _sigmoid(out_w * z0 + out_b)
            expvals.append(z0)
            probs.append(prob)

        expvals = np.array(expvals)
        probs   = np.array(probs)
        y_pred  = (probs > 0.5).astype(int)
        acc     = accuracy_score(y_int, y_pred)
        try:
            auc_val = float(roc_auc_score(y_int, probs))
        except Exception:
            auc_val = float("nan")

        results[label] = {
            "accuracy": acc,
            "auc":      auc_val,
            "probs":    probs,
            "expvals":  expvals,
        }

    if progress_callback:
        progress_callback(1.0, "Hardware noise sweep complete!")

    return results
