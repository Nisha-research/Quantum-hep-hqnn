"""
quantum_model.py
----------------
Hybrid Classical-Quantum Neural Network (HQNN) for particle event classification.

Architecture:
  ┌──────────────────────────────────────────────┐
  │  Classical CNN Feature Extractor             │
  │  Conv2D(8, 3×3, relu)                        │
  │  → MaxPool(2×2)                              │
  │  → Flatten                                   │
  │  → Dense(4, tanh)   ← 4-dim embedding        │
  ├──────────────────────────────────────────────┤
  │  Quantum Variational Classifier (4 qubits)   │
  │  qml.qnn.KerasLayer — batching handled       │
  │  Angle Encoding: RX(π·xi) on each qubit      │
  │  PQC: Rot + CNOT ring, L=2                   │
  │  Measurement: ⟨Z₀⟩                           │
  ├──────────────────────────────────────────────┤
  │  Output: Dense(1, sigmoid)                   │
  └──────────────────────────────────────────────┘

GRADIENT NOTE
-------------
The original hand-rolled QuantumLayer used tf.map_fn which internally wraps
execution in tf.while_loop.  This prevents TF's backprop from tracing through
the PennyLane QNode's TF computation graph, so quantum weight gradients are
always ~0 and the model never learns (loss ≈ ln(2), acc ≈ 0.5 throughout).

The fix is qml.qnn.KerasLayer — PennyLane's official Keras integration.
It handles per-sample batching internally using a vectorised map that keeps
the computation graph contiguous, so TF's gradient tape flows through it
correctly with diff_method="backprop".

INITIALISATION
--------------
Near-zero initialisation (Grant et al. 2019, Quantum 3, 214) is used for the
quantum weights.  Random uniform over [0, 2π] places the model in a high-
variance region of parameter space where gradients are already exponentially
small even before the barren-plateau effect from qubit count.  Initialising
near zero keeps gates close to identity, where local gradients are O(1).
"""

import numpy as np
import pennylane as qml
import tensorflow as tf


# ── Quantum circuit configuration ────────────────────────────────────────────
N_QUBITS = 4    # one qubit per feature dimension
N_LAYERS = 2    # PQC depth


# ── PennyLane device (CPU quantum simulator) ─────────────────────────────────
_dev = qml.device("default.qubit", wires=N_QUBITS)


# ── Quantum node ─────────────────────────────────────────────────────────────
#
# The circuit signature uses keyword argument `weights` so that
# qml.qnn.KerasLayer can automatically create and manage the trainable
# tf.Variable and wire its gradients into the Keras graph.
#
@qml.qnode(_dev, interface="tf", diff_method="backprop")
def _quantum_circuit(inputs: tf.Tensor, weights: tf.Variable) -> tf.Tensor:
    """
    4-qubit angle-encoded variational quantum circuit.

    Parameters
    ----------
    inputs  : tf.Tensor, shape (4,)
        Classical embedding values in [−1, 1] (tanh output).
    weights : tf.Variable, shape (N_LAYERS, N_QUBITS, 3)
        Trainable Rot gate parameters (phi, theta, omega) — managed by
        qml.qnn.KerasLayer.

    Returns
    -------
    tf.Tensor scalar — PauliZ expectation on qubit 0 ∈ [−1, 1].
    """
    # Angle encoding: map each feature to a qubit rotation
    for i in range(N_QUBITS):
        qml.RX(np.pi * inputs[i], wires=i)

    # PQC: Rot gates + CNOT ring, repeated L times
    for layer in range(N_LAYERS):
        for qubit in range(N_QUBITS):
            qml.Rot(
                weights[layer, qubit, 0],   # phi
                weights[layer, qubit, 1],   # theta
                weights[layer, qubit, 2],   # omega
                wires=qubit,
            )
        for qubit in range(N_QUBITS):
            qml.CNOT(wires=[qubit, (qubit + 1) % N_QUBITS])

    # Local observable: mitigates barren plateaus vs. a global cost
    return qml.expval(qml.PauliZ(0))


# Weight name must match the kwarg in _quantum_circuit
_WEIGHT_SHAPES = {"weights": (N_LAYERS, N_QUBITS, 3)}

# Near-zero init (Grant et al. 2019): keeps gates near identity at epoch 0
# where local gradients are O(1), avoiding the initialisation-induced plateau.
_WEIGHT_SPECS = {
    "weights": {
        "initializer": tf.keras.initializers.RandomUniform(
            minval=-0.1, maxval=0.1
        )
    }
}


# ── Model factories ───────────────────────────────────────────────────────────

def build_hqnn(learning_rate: float = 0.05) -> tf.keras.Model:
    """
    Build and compile the Hybrid Classical-Quantum Neural Network (HQNN).

    The quantum layer is implemented via qml.qnn.KerasLayer, which handles
    per-sample batching internally and keeps gradients flowing correctly
    through the PennyLane backprop interface.

    Parameters
    ----------
    learning_rate : float  (default 0.05 — higher than classical default to
                            compensate for the shallower gradient landscape of
                            the quantum circuit)

    Returns
    -------
    tf.keras.Model (compiled, ready to train)
    """
    inp = tf.keras.Input(shape=(28, 28, 1), name="detector_image")

    # Classical feature extractor
    x = tf.keras.layers.Conv2D(
        filters=8, kernel_size=3, activation="relu", padding="same",
        name="conv2d",
    )(inp)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="maxpool")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(4, activation="tanh", name="embedding")(x)

    # Quantum variational classifier
    # qml.qnn.KerasLayer creates and owns the tf.Variable for `weights`,
    # registers it as a trainable Keras weight, and applies the QNode to
    # each sample in the batch with correct gradient propagation.
    x = qml.qnn.KerasLayer(
        _quantum_circuit,
        _WEIGHT_SHAPES,
        output_dim=1,
        weight_specs=_WEIGHT_SPECS,
        name="quantum_layer",
    )(x)

    # Binary output: sigmoid maps ⟨Z₀⟩ ∈ [−1,1] to a probability
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="HQNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_classical_cnn(learning_rate: float = 0.01) -> tf.keras.Model:
    """
    Purely classical CNN benchmark — same feature extractor as HQNN,
    quantum layer replaced by Dense(8, relu) + Dense(1, sigmoid).

    Parameters
    ----------
    learning_rate : float

    Returns
    -------
    tf.keras.Model (compiled)
    """
    inp = tf.keras.Input(shape=(28, 28, 1), name="detector_image")

    x = tf.keras.layers.Conv2D(
        filters=8, kernel_size=3, activation="relu", padding="same",
        name="conv2d",
    )(inp)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="maxpool")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(4, activation="tanh", name="embedding")(x)
    x = tf.keras.layers.Dense(8, activation="relu", name="classical_hidden")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="Classical_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_hqnn_nqubits(
    n_qubits: int,
    n_layers: int = 2,
    learning_rate: float = 0.05,
) -> tf.keras.Model:
    """
    Build an HQNN with *n_qubits* for the qubit-scaling experiment.

    Creates a fresh PennyLane device and QNode scoped to this call so
    multiple qubit counts can coexist without shared state.  Uses
    qml.qnn.KerasLayer for correct gradient flow (same fix as build_hqnn).

    Parameters
    ----------
    n_qubits      : int   — number of qubits (2, 4, or 6 recommended)
    n_layers      : int   — PQC depth
    learning_rate : float

    Returns
    -------
    Compiled tf.keras.Model
    """
    dev_n = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev_n, interface="tf", diff_method="backprop")
    def _circuit_n(inputs, weights):
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
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    weight_specs  = {
        "weights": {
            "initializer": tf.keras.initializers.RandomUniform(-0.1, 0.1)
        }
    }

    inp = tf.keras.Input(shape=(28, 28, 1), name="detector_image")
    x = tf.keras.layers.Conv2D(
        8, 3, activation="relu", padding="same", name="conv2d",
    )(inp)
    x = tf.keras.layers.MaxPooling2D(2, name="maxpool")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(n_qubits, activation="tanh", name="embedding")(x)
    x = qml.qnn.KerasLayer(
        _circuit_n, weight_shapes, output_dim=1,
        weight_specs=weight_specs, name="quantum_layer",
    )(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name=f"HQNN_{n_qubits}q")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
