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
  │  Angle Encoding: RX(π·xi) on each qubit      │
  │  PQC: Rot + CNOT ring, L=2                   │
  │  Measurement: ⟨Z₀⟩                           │
  ├──────────────────────────────────────────────┤
  │  Output: Dense(1, sigmoid)                   │
  └──────────────────────────────────────────────┘

GRADIENT STRATEGY — diff_method="parameter-shift"
--------------------------------------------------
PennyLane 0.45 removed qml.qnn.KerasLayer.  The alternative backprop approach
(diff_method="backprop") requires TF to trace the entire circuit graph, which
fails when the circuit is wrapped in tf.map_fn (tf.while_loop breaks the tape).

With diff_method="parameter-shift" PennyLane computes gradients by evaluating
the circuit at shifted parameter values and registering the result via
@tf.custom_gradient.  TF sees the QNode as an opaque op with a known gradient
function — this propagates correctly through tf.map_fn without needing to trace
the circuit internals.  Cost: 2×N_params forward passes per backward step
(2×24 = 48 for this circuit), which is fast on a 4-qubit CPU simulator.

INITIALISATION — near-zero (Grant et al. 2019)
-----------------------------------------------
Random uniform [0, 2π] initialisation places quantum weights in the high-
variance flat region from the very first step.  Initialising near zero keeps
gates close to identity, where local gradients are O(1).
"""

import numpy as np
import pennylane as qml
import tensorflow as tf


# ── Quantum circuit configuration ────────────────────────────────────────────
N_QUBITS = 4
N_LAYERS = 2


# ── PennyLane device ─────────────────────────────────────────────────────────
_dev = qml.device("default.qubit", wires=N_QUBITS)


# ── Quantum node ─────────────────────────────────────────────────────────────
@qml.qnode(_dev, interface="tf", diff_method="parameter-shift")
def _quantum_circuit(inputs: tf.Tensor, weights: tf.Variable) -> tf.Tensor:
    """
    4-qubit angle-encoded PQC.

    diff_method="parameter-shift" registers a @tf.custom_gradient so TF can
    backprop through tf.map_fn without needing to trace the circuit graph.
    """
    for i in range(N_QUBITS):
        qml.RX(np.pi * inputs[i], wires=i)
    for layer in range(N_LAYERS):
        for qubit in range(N_QUBITS):
            qml.Rot(
                weights[layer, qubit, 0],
                weights[layer, qubit, 1],
                weights[layer, qubit, 2],
                wires=qubit,
            )
        for qubit in range(N_QUBITS):
            qml.CNOT(wires=[qubit, (qubit + 1) % N_QUBITS])
    return qml.expval(qml.PauliZ(0))


# ── Keras custom layer ────────────────────────────────────────────────────────
class QuantumLayer(tf.keras.layers.Layer):
    """
    Keras layer wrapping the PennyLane QNode.

    DTYPE STRATEGY
    --------------
    PennyLane's parameter-shift rule internally operates in float64.  If the
    quantum weights are float32, the custom_gradient it registers returns
    float64 gradients that TF tries to multiply against float32 tensors →
    "type float32 does not match type float64" error.

    Fix: store q_weights as float64, cast the batch input up to float64
    before the circuit, and cast the output back to float32 for the
    subsequent Dense layers.  tf.cast preserves gradient flow (identity
    gradient with dtype conversion), so the full chain is consistent:

        Dense (f32) → cast f32→f64 → QuantumLayer (f64) → cast f64→f32 → Dense (f32)
    """

    def __init__(self, n_layers: int = N_LAYERS, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers

    def build(self, input_shape):
        self.q_weights = self.add_weight(
            name="q_weights",
            shape=(self.n_layers, N_QUBITS, 3),
            dtype=tf.float64,          # must match parameter-shift gradient dtype
            initializer=tf.keras.initializers.RandomUniform(
                minval=-0.1, maxval=0.1
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """inputs: (batch, 4) float32  →  output: (batch, 1) float32"""
        inputs_f64 = tf.cast(inputs, tf.float64)
        results_f64 = tf.map_fn(
            lambda x: tf.expand_dims(_quantum_circuit(x, self.q_weights), axis=0),
            inputs_f64,
            fn_output_signature=tf.TensorSpec(shape=(1,), dtype=tf.float64),
        )
        return tf.cast(results_f64, tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"n_layers": self.n_layers})
        return config


# ── Model factories ───────────────────────────────────────────────────────────

def build_hqnn(learning_rate: float = 0.05) -> tf.keras.Model:
    """
    Compile the Hybrid Classical-Quantum Neural Network.

    lr=0.05 (vs 0.01 for CNN) compensates for the shallower gradient
    landscape of a 4-qubit parameter-shift circuit.
    """
    inp = tf.keras.Input(shape=(28, 28, 1), name="detector_image")
    x = tf.keras.layers.Conv2D(
        8, 3, activation="relu", padding="same", name="conv2d",
    )(inp)
    x = tf.keras.layers.MaxPooling2D(2, name="maxpool")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(4, activation="tanh", name="embedding")(x)
    x = QuantumLayer(name="quantum_layer")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="HQNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_classical_cnn(learning_rate: float = 0.01) -> tf.keras.Model:
    """Classical CNN benchmark — same feature extractor, no quantum layer."""
    inp = tf.keras.Input(shape=(28, 28, 1), name="detector_image")
    x = tf.keras.layers.Conv2D(
        8, 3, activation="relu", padding="same", name="conv2d",
    )(inp)
    x = tf.keras.layers.MaxPooling2D(2, name="maxpool")(x)
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
    """HQNN variant with n_qubits for the qubit-scaling experiment."""
    dev_n = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev_n, interface="tf", diff_method="parameter-shift")
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

    class _QuantumLayerN(tf.keras.layers.Layer):
        def __init__(self, nq, nl, **kwargs):
            super().__init__(**kwargs)
            self._nq = nq
            self._nl = nl

        def build(self, input_shape):
            self.q_weights = self.add_weight(
                name="q_weights",
                shape=(self._nl, self._nq, 3),
                dtype=tf.float64,          # match parameter-shift gradient dtype
                initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs):
            inputs_f64 = tf.cast(inputs, tf.float64)
            results_f64 = tf.map_fn(
                lambda x: tf.expand_dims(_circuit_n(x, self.q_weights), axis=0),
                inputs_f64,
                fn_output_signature=tf.TensorSpec(shape=(1,), dtype=tf.float64),
            )
            return tf.cast(results_f64, tf.float32)

    inp = tf.keras.Input(shape=(28, 28, 1), name="detector_image")
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same", name="conv2d")(inp)
    x = tf.keras.layers.MaxPooling2D(2, name="maxpool")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(n_qubits, activation="tanh", name="embedding")(x)
    x = _QuantumLayerN(n_qubits, n_layers, name="quantum_layer")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name=f"HQNN_{n_qubits}q")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
