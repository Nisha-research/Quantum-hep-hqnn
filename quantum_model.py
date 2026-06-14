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
  │  PQC: Rot + CNOT entanglement layers         │
  │  Measurement: ⟨Z₀⟩                           │
  ├──────────────────────────────────────────────┤
  │  Output: Dense(1, sigmoid)                   │
  └──────────────────────────────────────────────┘

The quantum device is PennyLane's 'default.qubit' CPU simulator (NISQ-era
emulation; no real QPU is required).
"""

import numpy as np
import pennylane as qml
import tensorflow as tf


# ── Quantum circuit configuration ────────────────────────────────────────────
N_QUBITS  = 4    # one qubit per feature dimension
N_LAYERS  = 2    # depth of the variational ansatz


# ── PennyLane device (CPU quantum simulator) ─────────────────────────────────
_dev = qml.device("default.qubit", wires=N_QUBITS)


# ── Quantum node ─────────────────────────────────────────────────────────────

@qml.qnode(_dev, interface="tf", diff_method="backprop")
def _quantum_circuit(inputs: tf.Tensor, weights: tf.Variable) -> tf.Tensor:
    """
    4-qubit variational quantum circuit.

    Parameters
    ----------
    inputs  : tf.Tensor, shape (4,)
        Classical embedding values in [–1, 1] (tanh output).
    weights : tf.Variable, shape (N_LAYERS, N_QUBITS, 3)
        Trainable Rot gate parameters (phi, theta, omega).

    Returns
    -------
    tf.Tensor, scalar
        PauliZ expectation value on qubit 0 ∈ [–1, 1].
    """
    # ── Angle encoding ────────────────────────────────────────────────────
    # Map each classical feature to a qubit rotation.
    # Multiplying by π maps the tanh range [–1, 1] → [–π, π].
    for i in range(N_QUBITS):
        qml.RX(np.pi * inputs[i], wires=i)

    # ── Parameterised Quantum Circuit (PQC) ───────────────────────────────
    # Each layer applies a Rot gate (arbitrary single-qubit rotation) to
    # every qubit, followed by a ring of CNOT gates to entangle them.
    for layer in range(N_LAYERS):
        # Single-qubit rotations
        for qubit in range(N_QUBITS):
            qml.Rot(
                weights[layer, qubit, 0],   # phi
                weights[layer, qubit, 1],   # theta
                weights[layer, qubit, 2],   # omega
                wires=qubit,
            )
        # Entanglement: ring topology CNOT gates
        for qubit in range(N_QUBITS):
            qml.CNOT(wires=[qubit, (qubit + 1) % N_QUBITS])

    # ── Measurement ───────────────────────────────────────────────────────
    return qml.expval(qml.PauliZ(0))


# ── Keras custom layer wrapping the QNode ────────────────────────────────────

class QuantumLayer(tf.keras.layers.Layer):
    """
    A tf.keras.layers.Layer that executes the quantum circuit.

    The quantum weights are stored as a tf.Variable so that TF's automatic
    differentiation can compute gradients through the QNode via PennyLane's
    backprop interface.
    """

    def __init__(self, n_layers: int = N_LAYERS, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers

    def build(self, input_shape):
        # shape: (N_LAYERS, N_QUBITS, 3) — 3 Euler angles per Rot gate
        self.q_weights = self.add_weight(
            name="q_weights",
            shape=(self.n_layers, N_QUBITS, 3),
            initializer=tf.keras.initializers.RandomUniform(
                minval=0, maxval=2 * np.pi
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply the quantum circuit to each sample in the batch.

        Parameters
        ----------
        inputs : tf.Tensor, shape (batch, 4)

        Returns
        -------
        tf.Tensor, shape (batch, 1)
        """
        # Process each sample individually (QNode does not natively batch)
        results = tf.map_fn(
            lambda x: tf.expand_dims(
                _quantum_circuit(x, self.q_weights), axis=0
            ),
            inputs,
            fn_output_signature=tf.RaggedTensorSpec(shape=[1], dtype=tf.float32),
        )
        return tf.reshape(results.flat_values, (-1, 1))

    def get_config(self):
        config = super().get_config()
        config.update({"n_layers": self.n_layers})
        return config


# ── Model factories ───────────────────────────────────────────────────────────

def build_hqnn(learning_rate: float = 0.01) -> tf.keras.Model:
    """
    Build and compile the Hybrid Classical-Quantum Neural Network (HQNN).

    Classical CNN compresses the 28×28 image to 4 features,
    which are then processed by the quantum variational circuit.

    Parameters
    ----------
    learning_rate : float

    Returns
    -------
    tf.keras.Model (compiled, ready to train)
    """
    inp = tf.keras.Input(shape=(28, 28, 1), name="detector_image")

    # ── Classical feature extractor ───────────────────────────────────────
    x = tf.keras.layers.Conv2D(
        filters=8, kernel_size=3, activation="relu", padding="same",
        name="conv2d"
    )(inp)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="maxpool")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(4, activation="tanh", name="embedding")(x)

    # ── Quantum variational classifier ────────────────────────────────────
    x = QuantumLayer(n_layers=N_LAYERS, name="quantum_layer")(x)

    # ── Binary output ─────────────────────────────────────────────────────
    # Map the PauliZ expectation (∈ [–1, 1]) through sigmoid for probability.
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
    Build and compile a purely classical CNN of equivalent parameter count
    for direct benchmark comparison against the HQNN.

    The quantum layer is replaced by a single Dense(8) + Dense(1) block
    to match approximate parameter budget.

    Parameters
    ----------
    learning_rate : float

    Returns
    -------
    tf.keras.Model (compiled, ready to train)
    """
    inp = tf.keras.Input(shape=(28, 28, 1), name="detector_image")

    x = tf.keras.layers.Conv2D(
        filters=8, kernel_size=3, activation="relu", padding="same",
        name="conv2d"
    )(inp)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="maxpool")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(4, activation="tanh", name="embedding")(x)

    # Classical equivalent of the quantum layer
    x = tf.keras.layers.Dense(8, activation="relu", name="classical_hidden")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="Classical_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
