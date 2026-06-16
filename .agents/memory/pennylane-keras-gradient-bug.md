---
name: PennyLane + Keras gradient flow bug
description: tf.map_fn breaks backprop through PennyLane QNodes — must use qml.qnn.KerasLayer
---

## Rule
`tf.map_fn` + `diff_method="backprop"` silently kills quantum weight gradients.
The correct fix for PennyLane 0.45 (where `qml.qnn.KerasLayer` was removed) is
`diff_method="parameter-shift"` — which works through `tf.map_fn` via `@tf.custom_gradient`.

## Why
`tf.map_fn` compiles to `tf.while_loop`. TF backprop cannot trace through a while_loop
to reach QNode graph ops (backprop mode).  With parameter-shift, PennyLane registers
the gradient externally as a `@tf.custom_gradient` — TF sees the QNode as a black-box
op with a known gradient function.  Custom-gradient ops propagate correctly through map_fn.

`qml.qnn.KerasLayer` was removed in PennyLane 0.38+ — do NOT use it.

## Symptoms of the dead-gradient bug
- val_accuracy locked at 0.500 for all epochs
- loss stuck at ≈ ln(2) ≈ 0.6931
- confusion matrix predicts one class 100% of the time

## How to apply
```python
@qml.qnode(dev, interface="tf", diff_method="parameter-shift")   # KEY: not "backprop"
def circuit(inputs, weights): ...

class QuantumLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.map_fn(
            lambda x: tf.expand_dims(tf.cast(circuit(x, self.q_weights), tf.float32), 0),
            inputs,
            fn_output_signature=tf.TensorSpec(shape=(1,), dtype=tf.float32),
        )
```

## Near-zero initialisation
Use `RandomUniform(-0.1, 0.1)` (Grant et al. 2019) — keeps gates near identity at epoch 0
where local gradients are O(1).  Uniform [0, 2π] initialisation is a barren-plateau trap.

## Learning rate
Use lr ≈ 0.05 for HQNN vs 0.01 for CNN — parameter-shift gradients are shallower than
classical backprop.

## Diagnostic circuits for Bloch sphere / entanglement analysis
Use separate QNodes WITHOUT `interface="tf"` for post-training analysis (pure numpy forward
pass).  Extracting weights: `model.get_layer("quantum_layer").q_weights.numpy()`.
