---
name: PennyLane + Keras gradient flow bug
description: tf.map_fn breaks backprop through PennyLane QNodes — must use parameter-shift + float64 weights; KerasLayer removed in PL 0.38+
---

## Rule
`tf.map_fn` + `diff_method="backprop"` silently kills quantum weight gradients.
The correct fix for PennyLane 0.45 (where `qml.qnn.KerasLayer` was removed) requires
TWO simultaneous conditions:

1. `diff_method="parameter-shift"` — registers @tf.custom_gradient, propagates through map_fn
2. `q_weights` declared with `dtype=tf.float64` — parameter-shift returns float64 gradients;
   float32 weights cause `"type float32 does not match type float64"` in tape.gradient

## Why
`tf.map_fn` compiles to `tf.while_loop`. TF backprop cannot trace QNode graph ops
through a while_loop (backprop mode). With parameter-shift, PennyLane registers the
gradient externally as `@tf.custom_gradient` — TF sees the QNode as an opaque op with
a known gradient function. This propagates correctly through map_fn.

Parameter-shift evaluates the circuit at ±π/2 shifted parameters using numpy (float64).
The resulting gradient tensors are float64. If q_weights are float32, the gradient
application step tries to multiply float64 × float32 → crashes.

`qml.qnn.KerasLayer` was removed in PennyLane 0.38+ — do NOT use it.

## Symptoms of the dead-gradient bug
- val_accuracy locked at 0.500 for all epochs
- loss stuck at ≈ ln(2) ≈ 0.6931
- confusion matrix predicts one class 100% of the time

## Symptom of the dtype bug (separate from dead gradient)
- Training crashes with: `TypeError: Input 'y' of 'Mul' Op has type float32 that does not match type float64 of argument 'x'`
- Traceback ends in `gradients = tape.gradient(loss, trainable_weights)`

## How to apply
```python
@qml.qnode(dev, interface="tf", diff_method="parameter-shift")   # KEY: not "backprop"
def circuit(inputs, weights): ...

class QuantumLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.q_weights = self.add_weight(
            name="q_weights",
            shape=(...),
            dtype=tf.float64,          # KEY: must be float64
            initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1),
            trainable=True,
        )

    def call(self, inputs):
        inputs_f64 = tf.cast(inputs, tf.float64)
        results_f64 = tf.map_fn(
            lambda x: tf.expand_dims(circuit(x, self.q_weights), axis=0),
            inputs_f64,
            fn_output_signature=tf.TensorSpec(shape=(1,), dtype=tf.float64),
        )
        return tf.cast(results_f64, tf.float32)  # back to float32 for Dense layers
```

Dtype chain: Dense(f32) → cast f32→f64 → QuantumLayer(f64) → cast f64→f32 → Dense(f32)
`tf.cast` has a pass-through gradient (just dtype-converts), so chain is fully differentiable.

## Near-zero initialisation
Use `RandomUniform(-0.1, 0.1)` (Grant et al. 2019) — keeps gates near identity at epoch 0
where local gradients are O(1). Uniform [0, 2π] initialisation is a barren-plateau trap.

## Learning rate
Use lr ≈ 0.05 for HQNN vs 0.01 for CNN — parameter-shift gradients are shallower than
classical backprop.

## Diagnostic circuits for Bloch sphere / entanglement analysis
Use separate QNodes WITHOUT `interface="tf"` for post-training analysis (pure numpy forward
pass). Extracting weights: `model.get_layer("quantum_layer").q_weights.numpy()`
