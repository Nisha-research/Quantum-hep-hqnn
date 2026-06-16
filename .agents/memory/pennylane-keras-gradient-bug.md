---
name: PennyLane + Keras gradient flow bug
description: tf.map_fn breaks backprop through PennyLane QNodes — must use qml.qnn.KerasLayer
---

## Rule
Never wrap a PennyLane QNode in `tf.map_fn` inside a custom `tf.keras.layers.Layer`.
Use `qml.qnn.KerasLayer` instead.

## Why
`tf.map_fn` internally uses `tf.while_loop`.  TF's backprop cannot trace through a
while_loop to differentiate a QNode using `diff_method="backprop"`.  Result: quantum
weight gradients are identically zero every step.  Symptoms are unmistakable:
- val_accuracy locked at 0.500 for all epochs
- loss stuck at ≈ ln(2) ≈ 0.6931 (binary cross-entropy of a constant 0.5 predictor)
- confusion matrix predicts one class 100% of the time

## How to apply
```python
weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
weight_specs  = {"weights": {"initializer": tf.keras.initializers.RandomUniform(-0.1, 0.1)}}
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=1, weight_specs=weight_specs)
```
The kwarg name in `weight_shapes` must match the argument name in the QNode signature.

## Near-zero initialisation
Uniform [0, 2π] init places quantum weights in the high-variance flat region immediately.
Use [-0.1, 0.1] (Grant et al. 2019) so gates start near identity where local gradients are O(1).

## Learning rate
Use lr ≈ 0.05 for the HQNN vs 0.01 for a classical CNN — quantum gradients are shallower.
