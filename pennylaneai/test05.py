import pennylane as qml
import numpy as np
import tensorflow as tf
import sklearn.datasets

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

weight_shapes = {"weights": (3, n_qubits, 3)}

qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=2)
clayer1 = tf.keras.layers.Dense(2)
clayer2 = tf.keras.layers.Dense(2, activation="softmax")
model = tf.keras.models.Sequential([clayer1, qlayer, clayer2])

data = sklearn.datasets.make_moons()
X = tf.constant(data[0])
Y = tf.one_hot(data[1], depth=2)

opt = tf.keras.optimizers.SGD(learning_rate=0.5)
model.compile(opt, loss='mae')
