import pennylane as qml
import tensorflow as tf
import numpy as np

n_qubits = 1
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.expval(qml.PauliZ(0)))


weight_shapes = {"weights": (1, n_qubits, 3)}

qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=1)
clayer1 = tf.keras.layers.Dense(1)
clayer2 = tf.keras.layers.Dense(1, activation="linear")
model = tf.keras.models.Sequential([clayer1, qlayer, clayer2])

X = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
Y = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

X1 = X.reshape((len(X), 1))
Y1 = Y.reshape((len(Y), 1))

opt = tf.keras.optimizers.SGD(learning_rate=0.5)
model.compile(opt, loss='mse',)


model.fit(X1, Y1, epochs=60, batch_size=1)
