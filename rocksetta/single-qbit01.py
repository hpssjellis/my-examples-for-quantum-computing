import pennylane as qml
import tensorflow as tf
import numpy as np

n_qubits = 1
layers = 1
data_dimension = 1
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


weight_shapes = {"weights": (layers, n_qubits, 3)}

qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
clayer1 = tf.keras.layers.Dense(n_qubits)
clayer2 = tf.keras.layers.Dense(data_dimension, activation="linear")
model = tf.keras.models.Sequential([clayer1, qlayer, clayer2])

X = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=np.float32)
Y = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=np.float32)

X = X.reshape((len(X), data_dimension))
Y = Y.reshape((len(Y), data_dimension))

opt = tf.keras.optimizers.SGD(learning_rate=0.5)
model.compile(opt, loss='mse',)

model.fit(X, Y, epochs=60, batch_size=1)
