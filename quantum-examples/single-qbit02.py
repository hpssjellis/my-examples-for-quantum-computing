import pennylane as qml
import tensorflow as tf
import sklearn.datasets
import numpy as np

n_qubits = 1
layers = 1
data_dimension = 1

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, m, c):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.RX(m, wires=0)
    qml.RZ(c, wires=0)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    

weight_shapes = {"m": 1, "c":1}

qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
clayer1 = tf.keras.layers.Dense(n_qubits)
clayer2 = tf.keras.layers.Dense(data_dimension, activation="linear")

model = tf.keras.models.Sequential([qlayer])

X = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
Y = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)
X1 = X.reshape((len(X), 1))
Y1 = Y.reshape((len(Y), 1))

model.compile(loss='mean_squared_error',

optimizer=tf.keras.optimizers.Adam(0.01))

model.fit(X, Y, epochs=60, batch_size=1)
