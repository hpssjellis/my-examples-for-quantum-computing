## Attempt to simplify Small Quantum Keras models
## By Rocksetta
## from  https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html
## from  https://github.com/hpssjellis/my-examples-for-quantum-computing/blob/main/pennylaneai/qml-demos/tf-tutorial_qnn_module_tf.py

import os

import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import pennylane as qml

X, y = make_moons(n_samples=200, noise=0.1)
print("X")
print(X)

print()
print("y")
print(y)

y_hot = tf.keras.utils.to_categorical(y, num_classes=2)  # one-hot encoded labels

print()
print("y_hot")
print(y_hot)

c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in y]  # colours for each class
plt.axis("off")
plt.scatter(X[:, 0], X[:, 1], c=c)
#plt.show()
plt.draw()
plt.pause(0.001)
input("Open Ports --> Open Preview or Browser --> push enter to continue")

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

n_layers = 6
weight_shapes = {"weights": (n_layers, n_qubits)}

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2))
model.add(qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits))
model.add(tf.keras.layers.Dense(2, activation="softmax"))

model.compile(tf.keras.optimizers.SGD(learning_rate=0.2), loss="mae", metrics=["accuracy"])


X = X.astype("float32")
y_hot = y_hot.astype("float32")
fitting = model.fit(X, y_hot, epochs=2, batch_size=5, validation_split=0.25, verbose=2)



model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Dense(2))
model2.add(tf.keras.layers.Dense(2, activation="softmax"))

model2.compile(tf.keras.optimizers.SGD(learning_rate=0.2), loss="mae", metrics=["accuracy"])

fitting = model2.fit(X, y_hot, epochs=2, batch_size=5, validation_split=0.25, verbose=2)



#model saving test.
#pure keras model saves fine


model2.summary()
model2.save('kerasOnlyModel')

#Keras and Quantum  model does not save

model.summary()
model.save('QuantumAndKerasModel') 



