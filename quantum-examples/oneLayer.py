import pennylane as qml

wires = 4

def layer(inputs, theta, phi, varphi, x, y, z):
    qml.templates.DisplacementEmbedding(inputs, wires=range(wires))
    qml.templates.Interferometer(theta, phi, varphi, wires=range(wires))
    for i in range(wires):
        qml.Displacement(x[i], 0, wires=i)
        qml.Rotation(y[i], wires=i)
        qml.Kerr(z[i], wires=i)
    return [qml.expval(qml.X(wires=i)) for i in range(wires)]

interferometer_shape = int(wires * (wires - 1) / 2)

weight_shapes = {
    "theta": interferometer_shape,
    "phi": interferometer_shape,
    "varphi": 4,
    "x": wires,
    "y": wires,
    "z": wires,
}

dev = qml.device("strawberryfields.fock", wires=wires, cutoff_dim=4)
qnode = qml.QNode(layer, dev)
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, wires)

import numpy as np
x = np.random.random((10, wires))
qlayer(x)