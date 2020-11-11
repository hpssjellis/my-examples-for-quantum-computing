import pennylane as qml
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.init import strong_ent_layers_uniform

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(weights, x=None):
    AngleEmbedding(x, [0,1])
    StronglyEntanglingLayers(weights, wires=[0,1])
    return qml.expval(qml.PauliZ(0))

init_weights = strong_ent_layers_uniform(n_layers=3, n_wires=2)
print(circuit(init_weights, x=[1., 2.]))
