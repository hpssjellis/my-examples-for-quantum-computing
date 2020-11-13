import pennylane as qml
import numpy as np


dev = qml.device('default.qubit', wires=2, analytic=True)

@qml.qnode(dev)
def circuit(x=None, y=None):
    qml.BasisState(np.array([x,y]), wires=[0,1])
    qml.CNOT(wires=[0,1])
    return qml.probs(wires=[1])

# Get the probability of the first wire being in state 1
print(circuit(x=0,y=0)[1])
print(circuit(x=0,y=1)[1])
print(circuit(x=1,y=0)[1])
print(circuit(x=1,y=1)[1])
