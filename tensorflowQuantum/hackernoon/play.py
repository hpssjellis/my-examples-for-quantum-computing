import cirq
import numpy as np
from cirq import Circuit
from cirq.devices import GridQubit

# creating circuit with 5 qubit
length = 5

qubits = [cirq.GridQubit(i, j) for i in range(length) for j in range(length)]
print(qubits)


circuit = cirq.Circuit()
H1 = cirq.H(qubits[0])
H2 = cirq.H(qubits[1])
H3 = cirq.H(qubits[2])
H4 = cirq.H(qubits[3])
H5 = cirq.H(qubits[4])

C1 = cirq.CNOT(qubits[0],qubits[1])
C2 = cirq.CNOT(qubits[1],qubits[2])
C3 = cirq.CNOT(qubits[2],qubits[3])
C4 = cirq.CNOT(qubits[3],qubits[4])

#swap
S1 = cirq.SWAP(qubits[0],qubits[4])

#Rotation
X1 = cirq.X(qubits[0])
X2 = cirq.X(qubits[1])
X3 = cirq.X(qubits[2])
X4 = cirq.X(qubits[3])
X5 = cirq.X(qubits[4])


moment1 = cirq.Moment([H1])
moment2 = cirq.Moment([H2])
moment3 = cirq.Moment([H3])
moment4 = cirq.Moment([H4])
moment5 = cirq.Moment([H5])
moment6 = cirq.Moment([C1])
moment7 = cirq.Moment([C2])
moment8 = cirq.Moment([C3])
moment9 = cirq.Moment([S1])
moment10 = cirq.Moment([X1])
moment11 = cirq.Moment([X2])
moment12 = cirq.Moment([X3])
moment13 = cirq.Moment([X4])
moment14 = cirq.Moment([X5])

#circuit
circuit = cirq.Circuit((moment1, moment2, moment3, moment4, moment5 ,moment6 ,moment7, moment8, moment9, moment10, moment11, moment12, moment13, moment14))
print(circuit)


