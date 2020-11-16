#import tensorflow and its libraries
import tensorflow as tf
import tensorflow_quantum as tfq
#import quantum circuit situmator and other numerical libraries
import cirq
import sympy
import numpy as np

# visualization tools
%matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

a, b = sympy.symbols('a b')

# ２qubits circuit
q0, q1 = cirq.GridQubit.rect(1, 2)

# rx gate on q0 with parameter a, and ry gate on q1 with parameter b
circuit = cirq.Circuit(
    cirq.rx(a).on(q0),
    cirq.ry(b).on(q1), cirq.CNOT(control=q0, target=q1))

SVGCircuit(circuit)


# state vector at a=0.5, b=-0.5
resolver = cirq.ParamResolver({a: 0.5, b: -0.5})
output_state_vector = cirq.Simulator().simulate(circuit, resolver).final_state
output_state_vector


#gets an array([ 0.9387913 +0.j        , -0.23971277+0.j        , 0.        +0.06120872j,  0.        -0.23971277j], dtype=complex64)

z0 = cirq.Z(q0)

qubit_map={q0: 0, q1: 1}

z0.expectation_from_wavefunction(output_state_vector, qubit_map).real



# result 0.8775825500488281



v_a = output_state_vector

v_b = np.conjugate(a)
#v_b

Z = np.array([[1,0],[0,-1]])
I = np.eye(2

v_b@np.kron(Z,I)@v_a

z0x1 = 0.5 * z0 + cirq.X(q1)

z0x1.expectation_from_wavefunction(output_state_vector, qubit_map).real

# to a order-1 tensor (vector)
circuit_tensor = tfq.convert_to_tensor([circuit])

print(circuit_tensor.shape)
print(circuit_tensor.dtype)

pauli_tensor = tfq.convert_to_tensor([z0, z0x1])
pauli_tensor.shape









