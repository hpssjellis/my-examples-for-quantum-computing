All from

https://quantum-computing.ibm.com/docs/iqx/operations-glossary




The H, or Hadamard
converts classical 0,1 to Quantum +,-




Pauli X gate 
flips state 0 to 1 or 1 to 0



CX gate
slow controlled-NOT gate
X on the target whenever the control is in state. If the control qubit is in a superposition, this gate creates entanglement



CCX gate
Toffoli or double controlled-X gate
two control qubits and one target
Often with H gate


SWAP gate
The SWAP gate swaps the states of two qubits.


CSWAP gate
Fredkin or CSWAP gate
swaps the states of the two target qubits if the control qubit is in the  state.


T gate
π/4 radian rotation about the Z axis



P or Phase or S gate
Applies a phase of i to the 1 state
Equivalent to a π/2 radian rotation about the Z axis




Pauli Z gate
Flips the +- states
multiplies by -1




Sdg gate
The inverse of the S gate.
It induces a −π/2 phase

Tdg gate
The inverse of the T gate.



U1 gate
The U1 gate applies a phase of e (i * theta) to the 1 state. 




Barrier operation
Don't combine gates


Reset operation
The reset operation returns a qubit to state 0


Measurement, not a reversible operation
In the standard basis, also known as the z basis or computational basis. 


The RX gate 
rotates the qubit state around the x axis by the given angle


The RY gate 
Rotatesthe qubit state around the y axis by e (i*Theta y) the given angle and does not introduce complex amplitudes.

RZ gate
Rotates the qubit state around the z axis by the given angle. It is a diagonal gate and is equivalent to U1 up to a phase of e(i*Theta/2)


U3 gate
The three parameters allow the construction of any single-qubit gate. Has a duration of one unit of gate time.


Pauli Y gate 
Ry for the angle theta. Applying X and Z, up to a global phase factor.


U2 gate
The two parameters control two different rotations within the gate. Has a duration of one unit of gate time.


CH gate
The controlled-Hadamard gate acts on a control and target qubit. It performs an H on the target whenever the control is in state 1.



CY gate
The controlled-Y gate acts on a control and target qubit. It performs a Y on the target whenever the control is in state 1.



CZ gate
The controlled-Z gate acts on a control and target qubit. It performs a Z on the target whenever the control is in state 1. This gate is symmetric; swapping the two qubits it acts on doesn’t change anything.


CRX gate
Applies the RX gate to the target qubit if the control qubit is in state 1, or alternatively in state 0 if the argument ctrl_state is set to 0.




CRY gate
Applies the RY gate to the target qubit if the control qubit is in state , or alternatively in state  if the argument ctrl_state is set to 0.It can be used to map functions to qubit amplitudes,
num_state_qubits, 
slope, 
offset, 
basis(‘X’, ‘Y’, ‘Z’), 
name of the circuit object.


CRZ gate
The controlled-RZ gate acts on a control and target qubit. It performs an RZ rotation on the target whenever the control is in state 1.

CU1 gate
Applies the U1 gate if the control qubit is in state 1, or alternatively in state 0 if the argument ctrl_state is set to 0. This is a diagonal and symmetric gate. One usage of this gate is in the quantum Fourier transform.



CU3 gate
Applies the U3 gate if the control qubit is in state 1, or alternatively in state 0 if the argument ctrl_state is set to 0.


RXX gate
The RXX gate implements the Mølmer–Sørensen gate, the native gate on ion-trap systems, can be expressed as a sum of RXX gates.




RZZ gate
The RZZ gate requires a single parameter: an angle expressed in radians. This gate is symmetric; swapping the two qubits it acts on doesn’t change anything.


