import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

# used to embed classical data in quantum circuits
def convert_to_circuit_cont(image):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 1)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.rx(value).on(qubits[i]))
    return circuit

# define classical dataset
length = 1000
np.random.seed(42)

# create a linearly increasing set for x from 0 to 1 in 1/length steps
x_train_sorted = np.asarray([[x/length] for x in range(0,length)], dtype=np.float32)
# p is used to shuffle x and y similarly
p = np.random.permutation(len(x_train_sorted))
x_train = x_train_sorted[p]
# y = x < 0.3 in {-1, 1} for Hinge loss
y_train_sorted = np.asarray([1 if (x/length)<0.30 else -1 for x in range(0,length)])
y_train = y_train_sorted[p]
# test == train for this example
x_test = x_train_sorted[:]
y_test = y_train_sorted[:]

# convert classical data into quantum circuits
x_train_circ = [convert_to_circuit_cont(x) for x in x_train]
x_test_circ = [convert_to_circuit_cont(x) for x in x_test]
x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

# define the PQC circuit, consisting out of 1 qubit with 1 gate (Rx) and 1 parameter
def create_quantum_model():
    data_qubits = cirq.GridQubit.rect(1, 1)  
    circuit = cirq.Circuit()
    a = sympy.Symbol("a")
    circuit.append(cirq.rx(a).on(data_qubits[0])),
    return circuit, cirq.Z(data_qubits[0])
model_circuit, model_readout = create_quantum_model()

# Build the Keras model.
model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    # The PQC layer returns the expected value of the readout gate, range [-1,1].
    tfq.layers.PQC(model_circuit, model_readout),
])

# used for logging progress during optimization
def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)
    return tf.reduce_mean(result)

# compile the model with Hinge loss and Adam, as done in the example. Have tried with various learning_rates
model.compile(
    loss = tf.keras.losses.Hinge(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    metrics=[hinge_accuracy])

EPOCHS = 20
BATCH_SIZE = 32
NUM_EXAMPLES = 1000

# fit the model
qnn_history = model.fit(
      x_train_tfcirc, y_train, 
      batch_size=32,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_tfcirc, y_test),
      use_multiprocessing=False)

results = model.predict(x_test_tfcirc)
results_mapped = [-1 if x<=0 else 1 for x in results[:,0]]
print(np.sum(np.equal(results_mapped, y_test)))
