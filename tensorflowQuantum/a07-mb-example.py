## example suggested by MichaelBroughton
## in my issues submission to TensorflowQuantum, which may now be closed.
## https://github.com/tensorflow/quantum/issues/443



import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

def my_embedding_circuit():
    # Note this must have the same number of free parameters as the layer that
    #   feeds into it from upstream. In this case you have 16.
    #   Can play around with different circuit architectures here too.
    qubits = cirq.GridQubit.rect(1, 16)
    symbols = sympy.symbols('alpha_0:16')
    circuit = cirq.Circuit()
    for qubit, symbol in zip(qubits, symbols):
        circuit.append(cirq.X(qubit) ** symbol)
    return circuit

def my_embedding_operators():
    # Get the measurement operators to go along with your circuit.
    qubits = cirq.GridQubit.rect(1, 16)
    return [cirq.Z(qubit) for qubit in qubits]

def create_hybrid_model():
    # A LeNet with a quantum twist.
    images_in         = tf.keras.layers.Input(shape=(28,28,1))
    dummy_input       = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string) # dummy input needed for keras to be happy.
    conv1             = tf.keras.layers.Conv2D(32, [3, 3], activation='relu')(images_in)
    conv2             = tf.keras.layers.Conv2D(64, [3, 3], activation='relu')(conv1)
    pool1             = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout1          = tf.keras.layers.Dropout(0.25)(pool1)
    flat1             = tf.keras.layers.Flatten()(dropout1)
    dense1            = tf.keras.layers.Dense(128, activation='relu')(flat1)
    dropout2          = tf.keras.layers.Dropout(0.5)(dense1)
    dense2            = tf.keras.layers.Dense(16)(dropout2)
    quantum_embedding = tfq.layers.ControlledPQC(
        my_embedding_circuit(), my_embedding_operators())([dummy_input, dense2])
    output            = tf.keras.layers.Dense(10)(quantum_embedding)

    model = tf.keras.Model(inputs = [images_in, dummy_input], outputs=[output])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    return model


hybrid_model = create_hybrid_model()

hybrid_model.summary()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

dummy_train = tfq.convert_to_tensor([cirq.Circuit() for _ in range(len(x_train))])

hybrid_model.fit(
      x=(x_train, dummy_train), y=y_train,
      batch_size=32,
      epochs=5,
      verbose=1)
      
      
hybrid_model.summary()      
