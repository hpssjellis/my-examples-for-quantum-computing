@qml.qnode(dev)
def circuit(inputs, conv_params):
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(np.pi * inputs[j], wires=j)

    # Random quantum circuit
    RandomLayers(conv_params, wires=list(range(4)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

qlayer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=4)

def MyModel():
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential([
        qlayer,
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)), 
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),                                                 
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


q_model = MyModel()

q_history = q_model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    batch_size=56,
    epochs=n_epochs,
    verbose=2,
)