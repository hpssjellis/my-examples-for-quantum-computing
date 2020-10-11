import os
import numpy as np
import tensorflow as tf

print(tf.version.VERSION)

#(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([[0],[1],[1],[0]])
#y_train = np.array([[0,0],[1,1],[1,1],[0,0]])

#train_labels = train_labels[:1000]
#test_labels = test_labels[:1000]

#train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
#test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation="softmax"),
    tf.keras.layers.Dense(2)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

# Train the model with the new callback
https://t.co/uN1YBjl5uC(x_train, y_train, epochs=500)

myResults = model.predict(x_train)

print("results")
print( str(x_train[0]) + ", " + str(myResults[0]))
print( str(x_train[1]) + ", " + str(myResults[1]))
print( str(x_train[2]) + ", " + str(myResults[2]))
print( str(x_train[3]) + ", " + str(myResults[3]))
print()
