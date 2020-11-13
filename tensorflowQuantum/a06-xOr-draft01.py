import tensorflow as tf
import numpy as np 


x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([[0],[1],[1],[0]])



model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, input_shape=(2,)))
model.add(tf.keras.layers.Dense(2))

#predictions = model(x_train[:1]).np()
#predictions

#tf.nn.softmax(predictions).np()

#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

#loss_fn(y_train[:1], predictions).np()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000)

#fair_nn_results =  model.evaluate(x_train, y_train, verbose=2)

#myResults =  model.evaluate(x_train, y_train, verbose=2)
myResults =  model.predict(x_train, verbose=0)

#probability_model = tf.keras.Sequential([
#  model,
#  tf.keras.layers.Softmax()
#])

#probability_model(x_train)

print(x_train)
print(myResults)


#model.save('./')



