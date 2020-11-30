#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

im_height = 128
im_width = 128




from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """ Perform a block consisting of two layers of Conv2d, with relu activation and batchnorm if True.
        
    """
    
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x




def get_classical_net(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model




input_img = Input((im_height, im_width, 1), name='img')
model = get_classical_net(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()




import pennylane as qml

n_qubits = 5
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def quantum_classifier(inputs, weights):
    
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    return qml.expval(qml.PauliZ(0))




weight_shapes = {"weights": (16, n_qubits, 3)}




qlayer = qml.qnn.KerasLayer(quantum_classifier, weight_shapes, output_dim=1)




intermediate_layer = Model(inputs=model.input, outputs=model.get_layer('batch_normalization_18').output)




pre_out = intermediate_layer.layers[-1].output




import numpy as np
pre_out = Activation('tanh') (pre_out) * tf.constant(np.pi / 2.0)




pre_out




pre_out = qlayer(pre_out)






