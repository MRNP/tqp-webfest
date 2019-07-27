""" Model of the neural network for quantum computations """

import tensorflow as tf
from tensorflow import keras


input1 = keras.layers.Input(shape=(74,))
x1 = keras.layers.Dense(74, activation='relu')(input1)
# x1 = tf.keras.backend.transpose(x1)
output = keras.layers.Dense(1, activation='relu')(x1)

model = keras.models.Model(inputs=[input1], outputs=output)
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=5)

