""" Model of the neural network for quantum computations """

import tensorflow as tf
from tensorflow import keras


json_file = 'rbm.json'

input1 = keras.layers.Input(shape=(74,))
x1 = keras.layers.Dense(74, activation='relu')(input1)

model = keras.models.Model(inputs=[input1], outputs=x1)
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# model.load_weights(json_file)
# model.fit(train_images, train_labels, epochs=5)
# model = model_from_json(json_file)

