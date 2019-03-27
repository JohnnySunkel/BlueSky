from keras.models import Sequential, Model
from keras import Input, layers
import numpy as np


# Define a sequential model
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation = 'relu', input_shape = (64,)))
seq_model.add(layers.Dense(32, activation = 'relu'))
seq_model.add(layers.Dense(10, activation = 'softmax'))

seq_model.summary()


# Define an equivalent model using the Keras functional API
input_tensor = Input(shape = (64,))
x = layers.Dense(32, activation = 'relu')(input_tensor)
x = layers.Dense(32, activation = 'relu')(x)
output_tensor = layers.Dense(10, activation = 'softmax')(x)

model = Model(input_tensor, output_tensor)

model.summary()

# Compile the model
model.compile(optimizer = 'rmsprop', 
              loss = 'categorical_crossentropy')

# Generate dummy Numpy data to train on
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

# Train and evaluate the model
model.fit(x_train,
          y_train,
          epochs = 10,
          batch_size = 128)

score = model.evaluate(x_train, y_train)
