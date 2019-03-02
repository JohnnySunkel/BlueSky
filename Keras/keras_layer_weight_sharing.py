from keras import Input, layers
from keras.models import Model


# Instantiate a single LSTM layer, once
lstm = layers.LSTM(32)

# Build the left branch of the model. Inputs are variable-length
# sequences of vectors of size 128.
left_input = Input(shape = (None, 128))
left_output = lstm(left_input)

# Build the right branch of the model. When you call an
# existing layer instance, you reuse it's weights.
right_input = Input(shape = (None, 128))
right_output = lstm(right_input)

# Build the classifier on top
merged = layers.concatenate([left_output, right_output], axis = -1)
predictions = layers.Dense(1, activation = 'sigmoid')(merged)

# Instantiate and train the model.When you train such a model, the
# weights of the LSTM layer are updated based on both inputs.
model = Model([left_input, right_input], predictions)
model.fit([left_data, right_data], targets)
