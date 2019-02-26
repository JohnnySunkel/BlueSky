from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential


# Number of words to consider as features
max_features = 10000

# Cut off texts after this number of words
maxlen = 500

# Load the data
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words = max_features)

# Reverse the sequences
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

# Pad the sequences
x_train = sequence.pad_sequences(x_train, maxlen = maxlen)
x_test = sequence.pad_sequences(x_test, maxlen = maxlen)

# Define the model
model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation = 'sigmoid'))

# Compile the model
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['acc'])

# Train the model
history = model.fit(x_train,
                    y_train,
                    epochs = 10,
                    batch_size = 128,
                    validation_split = 0.20)
