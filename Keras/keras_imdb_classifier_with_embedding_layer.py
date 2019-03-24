from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense


# Number of words to consider as features
max_features = 10000

# Cut off the text after this number of words (among the
# max_features most common words).
maxlen = 20

# Load the data as lists of integers
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words = max_features)

# Turn the lists of integers into 2D tensors of 
# shape (samples, max_len)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen = maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen = maxlen)

# Define the model
model = Sequential()
model.add(Embedding(10000, 8, input_length = maxlen))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

# Compile the model
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['acc'])

model.summary()

# Train the model
history = model.fit(x_train,
                    y_train,
                    epochs = 10,
                    batch_size = 32,
                    validation_split = 0.20)
