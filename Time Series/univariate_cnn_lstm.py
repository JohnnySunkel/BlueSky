# Univariate CNN LSTM
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D

# Split a univariate time series into samples
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # Gather input and output parts of this pattern
        seq_x = sequence[i: end_ix]
        seq_y = sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Create an input sequence
seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Set the number of time steps
n_steps = 4

# Split into samples
X, y = split_sequence(seq, n_steps)

# Reshape from [samples, timesteps] to [samples,
# subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

# Define the input CNN model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters = 64,
                                 kernel_size = 1,
                                 activation = 'relu'),
                                 input_shape = (None,
                                                n_steps,
                                                n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
model.add(TimeDistributed(Flatten()))

# Define the output LSTM model
model.add(LSTM(50, activation = 'relu'))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse')

# Fit the model
model.fit(X, y, epochs = 2000, verbose = False)

# Make predictions
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
y_hat = model.predict(x_input, verbose = False)
print(y_hat)
