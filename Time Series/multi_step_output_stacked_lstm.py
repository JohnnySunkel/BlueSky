# Multi-step LSTM
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # Gather the input and output parts of the pattern
        seq_x = sequence[i: end_ix]
        seq_y = sequence[end_ix: out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
        
# Create an input sequence
seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Set the number of time steps
n_steps_in, n_steps_out = 3, 2

# Split into samples
X, y = split_sequence(seq, n_steps_in, n_steps_out)

# Reshape from [samples, timesteps] to [samples,
# timesteps, features] 
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Define the model
model = Sequential()
model.add(LSTM(100,
               activation = 'relu',
               return_sequences = True,
               input_shape = (n_steps_in, n_features)))
model.add(LSTM(100, activation = 'relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer = 'adam', loss = 'mse')

# Fit the model
model.fit(X, y, epochs = 2000, verbose = False)

# Make predictions
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
y_hat = model.predict(x_input, verbose = False)
print(y_hat)
