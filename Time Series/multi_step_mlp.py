# Multi-step MLP
from numpy import array
from keras.models import Sequential
from keras.layers import Dense

# Split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # Gather the input and output parts of the pattern
        seq_x, seq_y = sequence[i: end_ix], sequence[end_ix: out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Create a toy input sequence
series = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Set the number of input and output time steps
n_steps_in, n_steps_out = 3, 2

# Split into samples
X, y = split_sequence(series, n_steps_in, n_steps_out)

# Define the model
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = n_steps_in))
model.add(Dense(n_steps_out))
model.compile(optimizer = 'adam', loss = 'mse')

# Fit the model
model.fit(X, y, epochs = 2000, verbose = 0)

# Make predictions
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in))
y_hat = model.predict(x_input, verbose = 0)
print(y_hat)
