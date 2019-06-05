# Univariate time series MLP
from numpy import array
from keras.models import Sequential
from keras.layers import Dense

# Split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the end of the sequence
        if end_ix > len(sequence) - 1:
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i: end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Define an input sequence
raw_seq = [318, 409, 330, 
           338, 345, 294,
           292, 347, 371,
           422, 465, 367,
           332, 350, 350, 
           343, 352, 358, 
           334, 378, 341, 
           292, 309, 335, 
           292, 324, 323]

# Choose the number of time steps
n_steps = 3

# Split into samples
X, y = split_sequence(raw_seq, n_steps)

# Define the model
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = n_steps))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')

# Fit the model
model.fit(X, y, epochs = 2000, verbose = 0)

# Make predictions
x_input = array([292, 324, 323])
x_input = x_input.reshape((1, n_steps))
y_hat = model.predict(x_input, verbose = 0)
print(y_hat)
