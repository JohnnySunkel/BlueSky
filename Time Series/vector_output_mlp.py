# Multiple parallel series MLP
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import Dense

# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequences[i: end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Define input sequences
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

# Reshape the data to [rows, columns]
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# Stack the columns horizontally
dataset = hstack((in_seq1, in_seq2, out_seq))

# Choose the number of time steps
n_steps = 3

# Convert into input/output
X, y = split_sequences(dataset, n_steps)

# Flatten the inputs
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))

# Determine the number of outputs
n_output = y.shape[1]

# Define the model
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = n_input))
model.add(Dense(n_output))
model.compile(optimizer = 'adam', loss = 'mse')

# Fit the model
model.fit(X, y, epochs = 2000, verbose = 0)

# Make predictions
x_input = array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])
x_input = x_input.reshape((1, n_input))
y_hat = model.predict(x_input, verbose = 0)
print(y_hat)
