# Multivariate multi-step output LSTM
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Split a multivariate time series into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequences)):
        # Find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the dataset
        if out_end_ix > len(sequences) - 1:
            break
        # Gather the input and output parts of this pattern
        seq_x = sequences[i: end_ix, :-1]
        seq_y = sequences[end_ix: out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Create synthetic input sequences
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

# Convert to [rows, columns]
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# Stack the columns horizontally
dataset = hstack((in_seq1, in_seq2, out_seq))

# Set the number of time steps
n_steps_in, n_steps_out = 3, 2

# Convert to input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)

# Set the number of features
n_features = X.shape[2]

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
x_input = array([[90, 95], [100, 105], [110, 115]])
x_input = x_input.reshape((1, n_steps_in, n_features))
y_hat = model.predict(x_input, verbose = False)
print(y_hat)
