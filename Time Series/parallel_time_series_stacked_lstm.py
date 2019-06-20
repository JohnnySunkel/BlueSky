# Parallel time series LSTM
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Split parallel time series into samples
def split_sequences(sequences, n_steps):
    X, y = [], []
    for i in range(len(sequences)):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # Gather input and output parts of the pattern
        seq_x = sequences[i: end_ix, :]
        seq_y = sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Create input sequences
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

# Convert to [rows, columns]
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# Stack the columns horizantally
dataset = hstack((in_seq1, in_seq2, out_seq))

# Set the number of time steps
n_steps = 3

# Convert to input/output
X, y = split_sequences(dataset, n_steps)

# Set the number of features
n_features = X.shape[2]

# Define the model
model = Sequential()
model.add(LSTM(100, 
               activation = 'relu',
               return_sequences = True,
               input_shape = (n_steps, n_features)))
model.add(LSTM(100, activation = 'relu'))
model.add(Dense(n_features))
model.compile(optimizer = 'adam', loss = 'mse')

# Fit the model
model.fit(X, y, epochs = 2000, verbose = False)

# Make predictions
x_input = array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])
x_input = x_input.reshape((1, n_steps, n_features))
y_hat = model.predict(x_input, verbose = False)
print(y_hat)
