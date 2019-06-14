# Parallel input multi-step CNN
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # Gather the input and output parts of the pattern
        seq_x = sequences[i: end_ix, :]
        seq_y = sequences[end_ix: out_end_ix, :]
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

# Stack the columns horizontally
dataset = hstack((in_seq1, in_seq2, out_seq))

# Set the number of time steps
n_steps_in, n_steps_out = 3, 2

# Convert to input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
    
# Flatten the outputs
n_output = y.shape[1] * y.shape[2]
y = y.reshape((y.shape[0], n_output))

# Set the number of features
n_features = X.shape[2]

# Define the model
model = Sequential()
model.add(Conv1D(filters = 64, 
                 kernel_size = 2,
                 activation = 'relu',
                 input_shape = (n_steps_in,
                                n_features)))
model.add(MaxPooling1D(pool_size = 2))
model.add(Flatten())
model.add(Dense(50, activation = 'relu'))
model.add(Dense(n_output))
model.compile(optimizer = 'adam', loss = 'mse')

# Fit the model
model.fit(X, y, epochs = 2000, verbose = 0)

# Make predictions
x_input = array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])
x_input = x_input.reshape((1, n_steps_in, n_features))
y_hat = model.predict(x_input, verbose = False)
print(y_hat)
