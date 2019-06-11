# Multivariate multi-headed 1D CNN
from numpy import array, hstack
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate

# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = [], []
    for i in range(len(sequences)):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # Gather input and output parts of the pattern
        seq_x = sequences[i: end_ix, :-1]
        seq_y = sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Define the input sequences
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

# Reshape to [rows, columns]
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# Stack the columns horizontally
dataset = hstack((in_seq1, in_seq2, out_seq))

# Set the number of time steps
n_steps = 3

# Convert to input/output
X, y = split_sequences(dataset, n_steps)

# One time series per head
n_features = 1

# Separate the input data
X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], n_features)
X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], n_features)

# First input model
visible1 = Input(shape = (n_steps, n_features))
cnn1 = Conv1D(filters = 64, kernel_size = 2, activation = 'relu')(visible1)
cnn1 = MaxPooling1D(pool_size = 2)(cnn1)
cnn1 = Flatten()(cnn1)

# Second input model
visible2 = Input(shape = (n_steps, n_features))
cnn2 = Conv1D(filters = 64, kernel_size = 2, activation = 'relu')(visible2)
cnn2 = MaxPooling1D(pool_size = 2)(cnn2)
cnn2 = Flatten()(cnn2)

# Merge input models
merge = concatenate([cnn1, cnn2])
dense = Dense(50, activation = 'relu')(merge)
output = Dense(1)(dense)

# Connect the input and output models
model = Model(inputs = [visible1, visible2], outputs = output)
model.compile(optimizer = 'adam', loss = 'mse')

# Fit the model
model.fit([X1, X2], y, epochs = 2000, verbose = 0)

# Reshape one sample for making predictions
x_input = array([[70, 75], [80, 85], [90, 95]])
x1 = x_input[:, 0].reshape((1, n_steps, n_features))
x2 = x_input[:, 1].reshape((1, n_steps, n_features))
y_hat = model.predict([x1, x2], verbose = 0)
print(y_hat)
