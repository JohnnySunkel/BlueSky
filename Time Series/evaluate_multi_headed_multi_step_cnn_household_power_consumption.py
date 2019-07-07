# Multi-headed multi-step CNN for the household
# power consumption dataset
from math import sqrt
from numpy import array, split
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Model
from keras.layers.merge import concatenate

# Split a univariate dataset into train/test sets
def split_dataset(data):
    # Split into standard weeks
    train, test = data[1: -328], data[-328: -6]
    # Restructure into windows of weekly data
    train = array(split(train, len(train) / 7))
    test = array(split(test, len(test) / 7))
    return train, test

# Evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # Calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # Calculate MSE
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # Calculate RMSE
        rmse = sqrt(mse)
        # Store
        scores.append(rmse)
    # Calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores

# Summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))
    
# Convert history into inputs and outputs
def to_supervised(train, n_input, n_out = 7):
    # Flatten the data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # Step over the entire history one time step at a time
    for _ in range(len(data)):
        # Define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # Ensure we have enough data for this instance
        if out_end < len(data):
            X.append(data[in_start: in_end, :])
            y.append(data[in_end: out_end, 0])
        # Move along one time step
        in_start += 1
    return array(X), array(y)

# Plot training history
def plot_history(history):
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label = 'train')
    plt.plot(history.history['val_loss'], label = 'test')
    plt.title('loss', y = 0, loc = 'center')
    plt.legend()
    # Plot RMSE
    plt.subplot(2, 1, 2)
    plt.plot(history.history['rmse'], label = 'train')
    plt.plot(history.history['val_rmse'], label = 'test')
    plt.title('rmse', y = 0, loc = 'center')
    plt.legend()
    plt.show()
    
# Train the model
def build_model(train, n_input):
    # Prepare the data
    train_x, train_y = to_supervised(train, n_input)
    # Define parameters
    verbose = False, 
    epochs = 25
    batch_size = 16
    n_timesteps = train_x.shape[1]
    n_features = train_x.shape[2]
    n_outputs = train_y.shape[1]
    # Create a channel for each variable
    in_layers, out_layers = list(), list()
    for _ in range(n_features):
        inputs = Input(shape = (n_timesteps, 1))
        conv1 = Conv1D(filters = 32,
                       kernel_size = 3,
                       activation = 'relu')(inputs)
        conv2 = Conv1D(filters = 32,
                       kernel_size = 3,
                       activation = 'relu')(conv1)
        pool1 = MaxPooling1D(pool_size = 2)(conv2)
        flat = Flatten()(pool1)
        # Store layers
        in_layers.append(inputs)
        out_layers.append(flat)
    # Merge heads
    merged = concatenate(out_layers)
    # Interpretation
    dense1 = Dense(200, activation = 'relu')(merged)
    dense2 = Dense(100, activation = 'relu')(dense1)
    outputs = Dense(n_outputs)(dense2)
    model = Model(inputs = in_layers, outputs = outputs)
    # Compile model
    model.compile(optimizer = 'adam', loss = 'mse')
    # Plot the model
    # plot_model(model, 
    #            show_shapes = True,
    #            to_file = 'multiheaded_cnn.png')
    # Fit the model
    input_data = [train_x[:, :, i].reshape((train_x.shape[0], 
              n_timesteps, 1)) for i in range(n_features)]
    model.fit(input_data,
              train_y, 
              epochs = epochs,
              batch_size = batch_size,
              verbose = verbose)
    return model
          
# Make a forecast
def forecast(model, history, n_input):
    # Flatten the data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # Retrieve last observations for input data
    input_x = data[-n_input:, :]
    # Reshape into n input arrays
    input_x = [input_x[:, i].reshape((1, 
               input_x.shape[0], 1)) for i in range(input_x.shape[1])]
    # Forecast the next week
    y_hat = model.predict(input_x, verbose = False)
    # We only want the vector forecast
    y_hat = y_hat[0]
    return y_hat

# Evaluate a single model
def evaluate_model(train, test, n_input):
    # Fit the model
    model = build_model(train, n_input)
    # History is a list of weekly data
    history = [x for x in train]
    # Walk forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # Predict the week
        y_hat_sequence = forecast(model, history, n_input)
        # Store the predictions
        predictions.append(y_hat_sequence)
        # Get real observation and add to history for
        # predicting the next week
        history.append(test[i, :])
    # Evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

# Load the dataset
dataset = read_csv('household_power_consumption_days.csv',
                   header = 0,
                   infer_datetime_format = True,
                   parse_dates = ['datetime'],
                   index_col = ['datetime'])

# Split into train/test sets
train, test = split_dataset(dataset.values)

# Evaluate the model and get scores
n_input = 7
score, scores = evaluate_model(train, test, n_input)

# Summarize the scores
summarize_scores('CNN', score, scores)

# Plot the scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
plt.plot(days, scores, marker = 'o', label = 'CNN')
plt.show()
