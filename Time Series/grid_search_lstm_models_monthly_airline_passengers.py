# Grid search LSTM models for the monthly
# airline passengers dataset
from numpy import array, mean
from math import sqrt
from pandas import DataFrame, concat, read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# Transform list into supervised learning format
def series_to_supervised(data, n_in, n_out = 1):
    df = DataFrame(data)
    cols = list()
    # Input sequence (t - n, ... t - 1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # Forecast sequence (t, t + 1, ... t + n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # Put it all together
    agg = concat(cols, axis = 1)
    # Drop rows with NaN values
    agg.dropna(inplace = True)
    return agg.values

# Calculate the RMSE
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# Difference the dataset
def difference(data, order):
    return [data[i] - data[i - order] for i in range(order, len(data))]

# Fit a model
def model_fit(train, config):
    # Unpack config
    n_input, n_nodes, n_epochs, n_batch, n_diff = config
    # Prepare the data
    if n_diff > 0:
        train = difference(train, n_diff)
    # Transform series into supervised format
    data = series_to_supervised(train, n_in = n_input)
    # Separate inputs and outputs
    train_x, train_y = data[:, :-1], data[:, -1]
    # Reshape input data into [samples, timesteps, features]
    n_features = 1
    train_x = train_x.reshape((train_x.shape[0],
                               train_x.shape[1],
                               n_features))
    # Define the model
    model = Sequential()
    model.add(LSTM(n_nodes, 
                   activation = 'relu', 
                   input_shape = (n_input, n_features)))
    model.add(Dense(n_nodes, activation = 'relu'))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'mse')
    # Fit the model
    model.fit(train_x, 
              train_y,
              epochs = n_epochs,
              batch_size = n_batch,
              verbose = False)
    return model

# Forecast with the fit model
def model_predict(model, history, config):
    # Unpack config
    n_input, _, _, _, n_diff = config
    # Prepare the data
    correction = 0.0
    if n_diff > 0:
        correction = history[-n_diff]
        history = difference(history, n_diff)
    # Shape input for the model
    x_input = array(history[-n_input:]).reshape((1, n_input, 1))
    # Make forecast
    y_hat = model.predict(x_input, verbose = False)
    # Correct the forecast if it was differenced
    return correction + y_hat[0]

# Walk forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # Split the dataset
    train, test = train_test_split(data, n_test)
    # Fit the model
    model = model_fit(train, cfg)
    # Seed history with training dataset
    history = [x for x in train]
    # Step over each time step in the test set
    for i in range(len(test)):
        # Fit model and make a forecast for history
        y_hat = model_predict(model, history, cfg)
        # Store the forecast in a list of predictions
        predictions.append(y_hat)
        # Add actual observations to history for the next loop
        history.append(test[i])
    # Estimate prediction error
    error = measure_rmse(test, predictions)
    print(' > %.3f' % error)
    return error

# Repeat evaluation of a model configuration
def repeat_evaluate(data, config, n_test, n_repeats = 10):
    # Convert config to a key
    key = str(config)
    # Fit and evaluate the model n times
    scores = [walk_forward_validation(data, 
                                      n_test, 
                                      config) for _ in range(n_repeats)]
    # Summarize scores
    result = mean(scores)
    print('> Model[%s] %.3f' % (key, result))
    return (key, result)

# Grid search model configurations
def grid_search(data, cfg_list, n_test):
    # Evaluate configs
    scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
    # Sort configs by error in ascending order
    scores.sort(key = lambda tup: tup[1])
    return scores

# Create a list of configs to try
def model_configs():
    # Define scope of configs
    n_input = [12]
    n_nodes = [100]
    n_epochs = [50]
    n_batch = [1, 150]
    n_diff = [12]
    # Create configs
    configs = list()
    for i in n_input:
        for j in n_nodes:
            for k in n_epochs:
                for l in n_batch:
                    for m in n_diff:
                        cfg = [i, j, k, l, m]
                        configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs

# Load the dataset
series = read_csv('monthly-airline-passengers.csv',
                  header = 0,
                  index_col = 0)
data = series.values
# Split the data
n_test = 12
# Model configs
cfg_list = model_configs()
# Grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# List the top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)
