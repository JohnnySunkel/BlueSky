# CNN LSTM forecast for the monthly car sales dataset
from math import sqrt
from numpy import array, mean, std
from pandas import DataFrame, concat, read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from matplotlib import pyplot as plt

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

# Fit a model
def model_fit(train, config):
    # Unpack config
    n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
    n_input = n_seq * n_steps
    # Prepare the data
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    train_x = train_x.reshape((train_x.shape[0], n_seq, n_steps, 1))
    # Define the CNN input model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters = n_filters,
                                     kernel_size = n_kernel,
                                     activation = 'relu'),
                                     input_shape = (None, n_steps, 1)))
    model.add(TimeDistributed(Conv1D(filters = n_filters,
                                     kernel_size = n_kernel,
                                     activation = 'relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
    model.add(TimeDistributed(Flatten()))
    # Define the LSTM output model
    model.add(LSTM(n_nodes,
                   activation = 'relu'))
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
    

# Forecast with a pre-fit model
def model_predict(model, history, config):
    # Unpack config
    n_seq, n_steps, _, _, _, _, _ = config
    n_input = n_seq * n_steps
    # Prepare the data
    x_input = array(history[-n_input:]).reshape((1, n_seq, n_steps, 1))
    # Forecast
    y_hat = model.predict(x_input, verbose = False)
    return y_hat[0]

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

# Repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats = 30):
    # Fit and evaluate the model n times
    scores = [walk_forward_validation(data, 
                                      n_test, 
                                      config) for _ in range(n_repeats)]
    return scores

# Summarize model performance
def summarize_scores(name, scores):
    # Print a summary
    scores_m, score_std = mean(scores), std(scores)
    print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
    # Box and whisker plot
    plt.boxplot(scores)
    plt.show()
    
series = read_csv('monthly-car-sales.csv',
                  header = 0,
                  index_col = 0)
data = series.values
# Split the dataset
n_test = 12
# Define configuration
config = [3, 12, 64, 3, 100, 200, 100]
# Grid search
scores = repeat_evaluate(data, config, n_test)
# Summarize scores
summarize_scores('CNN-LSTM', scores)
