# ARIMA forecast for the household power consumption dataset
from math import sqrt
from numpy import split, array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

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
    
# Evaluate a single model
def evaluate_model(model_func, train, test):
    # History is a list of weekly data
    history = [x for x in train]
    # Walk forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # Predict the week
        y_hat_sequence = model_func(history)
        # Store the predictions
        predictions.append(y_hat_sequence)
        # Get real observation and add to history for
        # predicting the next week
        history.append(test[i, :])
    predictions = array(predictions)
    # Evaluate predictions days for each week
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores

# Convert windows of weekly multivariate data into
# a series of total power
def to_series(data):
    # Extract just the total power from each week
    series = [week[:, 0] for week in data]
    # Flatten into a single series
    series = array(series).flatten()
    return series

# ARIMA forecast
def arima_forecast(history):
    # Convert history into a univariate series
    series = to_series(history)
    # Define the model
    model = ARIMA(series, order = (7, 0, 0))
    # Fit the model
    model_fit = model.fit(disp = False)
    # Make forecast
    y_hat = model_fit.predict(len(series), len(series) + 6)
    return y_hat

# Load the dataset
dataset = read_csv('household_power_consumption_days.csv',
                   header = 0,
                   infer_datetime_format = True,
                   parse_dates = ['datetime'],
                   index_col = ['datetime'])

# Split into train/test sets
train, test = split_dataset(dataset.values)

# Define the names and functions for the models we 
# want to evaluate
models = dict()
models['arima'] = arima_forecast

# Evaluate each model
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
for name, func in models.items():
    # Evaluate and get scores
    score, scores = evaluate_model(func, train, test)
    # Summarize scores
    summarize_scores(name, score, scores)
    # Plot scores
    plt.plot(days, scores, marker = 'o', label = name)

# Show the plot
plt.legend()
plt.show()
