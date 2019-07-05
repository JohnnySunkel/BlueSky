# Evaluate naive forecast strategies for the household
# power consumption dataset
from math import sqrt
from numpy import split, array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

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

# Daily persistence model
def daily_persistence(history):
    # Get the data for the prior week
    last_week = history[-1]
    # Get the total active power for the last day
    value = last_week[-1, 0]
    # Prepare 7 day forecast
    forecast = [value for _ in range(7)]
    return forecast

# Weekly persistence model
def weekly_persistence(history):
    # Get the data for the prior week
    last_week = history[-1]
    return last_week[:, 0]

# Week one year ago persistence model
def week_one_year_ago_persistence(history):
    # Get the data for the prior week one year ago
    last_week = history[-52]
    return last_week[:, 0]

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
models['daily'] = daily_persistence
models['weekly'] = weekly_persistence
models['week-oya'] = week_one_year_ago_persistence

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
