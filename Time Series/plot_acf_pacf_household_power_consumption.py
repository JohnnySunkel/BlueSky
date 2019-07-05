# ACF and PACF plots of the household power consumption dataset
from numpy import split, array
from pandas import read_csv
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Split a univariate dataset into train/test sets
def split_dataset(data):
    # Split into standard weeks
    train, test = data[1: -328], data[-328: -6]
    # Restructure into windows of weekly data
    train = array(split(train, len(train) / 7))
    test = array(split(test, len(test) / 7))
    return train, test

# Convert windows of weekly multivariate data into
# a series of total power
def to_series(data):
    # Extract just the total power from each week
    series = [week[:, 0] for week in data]
    # Flatten into a single series
    series = array(series).flatten()
    return series

# Load the dataset
dataset = read_csv('household_power_consumption_days.csv',
                   header = 0,
                   infer_datetime_format = True,
                   parse_dates = ['datetime'],
                   index_col = ['datetime'])

# Split into train/test sets
train, test = split_dataset(dataset.values)

# Convert training data into a series
series = to_series(train)

# Plots
plt.figure()
lags = 365

# ACF
axis = plt.subplot(2, 1, 1)
plot_acf(series, ax = axis, lags = lags)

# PACF
axis = plt.subplot(2, 1, 2)
plot_pacf(series, ax = axis, lags = lags)

# Show the plot
plt.show
