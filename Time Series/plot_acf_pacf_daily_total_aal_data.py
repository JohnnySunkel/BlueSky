# ACF and PACF plots of the daily total aal dataset
from numpy import split, array
from pandas import read_csv
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Split a univariate dataset into train/test sets
def split_dataset(data):
    # Split into standard weeks
    train, test = data[0: 728], data[728: 819]
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
dataset = read_csv('daily_total_aal_data_no_apr_simple.csv',
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
lags = 20

# ACF
axis = plt.subplot(2, 1, 1)
plot_acf(series, ax = axis, lags = lags)

# PACF
axis = plt.subplot(2, 1, 2)
plot_pacf(series, ax = axis, lags = lags)

# Show the plot
plt.show
