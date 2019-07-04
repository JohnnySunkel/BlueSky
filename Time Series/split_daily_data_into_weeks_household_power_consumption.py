# Split the household power consumption dataset
# into standard weeks
from numpy import split, array
from pandas import read_csv

# Split a univariate dataset into train/test sets
def split_dataset(data):
    # Split into standard weeks
    train, test = data[1: -328], data[-328: -6]
    # Restructure into windows of weekly data
    train = array(split(train, len(train) / 7))
    test = array(split(test, len(test) / 7))
    return train, test

# Load the dataset
dataset = read_csv('household_power_consumption_days.csv',
                   header = 0,
                   infer_datetime_format = True,
                   parse_dates = ['datetime'],
                   index_col = ['datetime'])
train, test = split_dataset(dataset.values)
# Validate the training data
print(train.shape)
print(train[0, 0, 0], train[-1, -1, 0])
# Validate test data
print(test.shape)
print(test[0, 0, 0], test[-1, -1, 0])
