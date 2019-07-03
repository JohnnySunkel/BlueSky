# Load and clean up the household power consumption data
from numpy import nan
from pandas import read_csv

# Load the dataset
dataset = read_csv('household_power_consumption.txt',
                   sep = ';',
                   header = 0,
                   low_memory = False,
                   infer_datetime_format = True,
                   parse_dates = {'datetime': [0, 1]},
                   index_col = ['datetime'])
# Summarize the dataset
print(dataset.shape)
print(dataset.head())

# Mark all missing values
dataset.replace('?', nan, inplace = True)

# Add a column for the remainder of the sub-metering
values = dataset.values.astype('float32')
dataset['sub_metering_4'] = (values[:, 0] * 1000 / 60) - \
    (values[:, 4] + values[:, 5] + values[:, 6])
    
# Save the updated dataset
dataset.to_csv('household_power_consumption.csv')

# Load the new dataset
dataset = read_csv('household_power_consumption.csv', 
                   header = 0,
                   infer_datetime_format = True,
                   parse_dates = ['datetime'],
                   index_col = ['datetime'])
print(dataset.head())
