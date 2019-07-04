# Load and clean up the household power consumption dataset
from numpy import nan, isnan
from pandas import read_csv

# Fill missing values with a value at the same time 24 hours ago
def fill_missing(values):
    one_day = 60 * 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if isnan(values[row, col]):
                values[row, col] = values[row - one_day, col]

# Load the dataset
dataset = read_csv('household_power_consumption.txt',
                   sep = ';',
                   header = 0,
                   low_memory = False,
                   infer_datetime_format = True,
                   parse_dates = {'datetime': [0, 1]},
                   index_col = ['datetime'])

# Mark all missing values
dataset.replace('?', nan, inplace = True)

# Make the dataset numeric
dataset = dataset.astype('float32')
                
# Fill missing
fill_missing(dataset.values)

# Add a column for the remainder of the sub-metering
values = dataset.values
dataset['sub_metering_4'] = (values[:, 0] * 1000 / 60) - \
    (values[:, 4] + values[:, 5] + values[:, 6])
    
# Save the updated dataset
dataset.to_csv('household_power_consumption.csv')
