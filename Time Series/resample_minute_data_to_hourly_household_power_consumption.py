# Resample minute data to daily data for the
# household power consumption dataset
from pandas import read_csv

# Load the dataset
dataset = read_csv('household_power_consumption.csv',
                   header = 0,
                   infer_datetime_format = True,
                   parse_dates = ['datetime'],
                   index_col = ['datetime'])

# Resample data to daily
daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()

# Summarize the new dataset
print(daily_data.shape)
print(daily_data.head())

# Save the new dataset
daily_data.to_csv('household_power_consumption_days.csv')
