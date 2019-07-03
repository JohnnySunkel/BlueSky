# Line plots for the household power consumption dataset
from pandas import read_csv
from matplotlib import pyplot as plt

# Load the dataset
dataset = read_csv('household_power_consumption.csv',
                   header = 0,
                   infer_datetime_format = True,
                   parse_dates = ['datetime'],
                   index_col = ['datetime'])

# Create a Line plot for each variable
plt.figure()
for i in range(len(dataset.columns)):
    # Create subplot
    plt.subplot(len(dataset.columns), 1, i + 1)
    # Get variable name
    name = dataset.columns[i]
    # Plot the data
    plt.plot(dataset[name])
    # Title
    plt.title(name, y = 0)
    # Turn off tick marks to remove clutter
    plt.yticks([])
    plt.xticks([])
plt.show()
