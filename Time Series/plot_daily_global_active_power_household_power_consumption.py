# Daily line plots for the household power consumption dataset
from pandas import read_csv
from matplotlib import pyplot as plt

# Load the dataset
dataset = read_csv('household_power_consumption.csv',
                   header = 0,
                   infer_datetime_format = True,
                   parse_dates = ['datetime'],
                   index_col = ['datetime'])

# Plot active power for each day
days = [x for x in range(1, 20)]
plt.figure()
for i in range(len(days)):
    # Prepare subplot
    ax = plt.subplot(len(days), 1, i + 1)
    # Determine the day to plot
    day = '2007-01-' + str(days[i])
    # Get all observations for the day
    result = dataset[day]
    # Plot the active power for the day
    plt.plot(result['Global_active_power'])
    # Add a title
    plt.title(day, y = 0, loc = 'left', size = 6)
    # Turn off tick marks to remove clutter
    plt.yticks([])
    plt.xticks([])
plt.show()
