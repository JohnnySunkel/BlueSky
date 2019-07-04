# Monthly histogram plots for the household 
# power consumption dataset
from pandas import read_csv
from matplotlib import pyplot as plt

# Load the dataset
dataset = read_csv('household_power_consumption.csv',
                   header = 0,
                   infer_datetime_format = True,
                   parse_dates = ['datetime'],
                   index_col = ['datetime'])

# Plot active power for each month
months = [x for x in range(1, 13)]
plt.figure()
for i in range(len(months)):
    # Prepare subplot
    ax = plt.subplot(len(months), 1, i + 1)
    # Determine the month to plot
    month = '2007-' + str(months[i])
    # Get all observations for the year
    result = dataset[month]
    # Plot the active power for the month
    result['Global_active_power'].hist(bins = 100)
    # Zoom in on the distribution
    ax.set_xlim(0, 5)
    # Add a title
    plt.title(month, y = 0, loc = 'right')
    # Turn off ticks to avoid clutter
    plt.xticks([])
    plt.yticks([])
plt.show()
