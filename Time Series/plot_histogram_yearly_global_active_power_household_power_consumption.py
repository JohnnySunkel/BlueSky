# Yearly histogram plots for the household 
# power consumption dataset
from pandas import read_csv
from matplotlib import pyplot as plt

# Load the dataset
dataset = read_csv('household_power_consumption.csv',
                   header = 0,
                   infer_datetime_format = True,
                   parse_dates = ['datetime'],
                   index_col = ['datetime'])

# Plot active power for each year
years = ['2007', '2008', '2009', '2010']
plt.figure()
for i in range(len(years)):
    # Prepare subplot
    ax = plt.subplot(len(years), 1, i + 1)
    # Determine the year to plot
    year = years[i]
    # Get all observations for the year
    result = dataset[str(year)]
    # Plot the active power for the year
    result['Global_active_power'].hist(bins = 100)
    # Zoom in on the distribution
    ax.set_xlim(0, 5)
    # Add a title
    plt.title(str(year), y = 0, loc = 'right')
    # Turn off ticks to avoid clutter
    plt.xticks([])
    plt.yticks([])
plt.show()
