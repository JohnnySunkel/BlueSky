# Load time series dataset
# import pandas as pd
# series = pd.read_csv('filename.csv', header = 0, index_col = 0)


# Example of defining a dataset
from numpy import array

# Define the dataset
data = list()
n = 5000
for i in range(n):
    data.append([i + 1, (i + 1) * 10])
data = array(data)
# Drop the time dimension (first column)
data = data[:, 1]
# Split the data into 25 sub-sets of 200 time steps each
samples = list()
length = 200
# Step over the 5000 samples in jumps of 200
for i in range(0, n, length):
    # Grab from i to i + 200
    sample = data[i: i + length]
    samples.append(sample)
# Convert the list of arrays into a 2D array
data = array(samples)
# Reshape into [samples, timesteps, features]
data = data.reshape((len(samples), length, 1))
print(data.shape)
