# Generating data using Hidden Markov Models

# import packages and functions
import datetime
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from time_series import read_data

# load data from the input file
data = np.loadtxt('data_1D.txt', delimiter = ',')

# extract the third column for training
X = np.column_stack([data[:, 2]])

# create a Gaussian HMM with 5 components and diagonal covariance
num_components = 5
hmm = GaussianHMM(n_components = num_components,
                  covariance_type = 'diag',
                  n_iter = 1000)

# train the HMM
print('\nTraining the Hidden Markov Model...')
hmm.fit(X)

# print HMM statistics
print('\nMeans and variances:')
for i in range(hmm.n_components):
    print('\nHidden state', i + 1)
    print('Mean =', round(hmm.means_[i][0], 2))
    print('Variance =', round(np.diag(hmm.covars_[i])[0], 2))
    
# generate and plot 1200 samples using the trained HMM
num_samples = 1200
generated_data, _ = hmm.sample(num_samples)
plt.plot(np.arange(num_samples), generated_data[:, 0], c = 'black')
plt.title('Generated data')

plt.show()
