# Build a Perceptron based classifier

# import packages
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# load input data
text = np.loadtxt('data_perceptron.txt')

# separate the data into datapoints and labels
data = text[:, :2]
labels = text[:, 2].reshape((text.shape[0], 1))

# plot the data
plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

# define the minimum and maximum values that each dimension can take
dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1

# number of neurons in the output layer
num_output = labels.shape[1]

# define a perceptron with 2 input neurons (one neuron
# for each dimension of the input data)
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
perceptron = nl.net.newp([dim1, dim2], num_output)

# train the perceptron
error_progress = perceptron.train(data, 
                                  labels, 
                                  epochs = 100, 
                                  show = 20,
                                  lr = 0.03)

# plot the training progress
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()

plt.show()