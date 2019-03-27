# Using Numpy to sample an artificial dataset
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic regression data
N = 100
w_true = 5
b_true = 2
noise_scale = 0.1
x_np = np.random.rand(N, 1)
noise = np.random.normal(scale = noise_scale, size = (N, 1))

# Convert shape of y_np to (N,)
y_np = np.reshape(w_true * x_np + b_true + noise, (-1))

# Plot the data
plt.scatter(x_np, y_np)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 1)
plt.title('Toy Linear Regression Data, '
          r'$y = 5x + 2 + N(0, 1)$')
plt.show() 


# Generate synthetic classification data
N = 100
# Zeros form a Gaussian centered at (-1, -1)
# epsilon is 0.1
x_zeros = np.random.multivariate_normal(
    mean = np.array((-1, -1)), cov = 0.1 * np.eye(2), size = (N // 2,))
y_zeros = np.zeros((N // 2,))
# Ones form a Gaussian centered at (1, 1)
# epsilon = 0.1
x_ones = np.random.multivariate_normal(
    mean = np.array((1, 1)), cov = 0.1 * np.eye(2), size = (N // 2,))
y_ones = np.ones((N // 2,))

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])

# Plot the data
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], color = 'blue')
plt.scatter(x_ones[:, 0], x_ones[:, 1], color = 'red')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Toy Logistic Regression Data')
plt.show()
