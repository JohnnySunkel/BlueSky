import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the famous Lena image
img = mpimg.imread('lena.png')

# What does it look like?
plt.imshow(img)
plt.show()

# Make it B&W
bw = img.mean(axis = 2)
plt.imshow(bw, cmap = 'gray')
plt.show()

# Create a Gaussian filter
W = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        dist = (i - 9.5)**2 + (j - 9.5)**2
        W[i, j] = np.exp(-dist / 50.)
W /= W.sum()  # normalize the kernel

# What does the filter look like?
plt.imshow(W, cmap = 'gray')
plt.show()

# Convolution
out = convolve2d(bw, W)
plt.imshow(out, cmap = 'gray')
plt.show()

# What's that weird black stuff on the edges? Check the
# size of the output
print(out.shape)
# After convolution, the output signal is N1 + N2 - 1

# We can also just make the output the same size as the input
out = convolve2d(bw, W, mode = 'same')
plt.imshow(out, cmap = 'gray')
plt.show()

# In color
out3 = np.zeros(img.shape)
print(out3.shape)
for i in range(3):
    out3[:, :, i] = convolve2d(img[:, :, i], W, mode = 'same')
# out3 /= out3.max()  # Can also do this if you didn't normalize the kernel
plt.imshow(out3)
plt.show()  # Does not look like anything
