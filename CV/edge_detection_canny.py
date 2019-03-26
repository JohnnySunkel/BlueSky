# Import required packages.
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read image.
image = cv2.imread('beach_pic.jpg')

# Calculate edges.
edges = cv2.Canny(image, 100, 200)

# Plot edges.
plt.imshow(edges)
