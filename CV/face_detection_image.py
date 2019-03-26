# Import packages.
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the Haar cascade files for face and eye detection
face_cascade = cv2.CascadeClassifier(
    'haar_cascade_files/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(
    'haar_cascade_files/haarcascade_eye.xml')

# Check if the face cascade file has been loaded correctly
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
    
# Check if the eye cascade file has been loaded correctly
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')
    
# Read image and convert to grayscale.
image = cv2.imread('beach_pic.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate coordinates.
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y: y + h, x: x + w]
    roi_color = image[y: y + h, x: x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    # Draw bounding boxes around detected features.
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
# Plot image.
plt.imshow(image)
