import cv2
import numpy as np


# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier(
    'haar_cascade_files/haarcascade_frontalface_default.xml')

# Check if the cascade file has been loaded correctly
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
    
# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define the scaling factor
scaling_factor = 0.5

# Iterate until the user presses the 'Esc' key
while True:
    # Capture the current frame
   _, frame = cap.read()
   
   # Resize the frame
   frame = cv2.resize(frame, None, 
                      fx = scaling_factor, 
                      fy = scaling_factor,
                      interpolation = cv2.INTER_AREA)
   
   # Convert the image to grayscale
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
   # Run the face detector on the grayscale image
   face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
   
   # Iterate through the detected faces and draw rectangles
   # around them
   for (x, y, w, h) in face_rects:
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

   # Display the output
   cv2.imshow('Face Detector', frame)
   
   # Check if the user pressed the 'Esc' key
   c = cv2.waitKey(1)
   if c == 27:
       break
   
# Release the video capture object
cap.release()

# Close all the windows
cv2.destroyAllWindows()
