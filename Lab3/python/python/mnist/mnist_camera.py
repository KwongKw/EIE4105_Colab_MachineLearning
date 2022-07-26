# https://devtalk.nvidia.com/default/topic/1027250/jetson-tx2/how-to-use-usb-webcam-in-jetson-tx2-with-python-and-opencv-/
# To run the program, type
#   python3 mnist-camera.py
# Type 'q' to quit

from __future__ import print_function
from keras.models import load_model
import cv2
import numpy as np

# Load CNN model, assuming that you save your CNN in mnist/models/mnist_cnn.h5
model_file = 'models/mnist_cnn.h5'
print('Loading %s' % model_file)
model = load_model(model_file)

cap = cv2.VideoCapture("/dev/video1")       # video0 is the built-in cam and video1 is the webcam
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()                 # frame is a numpy array with shape (480, 640, 3)

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # frame is a numpy array with shape (480, 640)

    # Increase the contrast to make the captured image to match the training data
    frame = cv2.addWeighted(frame, 2, frame, 0, 10)

    # Show the image on screen
    cv2.imshow('frame', frame)

    # Resize to (1,28,28,1)
    # Put your code here. Search for "OpenCV2 resize python" from the Internet.
    img = ....
    
    # Use black background and scale the image to [0,1] to match the training data
    img = (255-img)/255                             

    # Reshape the image from (28,28) to (1,28,28,1) to fit the input of your CNN
    # Put your code here. Search for "Numpy reshape"
    img = ....

    # Present the image to CNN for classification
    # Put your code here. Search for "Keras model predict_class" from the Internet.
    class_lb = ...
    print(class_lb)

    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break
            
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
