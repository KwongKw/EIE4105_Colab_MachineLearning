# Use the saved CNN model to classify a 28x28 JPEG file
# Usage:
#   python3 mnist_console.py images/test/0/1001.jpg
#   python3 mnist_console.py images/test/1/6799.jpg
#   python3 mnist_console.py images/test/2/5921.jpg


from __future__ import print_function
from keras.models import load_model
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load CNN model
model_file = 'models/mnist_cnn.h5'
print('Loading %s' % model_file)
model = load_model(model_file)

# Open the input jpeg file and convert it to numpy array
imgfile = sys.argv[1]
img = cv2.imread(imgfile)
img = img[:,:,0]/255
x = cv2.resize(img,(28,28))

# Pass the query array to the CNN for classification
x = np.reshape(x,[1,28,28,1])
postprob = model.predict(x) 
class_lb = np.argmax(postprob, axis=1)
print('The query digit is \'%s\'' % class_lb)

# Display query image on screen
plt.imshow(img)
plt.show()


