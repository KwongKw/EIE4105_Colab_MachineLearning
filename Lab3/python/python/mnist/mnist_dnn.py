# DNN for MNIST handwritten digit classification
# Require TensorFlow 2.0 or above
# Usage:
#   python3 mnist_dnn.py

from __future__ import print_function
from myDNN import train_dnn, test_dnn
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

batch_size = 128
num_classes = 10
epochs = 10
n_hiddens = [200, 200, 200]

# input image dimensions
img_rows, img_cols = 28, 28

# Load and normalize MNIST digits
(x_train, trn_lbs), (x_test, tst_lbs) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices (one-hot)
y_train = tf.keras.utils.to_categorical(trn_lbs, num_classes)
y_test = tf.keras.utils.to_categorical(tst_lbs, num_classes)

# Train a DNN
dnn = train_dnn(x_train, y_train, x_test, y_test, n_hiddens=n_hiddens,
                optimizer='adam', act='relu', n_epochs=epochs, bat_size=batch_size, verbose=1)

# Test the DNN
train_acc, _, _ = test_dnn(x_train, trn_lbs, dnn)
test_acc, _, _ = test_dnn(x_test, tst_lbs, dnn)
print('Train accuracy: %.2f%% ' % (train_acc * 100))
print('Test accuracy: %.2f%% ' % (test_acc * 100))