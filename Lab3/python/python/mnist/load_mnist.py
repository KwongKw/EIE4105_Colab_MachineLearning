from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_train_1h = np_utils.to_categorical(y_train)
    y_test_1h = np_utils.to_categorical(y_test)

    # Shuffle the order of the test vectors
    idx = np.arange(0, x_test.shape[0])
    np.random.shuffle(idx)
    x_test = x_test[idx, :]
    y_test = y_test[idx]

    # Return mnist data
    return x_train, y_train, y_train_1h, x_test, y_test, y_test_1h
