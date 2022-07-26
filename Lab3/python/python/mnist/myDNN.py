"""
This file allows user to train and test a DNN by calling two functions:
train_dnn() and test_dnn()

Author: M.W. Mak
Date: March 2022
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.layers import BatchNormalization
import numpy as np

def train_dnn(x_train, y_train_ohe, x_val, y_val_ohe, n_hiddens=[100,100],
              optimizer='adam', act='relu', n_epochs=10, bat_size=100, verbose=1):
    """
    :param x_train: N x D ndarray of training data, where D is the feature dimension
    :param y_train_ohe: N x nClass ndarray of target labels in one-hot format
    :param x_val: M x D ndarray of validation data
    :param y_val_ohe: M x nClass ndarray of target labels
    :param n_hiddens: Array containing no. of hidden nodes for each hidden layer
    :param act: Activation function
    :param n_epochs: number of epochs
    :param bat_size: mini-batch size
    :param return: keras.models.Sequential
    """

    # Create a DNN
    model = Sequential()

    # Define number of hidden layers and number of nodes in each layer according to n_hiddens
    model.add(Dense(n_hiddens[0], input_dim=x_train.shape[1], activation=act))
    for i in range(1, len(n_hiddens)):
        model.add(Dense(n_hiddens[i]))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dropout(0.2))
    model.add(Dense(y_train_ohe.shape[1], activation='softmax'))

    # Define loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Perform training
    model.fit(x_train, y_train_ohe, epochs=n_epochs, batch_size=bat_size, verbose=verbose,
              validation_data=(x_val, y_val_ohe))
    return model


def test_dnn(x_test, y_test, model):
    """
    :param x_test: N x D ndarray of test data, where D is the feature dimension
    :param y_test: N x nClass ndarray of target labels
    :param model: DNN model created by train_dnn()
    :return: test accuracy, no. of correct, and no. of samples in x_test
    """
    postprob = model.predict(x_test) 
    y_pred = np.argmax(postprob, axis=1)
    n_correct = np.sum(y_test == y_pred, axis=0)
    n_samples = x_test.shape[0]
    test_acc = n_correct / n_samples
    return test_acc, n_correct, n_samples