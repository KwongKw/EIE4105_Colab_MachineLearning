#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 23:16:36 2022

@author: enmwmak
"""
from mnist import load_mnist
from classifier import classifier
import numpy as np

# Compute accuracy
def accuracy(y, y_hat):
    return np.mean(y==y_hat)

# Load data
trainpath = '../data/noisy_train_digits.mat'
testpath = '../data/noisy_test_digits.mat'
train_data, train_labels, test_data, test_labels= load_mnist(trainpath, testpath)

# Train a full-cov GMM classifier
M = 8
gmm_cls = classifier(M, model_type='gmm', covariance_type='full', verbose=False)
gmm_cls.fit(train_data, train_labels)

# Predict the labels of the test data
y_test = gmm_cls.predict(test_data)

# Accuracy of GMM classifier on test data
print(accuracy(test_labels, y_test))