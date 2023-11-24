# coding=utf-8
import numpy as np

from data_process import data_convert
from softmax_regression import softmax_regression


def train(train_images, train_labels, test_images, test_labels, k, iters = 5, alpha = 0.5):
    m, n = train_images.shape
    # data processing
    x, y = data_convert(train_images, train_labels, m, k) # x:[m,n], y:[k,m]
    
    # Initialize theta.  Use a matrix where each column corresponds to a class,
    # and each row is a classifier coefficient for that class.
    theta = np.random.rand(k, n) # [k,n]
    # do the softmax regression
    theta, loss_list, train_acc_list, test_acc_list = softmax_regression(theta, x, y, iters, alpha, test_images, test_labels)
    return theta, loss_list, train_acc_list, test_acc_list

