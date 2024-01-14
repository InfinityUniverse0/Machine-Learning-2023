"""
Functions for LeNet-5
"""

import numpy as np


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    """softmax实现时，为防止溢出，减去最大值"""
    if x.ndim == 2:  # [N, num_classes]
        x_max = np.max(x, axis=1, keepdims=True)  # x_max: [N, 1]
        y = np.exp(x - x_max)  # in case of overflow
        y = y / np.sum(y, axis=1, keepdims=True)
        return y
    else:  # [num_classes]
        x_max = np.max(x)
        y = np.exp(x - x_max)  # in case of overflow
        y = y / np.sum(y)
        return y


def cross_entropy_loss(y, t, epsilon=1e-7):
    """
    Cross Entropy Loss
    :param y: output
    :param t: ground truth labels
    :param epsilon: = 1e-7  防止log的真数为0
    :return:
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)

    if t.shape[1] == y.shape[1]:  # one-hot encoding
        # Convert to NON one-hot encoding(True Label)
        t = np.argmax(t, axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + epsilon)) / batch_size

    # if t.shape[1] == 1:  # NOT one-hot encoding
    #     # Convert to ONE-HOT encoding
    #     temp = np.zeros(y.shape)
    #     temp[np.arange(batch_size), t.flatten()] = 1
    #     t = temp
    #
    # return -np.sum(t * np.log(y + epsilon)) / batch_size
