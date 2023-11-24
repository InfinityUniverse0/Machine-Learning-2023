# coding=utf-8
import numpy as np


def predict(test_images, theta):
    # # 二值化操作
    # test_images[test_images<=40]=0
    # test_images[test_images>40]=1
    scores = np.dot(test_images, theta.T)
    preds = np.argmax(scores, axis=1)
    return preds


def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    y_pred = y_pred.reshape(-1)
    y = y.reshape(-1)
    acc = (y_pred == y).mean()
    return acc
