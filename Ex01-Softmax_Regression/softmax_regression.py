# coding=utf-8
import numpy as np

from evaluate import predict, cal_accuracy


def softmax_regression(theta, x, y, iters, alpha, test_images, test_labels):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    """
    :param theta: [k,n]
    :param x: [m,n]
    :param y: [k,m]
    :param iters: int
    :param alpha: float
    :return: theta
    """
    loss_list = [] # store the loss of each iteration
    train_acc_list = [] # store the accuracy on train dataset of each iteration
    test_acc_list = [] # store the accuracy on test dataset of each iteration
    for i in range(iters):
        f = np.exp(np.dot(x, theta.T)) # [m,k]
        t = f # store the value of f
        f = np.sum(f * y.T, axis=1) / np.sum(f, axis=1) # [m,]
        f = np.log(f)
        f = - np.sum(f) / x.shape[0] # J(theta): float
        g = np.dot(((t / np.sum(t, axis=1).reshape(-1, 1)) - y.T).T, x) / x.shape[0] # Gradient: [k,n]
        # update theta using gradient descent
        theta = theta - alpha * g

        # log
        print("Iteration: %d, J(theta): %f" % (i + 1, f))
        loss_list.append(f)
        # print("theta after updated: ", theta)
        # print("Gradient: ", g)
        # print()

        # test on train dataset
        y_pred = np.argmax(np.dot(x, theta.T), axis=1)
        acc = (y_pred == np.argmax(y, axis=0)).mean()
        print("Accuracy on train dataset: ", acc)
        train_acc_list.append(acc)

        # test on test dataset
        y_pred = predict(test_images, theta)
        acc = cal_accuracy(y_pred, test_labels)
        print("Accuracy on test dataset: ", acc)
        test_acc_list.append(acc)

        print()
    return theta, loss_list, train_acc_list, test_acc_list

