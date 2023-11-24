# coding=utf-8

from data_process import load_mnist, load_data
from train import train
from evaluate import predict, cal_accuracy
from visualize import display_loss, display_accuracy
    

if __name__ == '__main__':
    # initialize the parameters needed
    mnist_dir = "./mnist_data/"
    train_data_dir = "train-images-idx3-ubyte"
    train_label_dir = "train-labels-idx1-ubyte"
    test_data_dir = "t10k-images-idx3-ubyte"
    test_label_dir = "t10k-labels-idx1-ubyte"
    k = 10
    iters = 1000
    alpha = 0.5

    # get the data
    train_images, train_labels, test_images, test_labels = load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir)
    print("Got data. ")

    # train the classifier
    theta, loss_list, train_acc_list, test_acc_list = train(train_images, train_labels, test_images, test_labels, k, iters, alpha)
    print("Finished training. ")

    # evaluate on the testset
    y_predict = predict(test_images, theta)
    accuracy = cal_accuracy(y_predict, test_labels)
    print("Finished test. ")

    print("The accuracy on test dataset is: ", accuracy)
    
    # show training loss
    display_loss(loss_list)
    # show training accuracy
    display_accuracy(train_acc_list, test_acc_list)
