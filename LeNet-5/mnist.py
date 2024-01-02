"""
MNIST Dataset Loader.
"""

import numpy as np
import os
from PIL import Image

# MNIST Dataset Path
MNIST_PATH = './MNIST'
TRAIN_IMAGE_PATH = os.path.join(MNIST_PATH, 'train-images-idx3-ubyte')
TRAIN_LABEL_PATH = os.path.join(MNIST_PATH, 'train-labels-idx1-ubyte')
TEST_IMAGE_PATH = os.path.join(MNIST_PATH, 't10k-images-idx3-ubyte')
TEST_LABEL_PATH = os.path.join(MNIST_PATH, 't10k-labels-idx1-ubyte')

# MNIST Dataset Parameters
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNEL = 1
NUM_CLASSES = 10


def load_dataset(file_path, is_img=True):
    """
    :param file_path:
    :param is_img: image or label
    :return data:
    For image, data: [n, C, H, W]
    For label, data: [n, 1]
    """
    if is_img:
        offset = 16  # 偏移量为16
        data_size = (IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH)
    else:
        offset = 8  # 偏移量为8
        data_size = (1, )
    with open(file_path, 'rb') as file:
        data = np.frombuffer(file.read(), np.uint8, offset=offset)
    data = data.reshape(-1, *data_size)
    return data


def one_hot_label_convert(labels, num_classes=NUM_CLASSES):
    """
    :param labels: [n, 1]
    :param num_classes: 10 for MNIST
    :return: [n, num_classes]
    """
    n = labels.shape[0]
    one_hot_labels = np.zeros((n, num_classes), dtype=np.uint8)
    # for i in range(n):
    #     one_hot_labels[i, labels[i]] = 1
    one_hot_labels[np.arange(n), labels.flatten()] = 1
    return one_hot_labels


def display_img(images, idx):
    """
    Display Image
    :param images: [n, C, H, W]
    :param idx:
    :return:
    """
    image = Image.fromarray(images[idx][0])
    image.show()


def load_mnist(normalize=False, one_hot_label=True):
    train_imgs = load_dataset(TRAIN_IMAGE_PATH, True)
    train_labels = load_dataset(TRAIN_LABEL_PATH, False)
    test_imgs = load_dataset(TEST_IMAGE_PATH, True)
    test_labels = load_dataset(TEST_LABEL_PATH, False)

    if normalize:
        train_imgs = train_imgs.astype(np.float32)
        train_imgs /= 255.0
        test_imgs = test_imgs.astype(np.float32)
        test_imgs /= 255.0

    if one_hot_label:
        train_labels = one_hot_label_convert(train_labels)
        test_labels = one_hot_label_convert(test_labels)

    return train_imgs, train_labels, test_imgs, test_labels


if __name__ == '__main__':
    pass
