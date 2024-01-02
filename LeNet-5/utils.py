"""
Utils for LeNet-5
"""

import numpy as np


def im2col(input_data, filter_h, filter_w, stride, padding):
    """
    Convert image to column
    :param input_data: [N, C, H, W]
    :param filter_h:
    :param filter_w:
    :param stride:
    :param padding:
    :return: [N*out_h*out_w, C*filter_h*filter_w]
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*padding - filter_h) // stride + 1
    out_w = (W + 2*padding - filter_w) // stride + 1

    img = np.pad(input_data, [(0,0), (0, 0), (padding, padding), (padding, padding)], 'constant')
    col = np.zeros((N, C, out_h, out_w, filter_h, filter_w))

    for y in range(out_h):
        y_begin = y*stride
        for x in range(out_w):
            x_begin = x*stride
            col[:, :, y, x, :, :] = img[:, :, y_begin:(y_begin+filter_h), x_begin:(x_begin+filter_w)]

    col = col.transpose(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w, C*filter_h*filter_w)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride, padding):
    """
    Convert column to image
    :param col: [N*out_h*out_w, C*filter_h*filter_w]
    :param input_shape: [N, C, H, W]
    :param filter_h:
    :param filter_w:
    :param stride:
    :param padding:
    :return: [N, C, H, W]
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    # col: [N, C, out_h, out_w, filter_h, filter_w]
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 1, 2, 4, 5)

    img = np.zeros((N, C, H + 2*padding, W + 2*padding))
    for y in range(out_h):
        y_begin = y*stride
        for x in range(out_w):
            x_begin = x*stride
            img[:, :, y_begin:(y_begin+filter_h), x_begin:(x_begin+filter_w)] = col[:, :, y, x, :, :]

    return img[:, :, padding:(H+padding), padding:(W+padding)]


def dataset_shuffle(imgs, labels):
    """
    Shuffle Dataset
    :param imgs:
    :param labels:
    :return:
    """
    permutation = np.random.permutation(imgs.shape[0])
    imgs = imgs[permutation]
    labels = labels[permutation]
    return imgs, labels


if __name__ == '__main__':
    pass
