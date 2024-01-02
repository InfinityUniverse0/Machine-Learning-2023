"""
Layers for LeNet-5
"""

import numpy as np
from utils import im2col, col2im
from functions import *


WEIGHT_INIT_STD = 0.01


class ReLU:
    """
    ReLU Activation Layer
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)

        y = x.copy()
        y[self.mask] = 0

        return y

    def backward(self, dout):
        dx = dout
        dx[self.mask] = 0
        return dx


class Convolution:
    """
    Convolution Layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """
        Convolution Layer
        :param in_channels: int
        :param out_channels: int
        :param kernel_size: int or tuple(h:int, w:int)
        :param stride: int = 1
        :param padding: int = 0
        :param bias: bool = True
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2  # kernel_size Assertion
        self.kernel_size = kernel_size

        self.W = np.random.randn(out_channels, in_channels, *kernel_size) * WEIGHT_INIT_STD
        if bias:
            self.b = np.random.randn(out_channels) * WEIGHT_INIT_STD
        else:
            self.b = None

        # Variables Init
        self.x = None
        self.col = None
        self.col_W = None

        # Gradients Init
        self.dW = None
        self.db = None

        # self.params = []
        # for i in range(out_channels):
        #     param = {'W': np.random.randn(in_channels, *kernel_size)}
        #     if bias:
        #         param['b'] = np.random.randn(1)
        #     self.params.append(param)

    def forward(self, x):
        """
        Forward Propagation
        :param x: [N, C, H, W]
        :return y: [N, out_channels, out_h, out_w]
        """
        N, C, H, W = x.shape
        filter_h, filter_w = self.kernel_size[0], self.kernel_size[1]
        out_h = (H + 2*self.padding - filter_h) // self.stride + 1
        out_w = (W + 2*self.padding - filter_w) // self.stride + 1

        # col: [N*out_h*out_w, C*filter_h*filter_w]
        col = im2col(x, filter_h, filter_w, stride=self.stride, padding=self.padding)
        # col_W: [in_channels*filter_h*filter_w, out_channels]
        col_W = self.W.reshape(self.out_channels, -1).T

        y = np.matmul(col, col_W)
        if self.bias:
            y += self.b

        y = y.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        # Store Variables
        self.x = x
        self.col = col
        self.col_W = col_W

        return y

    def backward(self, dout):
        """
        Backward Propagation
        :param dout: [N, out_channels, out_h, out_w]
        :return dx: [N, C, H, W]
        """
        filter_h, filter_w = self.kernel_size[0], self.kernel_size[1]

        # dout: [N*out_h*out_w, out_channels]
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # db: [out_channels]
        if self.bias:
            db = np.sum(dout, axis=0)
        else:
            db = None

        # dW: [in_channels*filter_h*filter_w, out_channels]
        dW = np.matmul(self.col.T, dout)
        # dW: [out_channels, in_channels, filter_h, filter_w]
        dW = dW.transpose(1, 0).reshape(self.out_channels, self.in_channels, filter_h, filter_w)

        # dcol: [N*out_h*out_w, in_channels*filter_h*filter_w]
        dcol = np.matmul(dout, self.col_W.T)

        # dx: [N, C, H, W]
        dx = col2im(dcol, self.x.shape, filter_h, filter_w, stride=self.stride, padding=self.padding)

        # Store Gradients
        self.dW = dW
        self.db = db

        return dx


class MaxPooling:
    """
    Pooling Layer
    - Max Pooling
    """
    def __init__(self, kernel_size, stride, padding=0):
        """
        Max Pooling Layer
        :param kernel_size: int or tuple(h:int, w:int)
        :param stride: int
        :param padding: int = 0
        """
        self.stride = stride
        self.padding = padding

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2  # kernel_size Assertion
        self.kernel_size = kernel_size

        # Variables Init
        self.x = None
        self.arg_max = None

    def forward(self, x):
        """
        Forward Propagation
        :param x: [N, C, H, W]
        :return y: [N, C, out_h, out_w]
        """
        N, C, H, W = x.shape
        pool_h, pool_w = self.kernel_size[0], self.kernel_size[1]
        out_h = (H + 2*self.padding - pool_h) // self.stride + 1
        out_w = (W + 2*self.padding - pool_w) // self.stride + 1

        # col: [N*out_h*out_w, C*pool_h*pool_w]
        col = im2col(x, pool_h, pool_w, stride=self.stride, padding=self.padding)
        # col: [N*out_h*out_w*C, pool_h*pool_w]
        col = col.reshape(-1, pool_h*pool_w)

        arg_max = np.argmax(col, axis=1)
        y = np.max(col, axis=1)
        y = y.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        # Store Variables
        self.x = x
        self.arg_max = arg_max

        return y

    def backward(self, dout):
        """
        Backward Propagation
        :param dout: [N, C, out_h, out_w]
        :return dx: [N, C, H, W]
        """
        N, C, out_h, out_w = dout.shape
        pool_h, pool_w = self.kernel_size[0], self.kernel_size[1]

        # dout: [N*out_h*out_w*C]
        dout = dout.transpose(0, 2, 3, 1).reshape(-1)

        dx = np.zeros(N*out_h*out_w*C, pool_h*pool_w)
        dx[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout

        # dx: [N*out_h*out_w, C*pool_h*pool_w]
        dx = dx.reshape(N*out_h*out_w, C*pool_h*pool_w)
        # dx: [N, C, H, W]
        dx = col2im(dx, self.x.shape, pool_h, pool_w, stride=self.stride, padding=self.padding)

        return dx


class Linear:
    """
    Linear Layer(Fully Connected Layer)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Linear Layer(Fully Connected Layer)
        :param in_features: int
        :param out_features: int
        :param bias: bool = True
        """
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.W = np.random.randn(in_features, out_features) * WEIGHT_INIT_STD
        if bias:
            self.b = np.random.randn(out_features) * WEIGHT_INIT_STD
        else:
            self.b = None

        # Variables Init
        self.x = None
        self.input_shape = None

        # Gradients Init
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Forward Propagation
        :param x: Tensor or Matrix
        :return y: [N, out_features]
        """
        self.input_shape = x.shape
        x = x.reshape(x.shape[0], -1)  # x: [N, in_features]
        self.x = x

        y = np.matmul(x, self.W)  # y: [N, out_features]
        if self.bias:
            y += self.b

        return y

    def backward(self, dout):
        """
        Backward Propagation
        :param dout: [N, out_features]
        :return dx: [self.input_shape]
        """
        if self.bias:
            self.db = np.sum(dout, axis=0)
        else:
            self.db = None

        self.dW = np.matmul(self.x.T, dout)
        dx = np.matmul(dout, self.W.T)  # dx: [N, in_features]
        dx = dx.reshape(self.input_shape)

        return dx


class SoftmaxWithLoss:
    """
    Softmax With Loss
    """
    def __init__(self):
        self.loss = None
        self.y = None  # Output of softmax
        self.t = None  # Ground-Truth Label

    def forward(self, x, t):
        """
        Forward Propagation
        :param x: Output of model
        :param t: Ground-Truth Label
        :return loss: scalar
        """
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_loss(self.y, t)
        return self.loss

    def backward(self, dout=1):
        """
        Backward Propagation
        :param dout: = 1
        :return dx: [N, num_classes]
        """
        batch_size = self.y.shape[0]

        if self.t.size == self.y.size:  # one-hot encoding
            dx = self.y - self.t
        else:  # NON one-hot encoding
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t.flatten()] -= 1

        return dx


if __name__ == '__main__':
    pass
