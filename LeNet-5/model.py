"""
LeNet-5 Model
"""

from layers import *
from collections import OrderedDict


class LeNet5:
    """
    LeNet-5
    """
    def __init__(self, img_channels=1, num_classes=10):
        """
        LeNet-5
        :param img_channels: int = 1
        :param num_classes: int = 10
        """
        self.layers = OrderedDict()

        self.layers['conv1'] = Convolution(img_channels, 6, kernel_size=5)
        self.layers['relu1'] = ReLU()
        self.layers['pool1'] = MaxPooling(kernel_size=2, stride=2)

        self.layers['conv2'] = Convolution(6, 16, kernel_size=5)
        self.layers['relu2'] = ReLU()
        self.layers['pool2'] = MaxPooling(kernel_size=2, stride=2)

        self.layers['fc1'] = Linear(16*4*4, 120)
        self.layers['relu3'] = ReLU()

        self.layers['fc2'] = Linear(120, 84)
        self.layers['relu4'] = ReLU()

        self.layers['fc3'] = Linear(84, num_classes)

        # 输出层 SoftmaxWithLoss 仅用于训练阶段，推理阶段不需要
        self.output_layer = SoftmaxWithLoss()

    def predict(self, x):
        """
        Predict (Inference Stage)
        :param x: [N, C, H, W] for batch case; or [C, H, W]
        :return : [N, num_classes] for batch case; or [num_classes]
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        Calculate Loss
        :param x: [N, C, H, W] for batch case; or [C, H, W]
        :param t: Ground-Truth Labels
        :return loss: scalar
        """
        x = self.predict(x)
        loss = self.output_layer.forward(x, t)
        return loss

    def backward(self):
        """
        Backward Propagation to Calculate Gradients
        Note: Must Be Called After loss()
        :return:
        """
        dout = self.output_layer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

    def gradient(self, x, t):
        """
        Forward and Backward
        :param x: input
        :param t: labels
        :return:
        """
        self.loss(x, t)
        self.backward()

    def get_accuracy(self, x, t):
        """
        Get Accuracy
        :param x: input
        :param t: labels
        :return acc:
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1 and t.shape[1] != 1:  # one-hot
            t = np.argmax(t, axis=1)

        t = t.reshape(-1)

        assert y.ndim == 1 and t.ndim == 1

        acc = np.sum(y == t) / y.size
        return acc
