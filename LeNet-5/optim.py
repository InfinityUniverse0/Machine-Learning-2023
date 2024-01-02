"""
Optimizer
"""

from layers import *


class SGD:
    """
    Stochastic Gradient Descent
    """
    def __init__(self, model_layers, lr):
        """
        Stochastic Gradient Descent
        :param model_layers:
        :param lr: learning rate
        """
        self.model_layers = model_layers
        self.lr = lr

    def step(self):
        """
        Update Model Parameters
        :return:
        """
        for layer in self.model_layers.values():
            if isinstance(layer, (Convolution, Linear)):
                layer.W -= self.lr * layer.dW
                if layer.b is not None:
                    layer.b -= self.lr * layer.db
