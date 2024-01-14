"""
Optimizer
"""

from layers import *


class SGD:
    """
    Stochastic Gradient Descent
    """
    def __init__(self, model_layers, lr=0.01):
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
            if is_weighted_layer(layer):
                layer.W -= self.lr * layer.dW
                if layer.bias:
                    layer.b -= self.lr * layer.db


class Adam:
    """
    Adam
    """
    def __init__(self, model_layers, lr=0.001, beta1=0.9, beta2=0.999):
        """
        Adam
        :param model_layers:
        :param lr: learning rate
        :param beta1:
        :param beta2:
        """
        self.model_layers = model_layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = {}
        self.v = {}

        for layer_name, layer in self.model_layers.items():
            if is_weighted_layer(layer):
                self.m[layer_name] = {}
                self.v[layer_name] = {}
                self.m[layer_name]['W'] = np.zeros_like(layer.W)
                self.v[layer_name]['W'] = np.zeros_like(layer.W)
                if layer.bias:
                    self.m[layer_name]['b'] = np.zeros_like(layer.b)
                    self.v[layer_name]['b'] = np.zeros_like(layer.b)

    def step(self):
        """
        Update Model Parameters
        :return:
        """
        self.t += 1
        for layer_name, layer in self.model_layers.items():
            if is_weighted_layer(layer):
                self.m[layer_name]['W'] = self.beta1 * self.m[layer_name]['W'] + (1 - self.beta1) * layer.dW
                self.v[layer_name]['W'] = self.beta2 * self.v[layer_name]['W'] + (1 - self.beta2) * (layer.dW ** 2)
                m_w = self.m[layer_name]['W'] / (1 - self.beta1 ** self.t)
                v_w = self.v[layer_name]['W'] / (1 - self.beta2 ** self.t)
                layer.W -= self.lr * m_w / (np.sqrt(v_w) + 1e-7)
                if layer.bias:
                    self.m[layer_name]['b'] = self.beta1 * self.m[layer_name]['b'] + (1 - self.beta1) * layer.db
                    self.v[layer_name]['b'] = self.beta2 * self.v[layer_name]['b'] + (1 - self.beta2) * (layer.db ** 2)
                    m_b = self.m[layer_name]['b'] / (1 - self.beta1 ** self.t)
                    v_b = self.v[layer_name]['b'] / (1 - self.beta2 ** self.t)
                    layer.b -= self.lr * m_b / (np.sqrt(v_b) + 1e-7)
