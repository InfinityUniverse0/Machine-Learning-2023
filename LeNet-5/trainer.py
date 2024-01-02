"""
Model Trainer
"""

from optim import *
from utils import dataset_shuffle
from math import ceil
import pickle


class Trainer:
    """
    Model Trainer
    """
    def __init__(
            self, model, x_train, t_train, x_test, t_test,
            optimizer: str, optimizer_params: dict,
            epoch: int, batch_size: int = 1,
            shuffle: bool = True
    ):
        """
        Model Trainer
        :param model:
        :param x_train:
        :param t_train:
        :param x_test:
        :param t_test:
        :param optimizer: str
        :param optimizer_params: dict
        :param epoch: int
        :param batch_size: int = 1
        :param shuffle: bool = True
        """
        self.model = model
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epoch = epoch
        self.batch_size = batch_size
        self.shuffle = shuffle

        optimizer_dict = {
            'sgd': SGD,
        }
        optimizer_params['model_layers'] = model.layers
        self.optimizer = optimizer_dict[optimizer.lower()](**optimizer_params)

        # Iteration Info
        self.iter_per_epoch = ceil(x_train.shape[0] / batch_size)

        # Evaluation
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_per_epoch(self, cur_epoch):
        if self.shuffle:
            x_train, t_train = dataset_shuffle(self.x_train, self.t_train)
        else:
            x_train, t_train = self.x_train.copy(), self.t_train.copy()

        idx = -self.batch_size
        for it in range(self.iter_per_epoch):
            idx += self.batch_size
            if it == self.iter_per_epoch - 1:
                x = x_train[idx:]
                t = t_train[idx:]
            else:
                x = x_train[idx:(idx+self.batch_size)]
                t = t_train[idx:(idx+self.batch_size)]

            # Calculate Gradients
            loss = self.model.loss(x, t)  # forward
            self.model.backward()  # backward

            # Update Model Parameters
            self.optimizer.step()

            # Save Evaluation
            self.train_loss_list.append(loss)

            # log
            print('Epoch: {} / {}  Iter: {} / {}  Train Loss: {}'.format(
                cur_epoch, self.epoch, (it + 1), self.iter_per_epoch, loss
            ))

        # Test
        train_acc = self.model.get_accuracy(self.x_train, self.t_train)
        test_acc = self.model.get_accuracy(self.x_test, self.t_test)
        self.train_acc_list.append(train_acc)
        self.test_acc_list.append(test_acc)

        # log
        print('Epoch: {} / {}  Train Acc: {}  Test Acc: {}'.format(
            cur_epoch, self.epoch, train_acc, test_acc
        ))

    def train(self, dump_eval: bool = True):
        for i in range(self.epoch):
            self.train_per_epoch(i + 1)

        if dump_eval:
            with open('train_loss_list.pkl', 'wb') as f:
                pickle.dump(self.train_loss_list, f)
            with open('train_acc_list.pkl', 'wb') as f:
                pickle.dump(self.train_acc_list, f)
            with open('test_acc_list.pkl', 'wb') as f:
                pickle.dump(self.test_acc_list, f)
            print('Dump Successfully!')
