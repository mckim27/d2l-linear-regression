# -*- coding:utf-8 -*-

from mxnet import nd
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon


class LinearModel():
    def __init__(self, num_inputs, lr, batch_size):
        self.w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
        print(self.w.shape)
        self.b = nd.zeros(shape=(1,))
        print(self.b.shape)
        self.lr = lr
        self.batch_size = batch_size
        self.w.attach_grad()
        self.b.attach_grad()

    def linreg(self, X):
        return nd.dot(X, self.w) + self.b

    def squared_loss(self, y_hat, y):
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

    def sgd(self):
        self.w[:] = self.w - self.lr * self.w.grad / self.batch_size
        self.b[:] = self.b - self.lr * self.b.grad / self.batch_size

    def print_result(self, true_w, true_b):
        print('Error in estimating w', true_w - self.w.reshape(true_w.shape))
        print('Error in estimating b', true_b - self.b)


class MxLinearModel:
    def __init__(self):
        self.net = nn.Sequential()
        self.net.add(nn.Dense(1))

        # mean 0, sd 0.01 , bias default = 0
        self.net.initialize(init.Normal(sigma=0.01))

        # The squared loss is also known as the L2 norm loss
        self.loss = gloss.L2Loss()

        self.trainer = gluon.Trainer(
            self.net.collect_params(),
            'sgd',
            {
                'learning_rate': 0.03
            }
        )

    def print_result(self, true_w, true_b):
        w = self.net[0].weight.data()
        print('Error in estimating w', true_w.reshape(w.shape) - w)

        b = self.net[0].bias.data()
        print('Error in estimating b', true_b - b)