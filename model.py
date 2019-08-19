# -*- coding:utf-8 -*-

from mxnet import nd


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