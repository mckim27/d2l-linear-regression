# -*- coding:utf-8 -*-

from mxnet import autograd, nd
from model import LinearModel
from data import Dataloader

TRUE_W = nd.array([2, -3.4])
TRUE_B = 4.2

num_inputs = 2
num_examples = 1000
batch_size = 10
lr = 0.03  # Learning rate
num_epochs = 10  # Number of iterations

data_loader = Dataloader(TRUE_W, TRUE_B, num_inputs, num_examples, batch_size)
features, labels = data_loader.get_data()

model = LinearModel(num_inputs, lr, batch_size)

for epoch in range(num_epochs):
    # Assuming the number of examples can be divided by the batch size, all
    # the examples in the training data set are used once in one epoch
    # iteration. The features and tags of mini-batch examples are given by X
    # and y respectively

    for X, y in data_loader.data_iter():
        with autograd.record():
            # Minibatch loss in X and y
            l = model.squared_loss(model.linreg(X), y)
        l.backward()  # Compute gradient on l with respect to [w,b]
        model.sgd()
        # sgd([w, b], lr, batch_size)  # Update parameters using their gradient

    train_l = model.squared_loss(model.linreg(features), labels)

    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

model.print_result(TRUE_W, TRUE_B)

