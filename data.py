# -*- coding:utf-8 -*-

from mxnet import ndarray as nd
from IPython import display
from matplotlib import pyplot as plt
import random


class Dataloader:

    def __init__(self, true_w, true_b, num_inputs:int, num_examples:int, batch_size:int):
        self.features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
        self.labels = nd.dot(self.features, true_w) + true_b
        self.labels += nd.random.normal(scale=0.01, shape=self.labels.shape)
        self.batch_size = batch_size

    def get_data(self):
        return self.features, self.labels

    def data_iter(self):
        num_examples = len(self.features)
        indices = list(range(num_examples))

        # The examples are read at random, in no particular order
        random.shuffle(indices)

        for i in range(0, num_examples, self.batch_size):
            j = nd.array(indices[i: min(i + self.batch_size, num_examples)])
            yield self.features.take(j), self.labels.take(j)
            # The “take” function will then return the corresponding element based # on the indices

    def __use_svg_display(self):
        # Display in vector graphics
        display.set_matplotlib_formats('svg')

    def __set_figsize(self, figsize=(3.5, 2.5)):
        self.__use_svg_display()
        # Set the size of the graph to be plotted
        plt.rcParams['figure.figsize'] = figsize

    def display_fig(self):
        self.__set_figsize()
        plt.figure(figsize=(10, 6))
        plt.scatter(self.features[:, 1].asnumpy(), self.labels.asnumpy(), 1)
