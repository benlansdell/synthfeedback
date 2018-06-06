import numpy as np
import numpy.random as rng

from tensorflow.examples.tutorials.mnist import input_data

class DataGenerator(object):
    def __init__(self, config):
        self.config = config
        # load data here
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]

    def test_batch(self, batch_size):
        return next_batch(batch_size)

class LinearDataGenerator(object):
    def __init__(self, config):
        self.config = config
        #Random 30-20-10 network
        n = 10
        m = 20
        p = 30
        #self.T1 = rng.randn(p,m)
        #self.T2 = rng.randn(m,n)

        self.T = rng.rand(p,n)

    def next_batch(self, batch_size):
        #train_x = rng.randn(batch_size, self.T1.shape[0])
        #train_y = np.dot(np.dot(train_x, self.T1), self.T2)

        train_x = rng.randn(batch_size, self.T.shape[0])
        train_y = np.dot(train_x, self.T)

        yield train_x, train_y

    def test_batch(self, batch_size):
        return self.next_batch(batch_size)

class MNISTDataGenerator(object):
    def __init__(self, config):
        self.config = config
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def next_batch(self, batch_size):
        yield self.mnist.train.next_batch(batch_size)

    def test_batch(self, batch_size):
        yield self.mnist.test.next_batch(batch_size)