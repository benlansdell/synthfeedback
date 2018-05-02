import numpy as np

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

class LinearDataGenerator(object):
	def __init__(self, config):
		self.config = config
		# load data here
		self.input = np.ones((500, 784))
		self.y = np.ones((500, 10))

	def next_batch(self, batch_size):
		idx = np.random.choice(500, batch_size)
		yield self.input[idx], self.y[idx]

class MNISTDataGenerator(object):
	def __init__(self, config):
		self.config = config
		self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	def next_batch(self, batch_size):
		yield self.mnist.train.next_batch(batch_size)