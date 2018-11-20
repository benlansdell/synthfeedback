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

class SmallXORDataGenerator(object):
    def __init__(self, config):
        self.config = config
        self.n_trials = config.num_iter_per_epoch
        self.trial_length = config.trial_length
        self.in_dim = config.in_dim

        self.min_onset = 2
        self.mean_onset = 2
        self.max_onset = 2
        self.min_delay = 2
        self.mean_delay = 2
        self.max_delay = 2
        self.separation = 1
        self.var = 3

    def next_batch(self, bs = None):
        batch_size = self.config.batch_size
        X = np.zeros((batch_size, self.in_dim, self.trial_length, self.n_trials))
        Y = np.zeros((batch_size, 1, self.trial_length, self.n_trials))

        #Choose a stim time, and a delay time
        stim_onset1 = np.maximum(np.minimum(rng.randn(batch_size, self.n_trials)*self.var + self.mean_onset, self.max_onset), self.min_onset).astype(int)
        stim_onset2 = stim_onset1 + self.separation
        delay_time = stim_onset2 + np.maximum(np.minimum(rng.randn(batch_size, self.n_trials)*self.var + self.mean_delay, self.max_delay), self.min_delay).astype(int)
        #Choose inputs
        X1 = (rng.random((batch_size, self.n_trials)) < 0.5)*2 - 1
        X2 = (rng.random((batch_size, self.n_trials)) < 0.5)*2 - 1
        for i in range(batch_size):
            for j in range(self.n_trials):
                X[i,0,delay_time[i,j],j] = 1
                X[i,1,stim_onset1[i,j],j] = X1[i,j]
                X[i,1,stim_onset2[i,j],j] = X2[i,j]
                X1p = (X1[i,j] + 1)/2
                X2p = (X2[i,j] + 1)/2
                Yp = 2*(X1p^X2p)-1
                Y[i,0,delay_time[i,j]+1,j] = Yp
        Y = np.array(Y, dtype = float)

        for i in range(self.n_trials):
            x = X[:, :, :, i]
            y = Y[:, :, :, i]
            yield (x, y)

    def test_batch(self, bs = None):
        return self.next_batch()

class XORDataGenerator(object):
    def __init__(self, config):
        self.config = config
        self.n_trials = config.num_iter_per_epoch
        self.trial_length = config.trial_length
        self.in_dim = config.in_dim

        self.min_onset = 10
        self.mean_onset = 15
        self.max_onset = 20
        self.min_delay = 5
        self.mean_delay = 10
        self.max_delay = 15
        self.separation = 3
        self.var = 3

    def next_batch(self):
        batch_size = self.config.batch_size
        X = np.zeros((batch_size, self.in_dim, self.trial_length, self.n_trials))
        Y = np.zeros((batch_size, 1, self.trial_length, self.n_trials))

        #Choose a stim time, and a delay time
        stim_onset1 = np.maximum(np.minimum(rng.randn(batch_size, self.n_trials)*self.var + self.mean_onset, self.max_onset), self.min_onset).astype(int)
        stim_onset2 = stim_onset1 + self.separation
        delay_time = stim_onset2 + np.maximum(np.minimum(rng.randn(batch_size, self.n_trials)*self.var + self.mean_delay, self.max_delay), self.min_delay).astype(int)
        #Choose inputs
        X1 = (rng.random((batch_size, self.n_trials)) < 0.5)*2 - 1
        X2 = (rng.random((batch_size, self.n_trials)) < 0.5)*2 - 1
        for i in range(batch_size):
            for j in range(self.n_trials):
                X[i,0,delay_time[i,j],j] = 1
                X[i,1,stim_onset1[i,j],j] = X1[i,j]
                X[i,1,stim_onset2[i,j],j] = X2[i,j]
                X1p = (X1[i,j] + 1)/2
                X2p = (X2[i,j] + 1)/2
                Yp = 2*(X1p^X2p)-1
                Y[i,0,delay_time[i,j]+1,j] = Yp
        Y = np.array(Y, dtype = float)

        for i in range(self.n_trials):
            x = X[:, :, :, i]
            y = Y[:, :, :, i]
            yield (x, y)

    def test_batch(self):
        return self.next_batch()

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