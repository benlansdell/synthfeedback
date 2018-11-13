#From: https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

import numpy as np
import numpy.random as rand
import tensorflow as tf
#%matplotlib inline
import matplotlib.pyplot as plt
from numpy import random as rng

from utils.utils import tf_matmul_r, tf_matmul_l

# Global config variables
num_steps = 7 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
in_dim = 2
state_size = 50
learning_rate = 1e-4
alpha2 = 1

def gen_data(size=1000000):
    mean_delay = 2
    go_prop = 0.05
    Y = []
    X = np.zeros((in_dim,size))
    X[0,rand.random(size)<go_prop] = 1
    #Add some go cues at random points
    for i in range(size):
        if X[0,i-1] == 1:
            #Choose inputs
            X1 = 1 if rand.random() < 0.5 else -1
            X2 = 1 if rand.random() < 0.5 else -1
            #Choose random delays (harder)
            #d1, d2 = (np.maximum(rand.randn(2)*1 + mean_delay, 0) + 1).astype(int)
            #if d1 == d2:
            #    d1 += 1
            #Choose fixed delays (easier)
            d1, d2 = 2,3
            X[1,i-d1] = X1
            X[1,i-d2] = X2
            X1p = (X1 + 1)/2
            X2p = (X2 + 1)/2
            Yp = 2*(X1p^X2p)-1
            Y.append(Yp)
        else:
            Y.append(0)
    return X, np.array(Y, dtype = float)

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = raw_x.shape[1]

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, in_dim, batch_partition_length], dtype=np.float32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.float32)
    for i in range(batch_size):
        data_x[i,:,:] = raw_x[:,batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, :, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)

"""
Placeholders
"""

x = tf.placeholder(tf.float32, [batch_size, in_dim, num_steps], name='input_placeholder')
y = tf.placeholder(tf.float32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size], dtype=np.float32)

"""
RNN Inputs
"""

# Turn our x placeholder into a list of one-hot tensors:
# rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
#x_one_hot = tf.one_hot(x, num_classes)

rnn_inputs = tf.unstack(x, axis=2)

#rnn_inputs = x

"""
Definition of rnn_cell

This is very similar to the __call__ method on Tensorflow's BasicRNNCell. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py#L95
"""
with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [in_dim + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [in_dim + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
        B = tf.Variable(rng.randn(in_dim + state_size, state_size)*alpha2, name="feedback_weights", dtype=tf.float32)
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)
    #return tf.tanh(tf_matmul_r(tf.concat([rnn_input, state], 1), W, B) + b)

"""
Adding rnn_cells to graph

This is a simplified version of the "static_rnn" function from Tensorflow's api. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn.py#L41
Note: In practice, using "dynamic_rnn" is a better choice that the "static_rnn":
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py#L390
"""
state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]

"""
Predictions, loss, training step

Losses is similar to the "sequence_loss"
function from Tensorflow's API, except that here we are using a list of 2D tensors, instead of a 3D tensor. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/loss.py#L30
"""

#logits and predictions
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, 1])
    b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))
logits = [tf.squeeze(tf.matmul(rnn_output, W) + b) for rnn_output in rnn_outputs]
#logits = [tf.squeeze(tf.matmul_r(rnn_output, W, B) + b) for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# Turn our y placeholder into a list of labels
y_as_list = tf.unstack(y, num=num_steps, axis=1)

#losses and train_step
#losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
#          logit, label in zip(logits, y_as_list)]
#losses = [tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
#          logit, label in zip(logits, y_as_list)]
losses = [tf.reduce_sum(tf.pow(logit-label, 2))/2 for \
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

"""
Train the network
"""

def train_network(num_epochs, num_steps, state_size=state_size, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                                  feed_dict={x:X, y:Y, init_state:training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 250 steps:", training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0

        #Test data
        training_state = np.zeros((batch_size, state_size))
        X_test = np.zeros((batch_size, 2, num_steps))
        Y_test = np.zeros((batch_size, num_steps))
        X_test[:,1,3] = 1
        X_test[:,1,2] = 1
        X_test[:,0,4] = 1
        Y_test[:,5] = -1
        output, loss = sess.run([logits, total_loss], feed_dict={x:X_test, y:Y_test, init_state:training_state})

    return output, training_losses

test_output, training_losses = train_network(30, num_steps)
#plt.plot(training_losses)

#Test trained network
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
