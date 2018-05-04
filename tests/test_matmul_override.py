import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from utils.utils import tf_matmul_r, tf_matmul_l
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import numpy.random as rng

n_tests = 10

tol = 1e-8

#Load some MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def next_batch(batch_size):
    yield mnist.train.next_batch(batch_size)

#Set up variables
batch_size = 32
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# set initial feedforward and feedback weights
m = 512
n = 10
p = 784

#Scale weight initialization
alpha0 = np.sqrt(2.0/p)
alpha1 = np.sqrt(2.0/m)
alpha2 = 1

#Plus one for bias terms
A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
W = tf.Variable(rng.randn(m+1,n)*alpha1, name="output_weights", dtype=tf.float32)
B = tf.Variable(rng.randn(m+1,n)*alpha2, name="feedback_weights", dtype=tf.float32)

e0 = tf.ones([batch_size, 1], tf.float32)
e1 = tf.ones([batch_size, 1], tf.float32)
x_aug = tf.concat([x, e0], 1)

# network architecture with ones added for bias terms
#FA, computed manually
h_man = tf.sigmoid(tf.matmul(x_aug, A))
h_aug_man = tf.concat([h_man, e1], 1)
y_p_man = tf.matmul(h_aug_man, W)

loss_man = tf.reduce_sum(tf.pow(y_p_man-y, 2))/2
grad_W_man = tf.gradients(xs=W, ys=loss_man)[0]
e_man = (y_p_man - y)
h_prime_man = h_man*(1-h_man)
grad_A_manual = tf.matmul(tf.transpose(x_aug), tf.multiply(h_prime_man, tf.matmul(e_man, tf.transpose(B[0:m,:]))))

#FA, computed automatically
x_aug = tf.concat([x, e0], 1)
h = tf.sigmoid(tf.matmul(x_aug, A))
h_aug = tf.concat([h, e1], 1)
#The key line! Replace W with B in any backprop step
y_p = tf_matmul_r(h_aug, W, B)
#y_p = tf.matmul(h_aug, W)

loss = tf.reduce_sum(tf.pow(y_p-y, 2))/2
grad_W_auto = tf.gradients(xs=W, ys = loss)[0]
grad_A_auto = tf.gradients(xs=A, ys = loss)[0]

norms_W = np.zeros(n_tests)
norms_A = np.zeros(n_tests)

#Compare to overridden matmul functions
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(n_tests):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		feed_dict = {x: batch_x, y: batch_y}
		[g_W_auto, g_W_man] = sess.run([grad_W_auto, grad_W_man], feed_dict=feed_dict)
		[g_A_auto, g_A_man] = sess.run([grad_A_auto, grad_A_manual], feed_dict=feed_dict)
		norms_W[i] = np.linalg.norm(g_W_auto - g_W_man, 'fro')/g_W_auto.shape[0]/g_W_auto.shape[1]
		norms_A[i] = np.linalg.norm(g_A_auto - g_A_man, 'fro')/g_A_auto.shape[0]/g_A_auto.shape[1]

#print norms_W
#print norms_A

assert (norms_W < tol).all(), "Backprop'ed gradients not within tolerance"
assert (norms_A < tol).all(), "Backprop'ed gradients not within tolerance"

print "Passes test to within %e."%tol