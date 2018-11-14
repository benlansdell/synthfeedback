#Based on: https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
import numpy as np
import numpy.random as rand
import tensorflow as tf
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
activation = tf.tanh
act_prime = lambda x: 1 - tf.multiply(x,x)

#Node pert params
lmbda = 1e-3
var_xi = 0.1
p_fire = 1 #1 in 20 times a neuron will add Gaussian noise to its output

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
init_gradW = tf.zeros([state_size+1, state_size], dtype=np.float32)
init_gradB = tf.zeros([state_size, state_size], dtype=np.float32)
init_gradU = tf.zeros([in_dim, state_size], dtype=np.float32)

"""
RNN Inputs
"""

#x_one_hot = tf.one_hot(x, num_classes)
rnn_inputs = tf.unstack(x, axis=2)

"""
Definition of rnn_cell
"""
ones0 = tf.ones([batch_size, 1], tf.float32)
U = tf.get_variable('U', [in_dim, state_size])
W = tf.get_variable('W', [state_size+1, state_size])
#B = tf.Variable(rng.randn(state_size+1, state_size)*alpha2, name="feedback_weights", dtype=tf.float32)
B = tf.Variable(rng.randn(state_size, state_size)*alpha2, name="feedback_weights", dtype=tf.float32)

#Add perturbed output. 
#
#Choose a random subset to randomly add noise
#
#Update the estimate for B on the basis of this
#
#

def rnn_cell(rnn_input, state, W, U):
    ones0 = tf.ones([batch_size, 1], tf.float32)
    state_p = tf.concat([state, ones0], 1)
    return activation(tf.matmul(rnn_input, U) + tf.matmul(state_p, W))

"""
Adding rnn_cells to graph
"""
state = init_state
state_p = init_state
rnn_outputs = []
rnn_pert_outputs = []
noise_outputs = []
for rnn_input in rnn_inputs:
    #Add noise
    mask = tf.random_uniform(state.shape) < p_fire
    xi = tf.multiply(tf.random_normal(state.shape)*var_xi, tf.to_float(mask))
    state = rnn_cell(rnn_input, state, W, U)
    state_p = rnn_cell(rnn_input, state_p, W, U) + xi
    rnn_outputs.append(state)
    rnn_pert_outputs.append(state_p)
    noise_outputs.append(xi)
final_state = rnn_outputs[-1]

"""
Predictions, loss, training step
"""
V = tf.get_variable('V', [state_size+1, 1])
logits = [tf.squeeze(tf.matmul(tf.concat([rnn_output, ones0], 1), V)) for rnn_output in rnn_outputs]
logits_as_t = tf.stack(logits, axis=1)
y_as_list = tf.unstack(y, num=num_steps, axis=1)

#losses and train_step
loss = [tf.pow(logit-label, 2)/2 for logit, label in zip(logits, y_as_list)]
losses = [tf.reduce_sum(tf.pow(logit-label, 2))/2 for logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)

#Perturbed outputs and loss
logits_pert = [tf.squeeze(tf.matmul(tf.concat([rnn_pert_output, ones0], 1), V)) for rnn_pert_output in rnn_pert_outputs]
logits_pert_as_t = tf.stack(logits_pert, axis=1)

loss_pert = [tf.pow(logit-label, 2)/2 for logit, label in zip(logits_pert, y_as_list)]
losses_pert = [tf.reduce_sum(tf.pow(logit-label, 2))/2 for logit, label in zip(logits_pert, y_as_list)]
total_loss_pert = tf.reduce_mean(losses_pert)

e0s = [(logit - label) for logit, label in zip(logits, y_as_list)]
delta0s = e0s

##############
## BACKPROP ##
##############

#grad_U = init_gradU
#grad_W = init_gradW
#for i in range(num_steps):
#    for j in range(i+1)[::-1]:
#        if j == i:
#            delta = tf.multiply(tf.matmul(delta0s[i][:,None],tf.transpose(V[0:state_size,:])), act_prime(rnn_outputs[j]))
#        else:
#            delta = tf.multiply(tf.matmul(delta, tf.transpose(W[0:state_size,:])), act_prime(rnn_outputs[j]))
#        grad_U = grad_U + tf.matmul(tf.transpose(rnn_inputs[j]), delta)
#        if j > 0:
#            grad_W = grad_W + tf.matmul(tf.transpose(tf.concat([rnn_outputs[j-1], ones0],1)), delta)
#
#grad_V = tf.gradients(xs=V, ys=total_loss)[0]
#grad_W = tf.gradients(xs=W, ys=total_loss)[0]
#grad_U = tf.gradients(xs=U, ys=total_loss)[0]

########################
## FEEDBACK ALIGNMENT ##
########################

#grad_U = init_gradU
#grad_W = init_gradW
#for i in range(num_steps):
#    for j in range(i+1)[::-1]:
#        if j == i:
#            delta = tf.multiply(tf.matmul(delta0s[i][:,None],tf.transpose(V[0:state_size,:])), act_prime(rnn_outputs[j]))
#        else:
#            delta = tf.multiply(tf.matmul(delta, tf.transpose(B)), act_prime(rnn_outputs[j]))
#        grad_U = grad_U + tf.matmul(tf.transpose(rnn_inputs[j]), delta)
#        if j > 0:
#            grad_W = grad_W + tf.matmul(tf.transpose(tf.concat([rnn_outputs[j-1], ones0],1)), delta)
#
#grad_V = tf.gradients(xs=V, ys=total_loss)[0]

###################
## SIGN MATCHING ##
###################

#C = tf.sign(W)
#
#grad_U = init_gradU
#grad_W = init_gradW
#for i in range(num_steps):
#    for j in range(i+1)[::-1]:
#        if j == i:
#            delta = tf.multiply(tf.matmul(delta0s[i][:,None],tf.transpose(V[0:state_size,:])), act_prime(rnn_outputs[j]))
#        else:
#            #Set to sign of W instead...
#            delta = tf.multiply(tf.matmul(delta, tf.transpose(C[0:state_size,:])), act_prime(rnn_outputs[j]))
#        grad_U = grad_U + tf.matmul(tf.transpose(rnn_inputs[j]), delta)
#        if j > 0:
#            grad_W = grad_W + tf.matmul(tf.transpose(tf.concat([rnn_outputs[j-1], ones0],1)), delta)
#
#grad_V = tf.gradients(xs=V, ys=total_loss)[0]

###############
## NODE PERT ##
###############

#Train network with node perturbation and REINFORCE
#Option 1: add training rule for B
#Perturb output with noise, keep noise for fitting

grad_U = init_gradU
grad_W = init_gradW
grad_B = init_gradB
for i in range(num_steps):
    for j in range(i+1)[::-1]:
        if j == i:
            delta = tf.multiply(tf.matmul(delta0s[i][:,None],tf.transpose(V[0:state_size,:])), act_prime(rnn_outputs[j]))
        else:
            delta = tf.multiply(tf.matmul(delta, tf.transpose(B)), act_prime(rnn_outputs[j]))
        grad_U = grad_U + tf.matmul(tf.transpose(rnn_inputs[j]), delta)
        if j > 0:
            grad_W = grad_W + tf.matmul(tf.transpose(tf.concat([rnn_outputs[j-1], ones0],1)), delta)
        grad_B = grad_B + tf.matmul(tf.matmul(B, tf.transpose(delta)) - \
            tf.transpose(tf.matmul(tf.diag(loss_pert[i] - loss[i])/var_xi/var_xi, noise_outputs[j])), delta)

grad_V = tf.gradients(xs=V, ys=total_loss)[0]

#grad_W = tf.gradients(xs=W, ys=total_loss)[0]
#grad_U = tf.gradients(xs=U, ys=total_loss)[0]

nb = tf.norm(B)
#new_B = B.assign(B - lmbda*grad_B)            
new_B = B.assign(B - lmbda*tf.clip_by_value(grad_B, -10, 10, name=None))

##################################################

new_U = U.assign(U - learning_rate*grad_U)            
new_W = W.assign(W - learning_rate*grad_W)           
new_V = V.assign(V - learning_rate*grad_V)          

#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
#train_step = [new_U, new_W, new_V]
train_step = [new_U, new_W, new_V, new_B]

#num_epochs = 30
#num_steps = 7
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#training_losses = []
#idx = 0
#epoch = next(gen_epochs(num_epochs, num_steps))
#verbose = True
#training_loss = 0
#training_state = np.zeros((batch_size, state_size))
#step = 0
#(X,Y) = next(epoch)
#tr_losses = sess.run([losses], feed_dict={x:X, y:Y, init_state:training_state})

def train_network(num_epochs, num_steps, state_size=state_size, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEpoch: %d"%idx)
            for step, (X, Y) in enumerate(epoch):
                tr_init_gradW = np.zeros((state_size+1, state_size))
                tr_init_gradB = np.zeros((state_size, state_size))
                tr_init_gradU = np.zeros((in_dim, state_size))
                tr_loss, tr_losses, training_loss_, training_state, _, norm_b = \
                    sess.run([loss, losses,
                              total_loss,
                              final_state,
                              train_step, nb],
                                  feed_dict={x:X, y:Y, init_state:training_state, \
                                  init_gradU:tr_init_gradU, init_gradW: tr_init_gradW, \
                                  init_gradB: tr_init_gradB})
                training_loss += training_loss_
                #print(np.mean(tr_loss), norm_b)
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step %d for last 100 steps: %f"%(step, training_loss/100))
                    training_losses.append(training_loss/100)
                    training_loss = 0

        #Test data
        training_state = np.zeros((batch_size, state_size))
        X_test = np.zeros((batch_size, in_dim, num_steps))
        Y_test = np.zeros((batch_size, num_steps))
        X_test[:,1,3] = 1
        X_test[:,1,2] = 1
        X_test[:,0,4] = 1
        Y_test[:,5] = -1
        output, tloss = sess.run([logits, total_loss], feed_dict={x:X_test, y:Y_test, init_state:training_state})

    return output, training_losses

test_output, training_losses = train_network(30, num_steps)
#plt.plot(training_losses)