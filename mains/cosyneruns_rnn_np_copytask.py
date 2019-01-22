#!/usr/bin/env python
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]=''

import numpy.random as rand
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import random as rng

import pickle

import argparse

from utils.utils import tf_matmul_r, tf_matmul_l

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-m', '--method',
        metavar='M',
        default='None',
        help='Algorithm name')
    argparser.add_argument(
        '-s', '--save', default=False)
    args = argparser.parse_args()
    return args

def main():

    args = get_args()
    method = args.method
    save = bool(args.save)
    
    anneal = True

    # Global config variables
    num_steps = 10 # number of truncated backprop steps ('n' in the discussion above)
    batch_size = 200
    in_dim = 2
    state_size = 50
    learning_rate = 5e-6
    alpha2 = 1
    activation = tf.tanh
    act_prime = lambda x: 1 - tf.multiply(x,x)

    #Node pert params
    lmbda = 5e-5
    var_xi = 0.5
    p_fire = 0.1 #prob of firing

    acclimatize = True

    grad_max = 10

    N_runs = 3
    N_epochs = 500

    N_inputs = 10

    fn_out = './experiments/rnn_np_copytask/%s_learning_rate_%f_lmbda_%f_varxi_%f_Ninputs_%d.npz'%(method, learning_rate, lmbda, var_xi, N_inputs)

    #Things to save with output
    params = {
    'num_steps': num_steps,
    'batch_size': batch_size,
    'in_dim': in_dim,
    'state_size': state_size,
    'learning_rate': learning_rate,
    'alpha2': alpha2,
    'lmbda': lmbda,
    'var_xi': var_xi,
    'p_fire': p_fire,
    'grad_max': grad_max,
    'N_runs': N_runs,
    'N_epoch': N_epochs,
    'acclimatize':acclimatize
    }

    #method = 'backprop'
    #method = 'feedbackalignment'
    #method = 'nodepert'
    #method = 'weightsym'

    print("Using %s"%method)
    print("For %d runs"%N_runs)
    print("Learning rate: %f"%learning_rate)
    print("Lambda learning rate: %f"%lmbda)
    print("Variance xi: %f"%var_xi)
    print("Saving results: %d"%save)

    def rnn_cell(rnn_input, state, W, U):
        ones0 = tf.ones([batch_size, 1], tf.float32)
        state_p = tf.concat([state, ones0], 1)
        return activation(tf.matmul(rnn_input, U) + tf.matmul(state_p, W))

    def gen_data(N_inputs, size=1000000):
        X = np.zeros((in_dim,size))
        Y = np.zeros(size)
        #Add some go cues at random points
        for i in range(0,size,2*N_inputs):
            Xs = [1 if a < 0.5 else -1 for a in rand.random(N_inputs)]
            Y[(i+N_inputs):(i+2*N_inputs)] = Xs 
            X[0,i:(i+N_inputs)] = Xs 
            X[1,i] = 1
            X[1,(i+N_inputs)] = -1
        return X, np.array(Y, dtype = float)

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
            yield gen_batch(gen_data(N_inputs), batch_size, num_steps)

    def train_network(num_epochs, num_steps, state_size=state_size, verbose=True):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            alignments = []
            for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):

                if idx < 4 and acclimatize:
                    ts = train_step_B
                else:
                    ts = train_step

                training_loss = 0
                training_state = np.zeros((batch_size, state_size))
                if verbose:
                    print("\nEpoch: %d"%idx)
                for step, (X, Y) in enumerate(epoch):
                    tr_init_gradW = np.zeros((state_size+1, state_size))
                    tr_init_gradB = np.zeros((state_size, state_size))
                    tr_init_gradU = np.zeros((in_dim, state_size))
                    tr_loss, tr_losses, training_loss_, training_state, _, align = \
                        sess.run([loss, losses,
                                  total_loss,
                                  final_state,
                                  ts, aments],
                                      feed_dict={x:X, y:Y, init_state:training_state, \
                                      init_gradU:tr_init_gradU, init_gradW: tr_init_gradW, \
                                      init_gradB: tr_init_gradB})
                    training_loss += training_loss_
                    #print(np.mean(tr_loss), norm_b)
                    if step % 100 == 0 and step > 0:
                        if verbose:
                            print("Average loss at step %d for last 100 steps: %f"%(step, training_loss/100))
                        training_losses.append(training_loss/100)
                        alignments.append(align)
                        training_loss = 0
                #if anneal:
                #    lmbda = lmbda / np.sqrt(idx)

            #Test data
            #training_state = np.zeros((batch_size, state_size))
            #X_test = np.zeros((batch_size, in_dim, num_steps))
            #Y_test = np.zeros((batch_size, num_steps))
            #X_test[:,1,3] = 1
            #X_test[:,1,2] = 1
            #X_test[:,0,4] = 1
            #Y_test[:,5] = -1
            #output, tloss = sess.run([logits, total_loss], feed_dict={x:X_test, y:Y_test, init_state:training_state})

        return training_losses, step, alignments

    ##############
    ## BACKPROP ##
    ##############

    def backprop():

        grad_U = init_gradU
        grad_W = init_gradW
        grad_B = init_gradB
        alnments = []
        for i in range(num_steps):
            for j in range(i+1)[::-1]:
                if j == i:
                    delta = tf.multiply(tf.matmul(delta0s[i][:,None],tf.transpose(V[0:state_size,:])), act_prime(rnn_outputs[j]))
                else:
                    delta = tf.multiply(tf.matmul(delta, tf.transpose(W[0:state_size,:])), act_prime(rnn_outputs[j]))
                grad_U = grad_U + tf.matmul(tf.transpose(rnn_inputs[j]), delta)
                if j > 0:
                    grad_W = grad_W + tf.matmul(tf.transpose(tf.concat([rnn_outputs[j-1], ones0],1)), delta)
        
        grad_V = tf.gradients(xs=V, ys=total_loss)[0]
        grad_W = tf.gradients(xs=W, ys=total_loss)[0]
        grad_U = tf.gradients(xs=U, ys=total_loss)[0]

        return grad_U, grad_W, grad_B, grad_V, alnments

    ########################
    ## FEEDBACK ALIGNMENT ##
    ########################

    def feedbackalignment():

        grad_U = init_gradU
        grad_W = init_gradW
        grad_B = init_gradB

        alnments = []
        for i in range(num_steps):
            for j in range(i+1)[::-1]:

                #Compute alignment for the last window, taking mean over all batches
                if i == num_steps - 1:
                    a1 = tf.gradients(xs=rnn_outputs[j], ys=loss[i])[0]
                    if j == i:
                        a2 = tf.matmul(delta0s[i][:,None], tf.transpose(V[0:state_size,:]))
                    else:
                        a2 = tf.matmul(delta, tf.transpose(B))
                    a = tf.reduce_mean(tf.reduce_sum(tf.multiply(a1,a2),1)/tf.norm(a1,axis=1)/tf.norm(a2,axis=1))
                    alnments.append(a)

                if j == i:
                    delta = tf.multiply(tf.matmul(delta0s[i][:,None],tf.transpose(V[0:state_size,:])), act_prime(rnn_outputs[j]))
                else:
                    delta = tf.multiply(tf.matmul(delta, tf.transpose(B)), act_prime(rnn_outputs[j]))
                grad_U = grad_U + tf.matmul(tf.transpose(rnn_inputs[j]), delta)
                if j > 0:
                    grad_W = grad_W + tf.matmul(tf.transpose(tf.concat([rnn_outputs[j-1], ones0],1)), delta)
        
        #Compute alignment between BP and FA for each time in the network

        grad_V = tf.gradients(xs=V, ys=total_loss)[0]

        return grad_U, grad_W, grad_B, grad_V, alnments

    ###################
    ## SIGN MATCHING ##
    ###################

    def weightsym():
        C = tf.sign(W)
        
        grad_U = init_gradU
        grad_B = init_gradB
        grad_W = init_gradW
        alnments = []
        for i in range(num_steps):
            for j in range(i+1)[::-1]:

                #Compute alignment for the last window, taking mean over all batches
                if i == num_steps - 1:
                    a1 = tf.gradients(xs=rnn_outputs[j], ys=loss[i])[0]
                    if j == i:
                        a2 = tf.matmul(delta0s[i][:,None], tf.transpose(V[0:state_size,:]))
                    else:
                        a2 = tf.matmul(delta, tf.transpose(B))
                    a = tf.reduce_mean(tf.reduce_sum(tf.multiply(a1,a2),1)/tf.norm(a1,axis=1)/tf.norm(a2,axis=1))
                    alnments.append(a)

                if j == i:
                    delta = tf.multiply(tf.matmul(delta0s[i][:,None],tf.transpose(V[0:state_size,:])), act_prime(rnn_outputs[j]))
                else:
                    #Set to sign of W instead...
                    delta = tf.multiply(tf.matmul(delta, tf.transpose(C[0:state_size,:])), act_prime(rnn_outputs[j]))
                grad_U = grad_U + tf.matmul(tf.transpose(rnn_inputs[j]), delta)
                if j > 0:
                    grad_W = grad_W + tf.matmul(tf.transpose(tf.concat([rnn_outputs[j-1], ones0],1)), delta)
        
        grad_V = tf.gradients(xs=V, ys=total_loss)[0]

        return grad_U, grad_W, grad_B, grad_V, alnments

    ###############
    ## NODE PERT ##
    ###############

    def nodepert():

        grad_U = init_gradU
        grad_W = init_gradW
        grad_B = init_gradB
        alnments = []
        for i in range(num_steps):
            for j in range(i+1)[::-1]:

                #Compute alignment for the last window, taking mean over all batches
                if i == num_steps - 1:
                    a1 = tf.gradients(xs=rnn_outputs[j], ys=loss[i])[0]
                    if j == i:
                        a2 = tf.matmul(delta0s[i][:,None], tf.transpose(V[0:state_size,:]))
                    else:
                        a2 = tf.matmul(delta, tf.transpose(B))
                    a = tf.reduce_mean(tf.reduce_sum(tf.multiply(a1,a2),1)/tf.norm(a1,axis=1)/tf.norm(a2,axis=1))
                    alnments.append(a)

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
        return grad_U, grad_W, grad_B, grad_V, alnments

    ##################################################


    if method == 'backprop':
        trainer = backprop
    elif method == 'feedbackalignment':
        trainer = feedbackalignment
    elif method == 'nodepert':
        trainer = nodepert
    elif method == 'weightsym':
        trainer = weightsym
    else:
        raise NotImplementedError

    x = tf.placeholder(tf.float32, [batch_size, in_dim, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.float32, [batch_size, num_steps], name='labels_placeholder')
    init_state = tf.zeros([batch_size, state_size], dtype=np.float32)
    init_gradW = tf.zeros([state_size+1, state_size], dtype=np.float32)
    init_gradB = tf.zeros([state_size, state_size], dtype=np.float32)
    init_gradU = tf.zeros([in_dim, state_size], dtype=np.float32)

    rnn_inputs = tf.unstack(x, axis=2)

    alignment = tf.zeros([in_dim, state_size], dtype=np.float32)

    ones0 = tf.ones([batch_size, 1], tf.float32)
    U = tf.get_variable('U', [in_dim, state_size])
    W = tf.get_variable('W', [state_size+1, state_size])
    B = tf.Variable(rng.randn(state_size, state_size)*alpha2, name="feedback_weights", dtype=tf.float32)

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

    V = tf.get_variable('V', [state_size+1, 1])
    logits = [tf.squeeze(tf.matmul(tf.concat([rnn_output, ones0], 1), V)) for rnn_output in rnn_outputs]
    logits_as_t = tf.stack(logits, axis=1)
    y_as_list = tf.unstack(y, num=num_steps, axis=1)
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

    grad_U, grad_W, grad_B, grad_V, aments = trainer()

    new_B = B.assign(B - lmbda*tf.clip_by_value(grad_B, -grad_max, grad_max, name=None))
    new_U = U.assign(U - learning_rate*grad_U)            
    new_W = W.assign(W - learning_rate*grad_W)           
    new_V = V.assign(V - learning_rate*grad_V)          

    train_step_B = [new_B]
    train_step = [new_U, new_W, new_V, new_B]

    #Save training losses, params, number of runs in epoch, alignment to BP 

    all_losses = []
    all_alignments = []

    for idx in range(N_runs):
        training_losses, n_in_epoch, alignments = train_network(N_epochs, num_steps)
        all_losses.append(training_losses)
        all_alignments.append(alignments)

    if save:
        with open(fn_out, 'wb') as f:
            pickle.dump(all_losses, f)
            pickle.dump(all_alignments, f)
            pickle.dump(n_in_epoch, f)
            pickle.dump(params, f)

if __name__ == '__main__':
    main()