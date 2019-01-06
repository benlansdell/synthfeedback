#!/usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import numpy as np
import numpy.random as rand
from numpy import random as rng
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import argparse
#from utils.utils import tf_matmul_r, tf_matmul_l

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

    #method = args.method
    #save = bool(args.save)

    method = 'backprop'
    #method = 'nodepert'
    #method = 'feedbackalignment'
    save = False

    anneal = True

    # Global config variables
    num_steps = 10 # number of truncated backprop steps ('n' in the discussion above)
    batch_size = 20
    in_dim = 4
    state_size = 50
    learning_rate = 1e-3
    alpha2 = 1
    activation = tf.tanh
    act_prime = lambda x: 1 - tf.multiply(x,x)

    #Node pert params
    lmbda = 5e-5
    var_xi = 0.5
    p_fire = 0.1 #prob of firing
    beta = 0.1

    acclimatize = False

    grad_max = 10

    N_epochs = 20
    N_episodes = 800

    report_rate = 100
    fn_out = './experiments/cartpole_rnn_np/%s_learning_rate_%f_lmbda_%f_varxi_%f.npz'%(method, learning_rate, lmbda, var_xi)

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
    'N_epochs': N_epochs,
    'N_episodes': N_episodes,
    'acclimatize':acclimatize
    }

    #method = 'backprop'
    #method = 'feedbackalignment'
    #method = 'nodepert'
    #method = 'weightsym'

    print("Using %s"%method)
    print("For %d epochs"%N_epochs)
    print("Learning rate: %f"%learning_rate)
    print("Lambda learning rate: %f"%lmbda)
    print("Variance xi: %f"%var_xi)
    print("Saving results: %d"%save)

    def rnn_cell(rnn_input, state, W, U):
        ones0 = tf.ones([batch_size, 1], tf.float32)
        state_p = tf.concat([state, ones0], 1)
        return activation(tf.matmul(rnn_input, U) + tf.matmul(state_p, W))

    def train_network(num_episodes, num_steps, state_size=state_size, verbose=True):
        xs = np.zeros((N_epochs, num_episodes, num_steps, in_dim))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            alignments = []

            for idx in range(N_epochs):
                print("Epoch: %d"%idx)
                training_loss = 0
                #training_x = np.zeros((batch_size, in_dim))
                training_x = rng.randn(batch_size, in_dim)
                #training_x[:,1] = np.pi
                training_state = np.zeros((batch_size, state_size))
                for step in range(num_episodes):
                    if step < 50 and acclimatize:
                        ts = train_step_B
                    else:
                        ts = train_step
                    tr_init_gradW = np.zeros((state_size+1, state_size))
                    tr_init_gradB = np.zeros((state_size, state_size))
                    tr_init_gradU = np.zeros((in_dim, state_size))
                    tr_init_gradV1 = np.zeros((state_size, state_size))
                    tr_init_gradS1 = np.zeros((state_size, state_size))
                    tr_loss, tr_losses, training_loss_, training_state, training_x, _, align, x_o = \
                        sess.run([loss, losses, total_loss, final_state, final_x, ts, aments, rnn_inputs], \
                                      feed_dict={init_state:training_state, init_x: training_x, \
                                      init_gradU: tr_init_gradU, init_gradW: tr_init_gradW, \
                                      init_gradB: tr_init_gradB, init_gradV1: tr_init_gradV1, \
                                      init_gradS1: tr_init_gradS1})
                    #print(np.array(x_o).shape)
                    xs[idx, step, :, :] = np.array(x_o)[:,0,:]
                    training_loss += training_loss_
                    if step % report_rate == 0 and step > 0:
                        if verbose:
                            print("Average loss at step %d for last %d steps: %f"%(step, report_rate, \
                                                                                   training_loss/report_rate))
                        training_losses.append(training_loss/report_rate)
                        alignments.append(align)
                        training_loss = 0

        return training_losses, step, alignments, xs

    ##############
    ## BACKPROP ##
    ##############

    def backprop():

        grad_U = init_gradU
        grad_W = init_gradW
        grad_B = init_gradB
        grad_V1 = init_gradV1
        grad_S1 = init_gradS1
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

        return grad_U, grad_W, grad_B, grad_V, grad_V1, grad_S1, alnments

    ########################
    ## FEEDBACK ALIGNMENT ##
    ########################

    def feedbackalignment():

        grad_U = init_gradU
        grad_W = init_gradW
        grad_B = init_gradB
        grad_V1 = init_gradV1
        grad_S1 = init_gradS1

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
        return grad_U, grad_W, grad_B, grad_V, grad_V1, grad_S1, alnments

    ###################
    ## SIGN MATCHING ##
    ###################

    def weightsym():
        C = tf.sign(W)
        
        grad_U = init_gradU
        grad_B = init_gradB
        grad_W = init_gradW
        grad_V1 = init_gradV1
        grad_S1 = init_gradS1

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

        return grad_U, grad_W, grad_B, grad_V, grad_V1, grad_S1, alnments

    ###############
    ## NODE PERT ##
    ###############

    def nodepert():

        grad_U = init_gradU
        grad_W = init_gradW
        grad_B = init_gradB
        grad_V1 = init_gradV1
        grad_S1 = init_gradS1
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

                np_est = tf.transpose(tf.matmul(tf.diag(loss_pert[i] - loss[i])/var_xi/var_xi, noise_outputs[j]))
                grad_V1 += tf.matmul(tf.transpose(delta), delta)
                grad_S1 += tf.matmul(np_est, delta)

        grad_V = tf.gradients(xs=V, ys=total_loss)[0]
        #grad_W = tf.gradients(xs=W, ys=total_loss)[0]
        #grad_U = tf.gradients(xs=U, ys=total_loss)[0]
        return grad_U, grad_W, grad_B, grad_V, grad_V1, grad_S1, alnments

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

    init_x = tf.zeros([batch_size, in_dim], dtype=np.float32)
    init_state = tf.zeros([batch_size, state_size], dtype=np.float32)
    init_gradW = tf.zeros([state_size+1, state_size], dtype=np.float32)
    init_gradB = tf.zeros([state_size, state_size], dtype=np.float32)
    init_gradU = tf.zeros([in_dim, state_size], dtype=np.float32)
    init_gradV1 = tf.zeros([state_size, state_size], dtype=np.float32)
    init_gradS1 = tf.zeros([state_size, state_size], dtype=np.float32)
    alignment = tf.zeros([in_dim, state_size], dtype=np.float32)

    ones0 = tf.ones([batch_size, 1], tf.float32)
    U = tf.Variable(rng.randn(in_dim, state_size)*alpha2, name="input_weights", dtype=tf.float32)
    W = tf.Variable(rng.randn(state_size+1, state_size)*alpha2, name="feedforward_weights", dtype=tf.float32)
    V = tf.Variable(rng.randn(state_size+1, 1)*alpha2, name="output_weights", dtype=tf.float32)

    V1 = tf.Variable(beta*np.eye(state_size), dtype=tf.float32)
    S1 = tf.Variable(np.zeros((state_size, state_size)), dtype=tf.float32)

    if method == 'nodepert':
        print("Using node pert B def")
        B = tf.matmul(S1, tf.matrix_inverse(V1))
    else:
        B = tf.Variable(rng.randn(state_size, state_size)*alpha2, name="feedback_weights", dtype=tf.float32)

    ##############################################
    ## Define the cartpole dynamics and network ##
    ##############################################

    x = init_x
    state = init_state
    state_p = init_state
    rnn_inputs = []
    rnn_outputs = []
    rnn_pert_outputs = []
    noise_outputs = []
    heights = []
    hs = []
    actions = []

    #Cart pole dynamics parameters 
    m = 1.1
    mp = 0.1
    g = 9.8
    l = 0.5
    tau = 0.01
    Fmax = 10
    max_h = 3
    gamma = 10

    #Equations of motion:
    #theta_dd = (m*g*sin(theta) - cos(theta)*(F + mp*l*theta_d*theta_d*sin(theta)))/((4/3)*m*l - mp*l*cos(theta)*cos(theta))
    #theta_d += tau*theta_dd
    #theta += tau*theta
    #h_dd = (F + mp*l*(theta_d*theta_d*sin(theta)-theta_dd*cos(theta)))/m
    #h_d += tau*h_dd
    #h += tau*h_d

    for idx in range(num_steps):
        mask = tf.random_uniform(state.shape) < p_fire
        xi = tf.multiply(tf.random_normal(state.shape)*var_xi, tf.to_float(mask))
        phi = tf.random_normal((batch_size,1))*Fmax/5
        #Compute new state
        state = rnn_cell(x, state, W, U)
        state_p = rnn_cell(x, state_p, W, U) + xi
        #Compute action
        action = tf.matmul(tf.concat([state, ones0], 1), V)
        F = tf.squeeze(Fmax*activation(action) + phi)
        #Compute new x
        #These could be wrong....?
        #theta_dd = (m*g*tf.sin(x[:,1]) - tf.cos(x[:,1])*(F + mp*l*x[:,0]*x[:,0]*tf.sin(x[:,1])))/((4/3)*m*l -\
        #            mp*l*tf.cos(x[:,1])*tf.cos(x[:,1]))
        #h_dd = (F + mp*l*(x[:,0]*x[:,0]*tf.sin(x[:,1])-theta_dd*tf.cos(x[:,1])))/m

        #Try these ones... from wikipedia:
        h_dd = (F - mp*l*x[:,0]*x[:,0]*tf.sin(x[:,1]) + mp*g*tf.sin(x[:,1])*tf.cos(x[:,1]))/(m - mp*tf.cos(x[:,1])*tf.cos(x[:,1]))
        theta_dd = (h_dd*tf.cos(x[:,1]) + g*tf.sin(x[:,1]))/l 

        h_dot = x[:,2] + tau*h_dd

        x_list = []
        x_list.append(x[:,0] + tau*theta_dd)   #x0 = theta_dot
        x_list.append(x[:,1] + tau*x[:,0])     #x1 = theta
        #x_list.append(tf.clip_by_value(x[:,2] + tau*h_dd, -2*max_h, 2*max_h))       #x2 = h_dot
        x_list.append(x[:,2] + tau*h_dd)       #x2 = h_dot
        #x_list.append(h_dot*tf.sign(x[:,3] + 10*max_h)*tf.sign(10*max_h - x[:,3]))
        #x_list.append(tf.clip_by_value(x[:,3] + tau*x[:,2], -10*max_h, 10*max_h))     #x3 = h
        x_list.append(x[:,3] + tau*x[:,2])     #x3 = h
        x = tf.stack(x_list, axis = 1)
        height = tf.cos(x[:,1])
        heights.append(height)
        actions.append(action)
        hs.append(x[:,3])
        rnn_inputs.append(x)
        rnn_outputs.append(state)
        rnn_pert_outputs.append(state_p)
        noise_outputs.append(xi)

    final_x = rnn_inputs[-1]
    final_state = rnn_outputs[-1]

    #Define loss function....
    #loss = 

    loss = [gamma*tf.pow(height-1, 2)/2 + tf.pow(tf.maximum(0.0, tf.abs(h) - max_h),2)/2 for h, height in zip(hs, heights)]
    losses = [gamma*tf.reduce_sum(tf.pow(height-1, 2))/2 + tf.pow(tf.maximum(0.0, tf.abs(h) - max_h),2)/2 for h, height in zip(hs, heights)]
    total_loss = tf.reduce_mean(losses)

    #Perturbed outputs and loss
    loss_pert = [gamma*tf.pow(height-1, 2)/2 + tf.pow(tf.maximum(0.0, tf.abs(h) - max_h),2)/2 for h, height in zip(hs, heights)]
    losses_pert = [gamma*tf.reduce_sum(tf.pow(height-1, 2))/2 + tf.pow(tf.maximum(0.0, tf.abs(h) - max_h),2)/2 for h, height in zip(hs, heights)]
    total_loss_pert = tf.reduce_mean(losses_pert)

    #This is wrong... need to be related to how the network activations relate to the height and displacement...
    #Use autograd
    #e0s = [(height+1) for height in heights]
    e0s = [tf.gradients(xs=action, ys=lo)[0][:,0] for (action,lo) in zip(actions, losses)]
    delta0s = e0s

    ##################################################

    grad_U, grad_W, grad_B, grad_V, grad_V1, grad_S1, aments = trainer()

    new_U = U.assign(U - learning_rate*grad_U)            
    new_W = W.assign(W - learning_rate*grad_W)           
    new_V = V.assign(V - learning_rate*grad_V)          
    new_V1 = V1.assign(V1 + grad_V1)
    new_S1 = S1.assign(S1 + grad_S1)

    if method == 'nodepert':
        train_step_B = [new_V1, new_S1]
        train_step = [new_U, new_W, new_V, new_V1, new_S1]
    else:
        new_B = B.assign(B - lmbda*tf.clip_by_value(grad_B, -grad_max, grad_max, name=None))
        train_step_B = [new_B]
        train_step = [new_U, new_W, new_V, new_B]

    #Save training losses, params, number of runs in epoch, alignment to BP
    all_losses = []
    all_alignments = []

    training_losses, n_in_epoch, alignments, xs = train_network(N_episodes, num_steps)
    all_losses.append(training_losses)
    all_alignments.append(alignments)

    if save:
        with open(fn_out, 'wb') as f:
            pickle.dump(all_losses, f)
            pickle.dump(all_alignments, f)
            pickle.dump(n_in_epoch, f)
            pickle.dump(params, f)
            pickle.dump(xs, f)

if __name__ == '__main__':
    main()