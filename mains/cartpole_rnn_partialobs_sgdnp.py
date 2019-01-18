
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import numpy as np
import numpy.random as rand
from numpy import random as rng
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import argparse
import seaborn as sns
from utils.utils import tf_matmul_r, tf_matmul_l

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-m', '--method',
        metavar='M',
        default='None',
        help='Algorithm name')
    argparser.add_argument(
        '-s', '--save', default=True)
    args = argparser.parse_args()
    return args

def main():
    args = get_args()
    method = args.method
    save = bool(args.save)

    # Global config variables
    anneal = True
    num_steps = 10 # number of truncated backprop steps ('n' in the discussion above)
    batch_size = 20
    in_dim = 4
    state_size = 50
    learning_rate = 1e-3
    alpha2 = 1
    activation = tf.tanh
    act_prime = lambda x: 1.0 - tf.multiply(x,x)
    acclimatize = True
    grad_max = 10
    N_epochs = 5000
    N_episodes = 10

    #Node pert params
    lmbda = 5e-3
    var_xi = 0.01
    p_fire = 1.0 #prob of firing

    beta = 0.1

    report_rate = 100
    fn_out = './experiments/cartpole_rnn_partialobs_sgdnp/%s_learning_rate_%f_lmbda_%f_varxi_%f.npz'%(method, learning_rate, lmbda, var_xi)

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

    print("Using %s"%method)
    print("For %d epochs"%N_epochs)
    print("Learning rate: %f"%learning_rate)
    print("Lambda learning rate: %f"%lmbda)
    print("Variance xi: %f"%var_xi)
    print("Saving results: %d"%save)

    def rnn_cell_bp(rnn_input, state, W, U, B):
        ones0 = tf.ones([batch_size, 1], tf.float32)
        state_p = tf.concat([state, ones0], 1)
        return activation(tf.matmul(rnn_input[:,0:-1:2], U) + tf.matmul(state_p, W))

    def rnn_cell_fa(rnn_input, state, W, U, B):
        ones0 = tf.ones([batch_size, 1], tf.float32)
        state_p = tf.concat([state, ones0], 1)
        #return activation(tf.matmul(rnn_input[:,0:-1:2], U) + tf_matmul_r(state_p, W, B))
        return activation(tf_matmul_r(rnn_input[:,0:-1:2], U, B[0:2,:]) + tf_matmul_r(state_p, W, B))

    if method == 'backprop':
        rnn_cell = rnn_cell_bp
    else:
        rnn_cell = rnn_cell_fa

    def train_network(num_episodes, num_steps, state_size=state_size, verbose=True):
        xs = np.zeros((N_epochs, num_episodes, num_steps, batch_size, in_dim))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            alignments = []

            for idx in range(N_epochs):
                print("Epoch: %d"%idx)
                if idx < 4 and acclimatize:
                    ts = train_step_B
                else:
                    ts = train_step
                training_loss = 0
                training_x = np.zeros((batch_size, in_dim))
                training_state = np.zeros((batch_size, state_size))
                for step in range(num_episodes):
                    tr_init_gradW = np.zeros((state_size+1, state_size))
                    tr_init_gradB = np.zeros((state_size+1, state_size))
                    tr_init_gradC = np.zeros((state_size+1, 1))
                    tr_init_gradU = np.zeros((int(in_dim/2), state_size))
                    tr_loss, tr_losses, training_loss_, training_state, training_x, _, align, x_o = \
                        sess.run([loss, losses, total_loss, final_state, final_x, ts, aments, rnn_inputs], \
                                      feed_dict={init_state:training_state, init_x: training_x, \
                                      init_gradU: tr_init_gradU, init_gradW: tr_init_gradW, \
                                      init_gradB: tr_init_gradB, init_gradC: tr_init_gradC})
                    xs[idx, step, :, :, :] = np.array(x_o)[:,:,:]
                    training_loss += training_loss_
                if idx % report_rate == 0 and idx > 0:
                    if verbose:
                        print("Average loss at epoch %d for last %d steps: %f"%(idx, report_rate, \
                                                                               training_loss/report_rate/num_episodes))
                    training_losses.append(training_loss/report_rate/num_episodes)
                    alignments.append(align)
                    training_loss = 0

        return training_losses, step, alignments, xs

    ##############
    ## BACKPROP ##
    ##############

    def backprop():
        grad_B = init_gradB
        grad_C = init_gradC
        alnments = []
        grad_V = tf.gradients(xs=V, ys=total_loss)[0]
        grad_W = tf.gradients(xs=W, ys=total_loss)[0]
        grad_U = tf.gradients(xs=U, ys=total_loss)[0]
        return grad_U, grad_W, grad_B, grad_C, grad_V, alnments

    ###############
    ## NODE PERT ##
    ###############

    def nodepert():
        grad_B = init_gradB
        grad_C = init_gradC
        alnments = []
        for i in range(num_steps):
            for j in range(i+1)[::-1]:
                np_est = tf.matmul(tf.diag(loss_pert[i] - loss[i])/var_xi/var_xi, noise_outputs[j])
                delta = tf.gradients(xs = rnn_outputs[j], ys = loss[i])[0]
                aux_loss = tf.reduce_sum(tf.pow(np_est - delta, 2))
                grad_B += tf.squeeze(tf.gradients(xs = B, ys = aux_loss))

        grad_V = tf.gradients(xs=V, ys=total_loss)[0]
        grad_W = tf.gradients(xs=W, ys=total_loss)[0]
        grad_U = tf.gradients(xs=U, ys=total_loss)[0]
        return grad_U, grad_W, grad_B, grad_C, grad_V, alnments

    if method == 'backprop':
        trainer = backprop
    elif method == 'feedbackalignment':
        trainer = backprop
    elif method == 'nodepert':
        trainer = nodepert
    else:
        raise NotImplementedError

    init_x = tf.zeros([batch_size, in_dim], dtype=np.float32)
    init_state = tf.zeros([batch_size, state_size], dtype=np.float32)
    init_gradW = tf.zeros([state_size+1, state_size], dtype=np.float32)
    init_gradB = tf.zeros([state_size+1, state_size], dtype=np.float32)
    init_gradC = tf.zeros([state_size+1, 1], dtype=np.float32)
    init_gradU = tf.zeros([in_dim/2, state_size], dtype=np.float32)
    alignment = tf.zeros([in_dim/2, state_size], dtype=np.float32)

    ones0 = tf.ones([batch_size, 1], tf.float32)
    U = tf.Variable(rng.randn(int(in_dim/2), state_size)*alpha2, name="input_weights", dtype=tf.float32)
    W = tf.Variable(rng.randn(state_size+1, state_size)*alpha2, name="feedforward_weights", dtype=tf.float32)
    V = tf.Variable(rng.randn(state_size+1, 1)*alpha2, name="output_weights", dtype=tf.float32)

    B = tf.Variable(rng.randn(state_size+1, state_size)*alpha2, name="feedback_weights", dtype=tf.float32)
    C = tf.Variable(rng.randn(state_size+1, 1)*alpha2, name="feedback_weights", dtype=tf.float32)

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

    m = 1.1
    mp = 0.1
    g = 9.8
    l = 0.5
    tau = 0.04
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
        mask = tf.random_uniform([batch_size, state_size]) < p_fire
        xi = tf.multiply(tf.random_normal([batch_size, state_size])*var_xi, tf.to_float(mask))
        phi = tf.random_normal((batch_size,1))*Fmax/500
        #Compute new state
        state = rnn_cell(x, state, W, U, B)
        state_p = rnn_cell(x, state_p, W, U, B) + xi[:,0:state_size]
        #Compute action
        if method == 'backprop':
            action = tf.matmul(tf.concat([state, ones0], 1), V)
        else:
            action = tf_matmul_r(tf.concat([state, ones0], 1), V, C)            
        F = tf.squeeze(Fmax*activation(action) + phi)
        #Compute new x
        theta_dd = (m*g*tf.sin(x[:,1]) - tf.cos(x[:,1])*(F + mp*l*x[:,0]*x[:,0]*tf.sin(x[:,1])))/((4/3)*m*l -\
                    mp*l*tf.cos(x[:,1])*tf.cos(x[:,1]))
        h_dd = (F + mp*l*(x[:,0]*x[:,0]*tf.sin(x[:,1])-theta_dd*tf.cos(x[:,1])))/m
        
        #h_dd = (F - mp*l*x[:,0]*x[:,0]*tf.sin(x[:,1]) + mp*g*tf.sin(x[:,1])*tf.cos(x[:,1]))/(m - mp*tf.cos(x[:,1])*tf.cos(x[:,1]))
        #theta_dd = (h_dd*tf.cos(x[:,1]) + g*tf.sin(x[:,1]))/l 

        x_list = []
        x_list.append(x[:,0] + tau*theta_dd)   #x0 = theta_dot
        x_list.append(x[:,1] + tau*x[:,0])     #x1 = theta
        x_list.append(x[:,2] + tau*h_dd)       #x2 = h_dot
        x_list.append(x[:,3] + tau*x[:,2])     #x3 = h
        #x_list.append(tf.clip_by_value(x[:,3] + tau*x[:,2], -4*max_h, 4*max_h))     #x3 = h
        x = tf.stack(x_list, axis = 1)
        #height = tf.cos(x[:,1])
        height = x[:,1]
        heights.append(height)
        hs.append(x[:,2])
        rnn_inputs.append(x)
        rnn_outputs.append(state)
        rnn_pert_outputs.append(state_p)
        noise_outputs.append(xi)
        actions.append(action)
        
    final_x = rnn_inputs[-1]
    final_state = rnn_outputs[-1]

    #Define loss function....
    loss = [gamma*tf.pow(height, 2)/2 + tf.pow(tf.maximum(0.0, tf.abs(h) - max_h),2)/2 for h, height in zip(hs, heights)]
    losses = [gamma*tf.reduce_sum(tf.pow(height, 2))/2 + tf.pow(tf.maximum(0.0, tf.abs(h) - max_h),2)/2 for h, height in zip(hs, heights)]
    total_loss = tf.reduce_mean(losses)

    #Perturbed outputs and loss
    loss_pert = [gamma*tf.pow(height, 2)/2 + tf.pow(tf.maximum(0.0, tf.abs(h) - max_h),2)/2 for h, height in zip(hs, heights)]
    losses_pert = [gamma*tf.reduce_sum(tf.pow(height, 2))/2 + tf.pow(tf.maximum(0.0, tf.abs(h) - max_h),2)/2 for h, height in zip(hs, heights)]
    total_loss_pert = tf.reduce_mean(losses_pert)

    ##################################################

    grad_U, grad_W, grad_B, grad_C, grad_V, aments = trainer()
    new_U = U.assign(U - learning_rate*grad_U)            
    new_W = W.assign(W - learning_rate*grad_W)           
    new_V = V.assign(V - learning_rate*grad_V)          

    new_C = C.assign(C - lmbda*tf.clip_by_value(grad_C, -grad_max, grad_max, name=None))
    new_B = B.assign(B - lmbda*tf.clip_by_value(grad_B, -grad_max, grad_max, name=None))
    train_step_B = [new_B, new_C]
    train_step = [new_U, new_W, new_V, new_B, new_C]

    #Save training losses, params, number of runs in epoch, alignment to BP
    all_losses = []
    all_alignments = []
    training_losses, n_in_epoch, alignments, xs = train_network(N_episodes, num_steps)
    all_losses.append(training_losses)
    all_alignments.append(alignments)

    if save:
        with open(fn_out, 'wb') as f:
            to_save = {
                'all_losses': all_losses,
                'all_alignments': all_alignments,
                'n_in_epoch': n_in_epoch,
                'params': params,
                'xs':xs
            }
            pickle.dump(to_save, f)

if __name__ == '__main__':
    main()