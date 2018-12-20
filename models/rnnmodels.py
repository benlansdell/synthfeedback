from base.base_model import BaseModel
import tensorflow as tf

from numpy import random as rng
import numpy as np 
from functools import reduce

from utils.utils import tf_matmul_r, tf_matmul_l, tf_eigvecs, tf_eigvals

"""All RNN models"""

class BPTTModel(BaseModel):
    def __init__(self, config):
        super(BPTTModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def rnn_cell(self, rnn_input, state, W, U):
        batch_size = self.config.batch_size
        ones0 = tf.ones([batch_size, 1], tf.float32)
        state_p = tf.concat([state, ones0], 1)
        return self.activation(tf.matmul(rnn_input, U) + tf.matmul(state_p, W))

    def build_model(self):

        self.is_training = tf.placeholder(tf.bool)

        self.activation = tf.tanh
        self.act_prime = lambda x: 1 - tf.multiply(x,x)

        batch_size = self.config.batch_size
        in_dim = self.config.in_dim
        num_steps = self.config.trial_length
        state_size = self.config.state_size[0]
        learning_rate = self.config.learning_rate
        alpha2 = 1

        x = tf.placeholder(tf.float32, [batch_size, in_dim, num_steps], name='input_placeholder')
        y = tf.placeholder(tf.float32, [batch_size, 1, num_steps], name='labels_placeholder')
        init_state = tf.zeros([batch_size, state_size], dtype=np.float32)
        init_gradW = tf.zeros([state_size+1, state_size], dtype=np.float32)
        init_gradU = tf.zeros([in_dim, state_size], dtype=np.float32)
        rnn_inputs = tf.unstack(x, axis=2)
        alignment = tf.zeros([in_dim, state_size], dtype=np.float32)
        ones0 = tf.ones([batch_size, 1], tf.float32)
        U = tf.get_variable('U', [in_dim, state_size])
        W = tf.get_variable('W', [state_size+1, state_size])
    
        self.init_state = init_state

        state = init_state
        state_p = init_state
        rnn_outputs = []
        rnn_pert_outputs = []
        noise_outputs = []
        for rnn_input in rnn_inputs:
            state = self.rnn_cell(rnn_input, state, W, U)
            rnn_outputs.append(state)
            final_state = rnn_outputs[-1]
    
        self.final_state = final_state

        V = tf.get_variable('V', [state_size+1, 1])
        logits = [tf.squeeze(tf.matmul(tf.concat([rnn_output, ones0], 1), V)) for rnn_output in rnn_outputs]
        logits_as_t = tf.stack(logits, axis=1)
        y_as_list = tf.unstack(tf.squeeze(y), num=num_steps, axis=1)
        loss = [tf.pow(logit-label, 2)/2 for logit, label in zip(logits, y_as_list)]
        losses = [tf.reduce_sum(tf.pow(logit-label, 2))/2 for logit, label in zip(logits, y_as_list)]
        total_loss = tf.reduce_mean(losses)

        grad_V = tf.gradients(xs=V, ys=total_loss)[0]
        grad_W = tf.gradients(xs=W, ys=total_loss)[0]
        grad_U = tf.gradients(xs=U, ys=total_loss)[0]

        new_U = U.assign(U - learning_rate*grad_U)            
        new_W = W.assign(W - learning_rate*grad_W)           
        new_V = V.assign(V - learning_rate*grad_V)
        self.acc = tf.reduce_mean(tf.to_float(tf.equal((y_as_list[-1] > 0),(logits[-1]>0))))
        self.pred = logits
        self.loss = total_loss
        self.x = x
        self.y = y      
        self.train_step = [new_U, new_W, new_V]

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _set_training_metrics(self, e, Bs, Ws, dls, ls, eta):

        k = 6
        #Alignment of B_{i+1}e with dl_i (feedback alignment)
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(e, tf.transpose(Bs[idx][:,:]))[0,:]
            delta_bp = dls[idx][0,:]
            #error_fa = tf.norm(delta_fa - dls[idx])
            alignment = 180/np.pi*tf.reduce_sum(tf.multiply(delta_fa,delta_bp))/tf.norm(delta_fa)/tf.norm(delta_bp)
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)


class FARNNModel(BaseModel):
    def __init__(self, config):
        super(FARNNModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def rnn_cell(self, rnn_input, state, W, U, B):
        batch_size = self.config.batch_size
        ones0 = tf.ones([batch_size, 1], tf.float32)
        state_p = tf.concat([state, ones0], 1)
        return self.activation(tf.matmul(rnn_input, U) + tf.matmul(state_p, W))

    def build_model(self):

        self.is_training = tf.placeholder(tf.bool)

        self.activation = tf.tanh
        self.act_prime = lambda x: 1 - tf.multiply(x,x)

        batch_size = self.config.batch_size
        in_dim = self.config.in_dim
        num_steps = self.config.trial_length
        state_size = self.config.state_size[0]
        learning_rate = self.config.learning_rate
        alpha2 = 1

        x = tf.placeholder(tf.float32, [batch_size, in_dim, num_steps], name='input_placeholder')
        y = tf.placeholder(tf.float32, [batch_size, 1, num_steps], name='labels_placeholder')
        init_state = tf.zeros([batch_size, state_size], dtype=np.float32)
        init_gradW = tf.zeros([state_size+1, state_size], dtype=np.float32)
        init_gradB = tf.zeros([state_size, state_size], dtype=np.float32)
        init_gradU = tf.zeros([in_dim, state_size], dtype=np.float32)
        rnn_inputs = tf.unstack(x, axis=2)
        alignment = tf.zeros([in_dim, state_size], dtype=np.float32)
        ones0 = tf.ones([batch_size, 1], tf.float32)
        U = tf.get_variable('U', [in_dim, state_size])
        W = tf.get_variable('W', [state_size+1, state_size])
        B = tf.get_variable('B', [state_size, state_size])
    
        self.init_state = init_state

        state = init_state
        state_p = init_state
        rnn_outputs = []
        rnn_pert_outputs = []
        noise_outputs = []
        for rnn_input in rnn_inputs:
            state = self.rnn_cell(rnn_input, state, W, U, B)
            rnn_outputs.append(state)
            final_state = rnn_outputs[-1]
    
        self.final_state = final_state

        V = tf.get_variable('V', [state_size+1, 1])
        logits = [tf.squeeze(tf.matmul(tf.concat([rnn_output, ones0], 1), V)) for rnn_output in rnn_outputs]
        logits_as_t = tf.stack(logits, axis=1)
        y_as_list = tf.unstack(tf.squeeze(y), num=num_steps, axis=1)
        loss = [tf.pow(logit-label, 2)/2 for logit, label in zip(logits, y_as_list)]
        losses = [tf.reduce_sum(tf.pow(logit-label, 2))/2 for logit, label in zip(logits, y_as_list)]
        total_loss = tf.reduce_mean(losses)

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
                        delta = tf.multiply(tf.matmul(delta0s[i][:,None],tf.transpose(V[0:state_size,:])), self.act_prime(rnn_outputs[j]))
                    else:
                        delta = tf.multiply(tf.matmul(delta, tf.transpose(B)), self.act_prime(rnn_outputs[j]))
                    grad_U = grad_U + tf.matmul(tf.transpose(rnn_inputs[j]), delta)
                    if j > 0:
                        grad_W = grad_W + tf.matmul(tf.transpose(tf.concat([rnn_outputs[j-1], ones0],1)), delta)
            
            #Compute alignment between BP and FA for each time in the network
            grad_V = tf.gradients(xs=V, ys=total_loss)[0]
    
            return grad_U, grad_W, grad_B, grad_V, alnments

        e0s = [(logit - label) for logit, label in zip(logits, y_as_list)]
        delta0s = e0s

        grad_U, grad_W, grad_B, grad_V, alments = feedbackalignment()

        new_B = B.assign(B - learning_rate*grad_B)            
        new_U = U.assign(U - learning_rate*grad_U)            
        new_W = W.assign(W - learning_rate*grad_W)           
        new_V = V.assign(V - learning_rate*grad_V)
        self.acc = tf.reduce_mean(tf.to_float(tf.equal((y_as_list[-1] > 0),(logits[-1]>0))))
        self.pred = logits
        self.loss = total_loss
        self.x = x
        self.y = y      
        self.train_step = [new_U, new_W, new_V]

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _set_training_metrics(self, e, Bs, Ws, dls, ls, eta):

        k = 6
        #Alignment of B_{i+1}e with dl_i (feedback alignment)
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(e, tf.transpose(Bs[idx][:,:]))[0,:]
            delta_bp = dls[idx][0,:]
            #error_fa = tf.norm(delta_fa - dls[idx])
            alignment = 180/np.pi*tf.reduce_sum(tf.multiply(delta_fa,delta_bp))/tf.norm(delta_fa)/tf.norm(delta_bp)
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)