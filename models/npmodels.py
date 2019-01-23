from base.base_model import BaseModel
import tensorflow as tf

from numpy import random as rng
import numpy as np 

from utils.utils import tf_matmul_r, tf_matmul_l, tf_eigvecs, tf_eigvals

"""Node perturbation models"""

#Network building functions
def weight_variable(shape):
    sigma = np.sqrt(2.0/shape[0])
    initial = tf.truncated_normal(shape, stddev=sigma)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def weight_w_bias(shape):
    sigma = np.sqrt(2.0/shape[0])
    W = tf.truncated_normal(shape, stddev=sigma)
    b = tf.constant(0.1, shape=[1,shape[1]])
    initial = tf.concat([W, b], 0)
    return tf.Variable(initial)

def fc_layer(prev, input_size, output_size, config):
    W = weight_variable([input_size+1, output_size])
    e = tf.ones([config.batch_size, 1], tf.float32)
    prev_aug = tf.concat([prev, e], 1)
    return tf.matmul(prev_aug, W), W, prev_aug

def fa_layer(prev, input_size, output_size, config):
    B = weight_variable([input_size+1, output_size])
    W = weight_w_bias([input_size, output_size])
    e = tf.ones([config.batch_size, 1], tf.float32)
    prev_aug = tf.concat([prev, e], 1)
    return tf_matmul_r(prev_aug, W, B), W, B

def fc_layer_noise_act(prev, W, var_xi, config, activation):
    e = tf.ones([config.batch_size, 1], tf.float32)
    prev_aug = tf.concat([prev, e], 1)
    n = tf.matmul(prev_aug, W)
    xi = tf.random_normal(shape=tf.shape(n), mean=0.0, stddev=var_xi, dtype=tf.float32)
    l = activation(n)
    return l + xi, n, xi

def tf_align(x, y):
    #Check they have the right dimensions...
    #if x.get_shape() != y.get_shape():
    #print "Vectors different shape"
    #print x.get_shape(), y.get_shape()
    theta = 180/np.pi*tf.abs(tf.acos(tf.reduce_sum(tf.multiply(x,y))/tf.norm(x)/tf.norm(y)))
    #if (theta > 90) and (theta < 180):
    #    theta = 90 - theta
    return tf.cond(tf.logical_and(tf.less(90.0,theta), tf.less(theta,180.0)), lambda: 180.0 - theta, lambda: theta)

class NPModel4(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(NPModel4, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # set initial feedforward and feedback weights
        p = self.config.state_size[0]
        #m = 512
        #j = 200
        m = 50
        j = 20
        n = 10
        var_xi = self.config.var_xi

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)
        B1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights2", dtype=tf.float32)

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        e1 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = tf.sigmoid(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e1], 1)
        xi1 = tf.random_normal(shape=tf.shape(h1_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h1_tilde = h1_aug + xi1
        h2 = tf.sigmoid(tf.matmul(h1_tilde, W1))
        h2_aug = tf.concat([h2, e1], 1)
        xi2 = tf.random_normal(shape=tf.shape(h2_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h2_tilde = h2_aug + xi2
        y_p = tf.matmul(h2_tilde, W2)

        #Compute unperturbed output
        h2_0 = tf.sigmoid(tf.matmul(h1_aug, W1))
        h2_0_aug = tf.concat([h2_0, e1], 1)
        y_p_0 = tf.matmul(h2_0_aug, W2)

        self.trainable = [A, W1, W2, B1, B2]

        with tf.name_scope("loss"):
            #mean squared error
            self.loss_p = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss = tf.reduce_sum(tf.pow(y_p_0-self.y, 2))/2
            e = (y_p_0 - self.y)
            h1_prime_0 = tf.multiply(h1_aug, 1-h1_aug)[:,0:m]
            h2_prime_0 = tf.multiply(h2_0_aug, 1-h2_0_aug)[:,0:j]

            #Compute updates for W and A (based on B)
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            lmda2 = tf.matmul(e, tf.transpose(B2[0:j,:]))
            d2 = np.multiply(h2_prime_0, lmda2)
            grad_W1 = tf.matmul(tf.transpose(h1_aug), d2)
            lmda1 = tf.matmul(d2, tf.transpose(B1[0:m,:]))
            d1 = np.multiply(h1_prime_0, lmda1)
            grad_A = tf.matmul(tf.transpose(x_aug), d1)
            grad_B1 = tf.matmul(tf.matmul(B1, tf.transpose(d2)) - tf.transpose(xi1)*(self.loss_p - self.loss)/var_xi/var_xi, d2)
            grad_B2 = tf.matmul(tf.matmul(B2, tf.transpose(e)) - tf.transpose(xi2)*(self.loss_p - self.loss)/var_xi/var_xi, e)

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_A = A.assign(A - self.config.learning_rate*grad_A)

            #Train with SGD
            new_B1 = B1.assign(B1 - self.config.lmda_learning_rate*grad_B1)
            new_B2 = B2.assign(B2 - self.config.lmda_learning_rate*grad_B2)

            self.train_step = [new_W1, new_A, new_B1, new_W2, new_B2]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Save training metrics
            Bs = [B1, B2]
            Ws = [W1, W2]
            es = [d2, e]
            gradBs = [tf.norm(grad_B1), tf.norm(grad_B2)]
            self._set_training_metrics(Ws, Bs, es, gradBs)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _set_training_metrics(self, Ws, Bs, es, gradBs):
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
            delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
            alignment = tf.abs(tf_align(delta_fa, delta_bp))
            norm = tf.norm(Ws[idx] - Bs[idx], 2)
            sgn_cong = tf.reduce_mean((tf.sign(Ws[idx])*tf.sign(Bs[idx])+1)/2)
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)
            self.training_metric_tags.append('norm_W%d_B%d'%(idx+2, idx+2))
            self.training_metrics.append(norm)
            self.training_metric_tags.append('norm_gradB%d'%(idx+2))
            self.training_metrics.append(gradBs[idx])
            self.training_metric_tags.append('sign_cong%d'%(idx+2))
            self.training_metrics.append(sgn_cong)


class NPModel4_ExactLsq(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(NPModel4_ExactLsq, self).__init__(config)

        self.m = 50
        self.j = 20

        self.build_model()
        #Whether to save or not....
        #self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        #Set initial feedforward and feedback weights
        p = self.config.state_size[0]
        #m = 512
        #j = 200
        m = self.m
        j = self.j
        n = 10
        var_xi = self.config.var_xi
        gamma = self.config.gamma

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)

        V1 = tf.Variable(gamma*np.eye(j), dtype=tf.float32)
        V2 = tf.Variable(gamma*np.eye(n), dtype=tf.float32)
        S1 = tf.Variable(np.zeros((m+1,j)), dtype=tf.float32)
        S2 = tf.Variable(np.zeros((j+1,n)), dtype=tf.float32)

        #The exact least squares solution for synth grad estimation
        B1 = tf.matmul(S1, tf.matrix_inverse(V1))
        B2 = tf.matmul(S2, tf.matrix_inverse(V2))

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        e1 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = tf.sigmoid(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e1], 1)
        xi1 = tf.random_normal(shape=tf.shape(h1_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h1_tilde = h1_aug + xi1
        h2 = tf.sigmoid(tf.matmul(h1_tilde, W1))
        h2_aug = tf.concat([h2, e1], 1)
        xi2 = tf.random_normal(shape=tf.shape(h2_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h2_tilde = h2_aug + xi2
        y_p = tf.matmul(h2_tilde, W2)

        #Compute unperturbed output
        h2_0 = tf.sigmoid(tf.matmul(h1_aug, W1))
        h2_0_aug = tf.concat([h2_0, e1], 1)
        y_p_0 = tf.matmul(h2_0_aug, W2)

        self.trainable = [A, W1, W2, V1, V2, S1, S2]

        with tf.name_scope("loss"):
            #mean squared error
            self.loss_p = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss = tf.reduce_sum(tf.pow(y_p_0-self.y, 2))/2
            e = (y_p_0 - self.y)
            h1_prime_0 = tf.multiply(h1_aug, 1-h1_aug)[:,0:m]
            h2_prime_0 = tf.multiply(h2_0_aug, 1-h2_0_aug)[:,0:j]

            #Compute updates for W and A (based on B)
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            lmda2 = tf.matmul(e, tf.transpose(B2[0:j,:]))
            d2 = np.multiply(h2_prime_0, lmda2)
            grad_W1 = tf.matmul(tf.transpose(h1_aug), d2)
            lmda1 = tf.matmul(d2, tf.transpose(B1[0:m,:]))
            d1 = np.multiply(h1_prime_0, lmda1)
            grad_A = tf.matmul(tf.transpose(x_aug), d1)

            np_est1 = tf.transpose(xi1)*(self.loss_p - self.loss)/var_xi/var_xi
            np_est2 = tf.transpose(xi2)*(self.loss_p - self.loss)/var_xi/var_xi
            #grad_B1 = tf.matmul(tf.matmul(B1, tf.transpose(d2)) - np_est1, d2)
            #grad_B2 = tf.matmul(tf.matmul(B2, tf.transpose(e)) - np_est2, e)

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_A = A.assign(A - self.config.learning_rate*grad_A)

            #Train with exact least squares solution
            grad_V1 = tf.matmul(tf.transpose(d2), d2)
            grad_V2 = tf.matmul(tf.transpose(e), e)
            grad_S1 = tf.matmul(np_est1, d2)
            grad_S2 = tf.matmul(np_est2, e)

            #Update V and S
            new_V1 = V1.assign(V1 + grad_V1)                                            
            new_V2 = V2.assign(V2 + grad_V2)            
            new_S1 = S1.assign(S1 + grad_S1)
            new_S2 = S2.assign(S2 + grad_S2)            
            self.train_step = [new_W1, new_A, new_V1, new_S1, new_W2, new_V2, new_S2]
            self.train_step_warmup = [new_V1, new_S1, new_V2, new_S2]

            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Save training metrics
            Bs = [B1, B2]
            Ws = [W1, W2]
            es = [d2, e]
            #gradBs = [tf.norm(grad_B1), tf.norm(grad_B2)]
            self._set_training_metrics(Ws, Bs, es)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _set_training_metrics(self, Ws, Bs, es):
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
            delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
            alignment = tf.abs(tf_align(delta_fa, delta_bp))
            norm = tf.norm(Ws[idx] - Bs[idx])/tf.norm(Ws[idx])
            sgn_cong = tf.reduce_mean((tf.sign(Ws[idx])*tf.sign(Bs[idx])+1)/2)
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)
            self.training_metric_tags.append('norm_W%d_B%d'%(idx+2, idx+2))
            self.training_metrics.append(norm)
            self.training_metric_tags.append('sign_cong%d'%(idx+2))
            self.training_metrics.append(sgn_cong)

class NPModel4_ExactLsq_CorrectBatch(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(NPModel4_ExactLsq_CorrectBatch, self).__init__(config)

        self.m = 50
        self.j = 20

        self.build_model()
        #Whether to save or not....
        #self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        #Set initial feedforward and feedback weights
        p = self.config.state_size[0]
        #m = 512
        #j = 200
        m = self.m
        j = self.j
        n = 10
        var_xi = self.config.var_xi
        gamma = self.config.gamma

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)

        V1 = tf.Variable(gamma*np.eye(j), dtype=tf.float32)
        V2 = tf.Variable(gamma*np.eye(n), dtype=tf.float32)
        S1 = tf.Variable(np.zeros((m+1,j)), dtype=tf.float32)
        S2 = tf.Variable(np.zeros((j+1,n)), dtype=tf.float32)

        #The exact least squares solution for synth grad estimation
        B1 = tf.matmul(S1, tf.matrix_inverse(V1))
        B2 = tf.matmul(S2, tf.matrix_inverse(V2))

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        e1 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = tf.sigmoid(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e1], 1)
        xi1 = tf.random_normal(shape=tf.shape(h1_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h1_tilde = h1_aug + xi1
        h2 = tf.sigmoid(tf.matmul(h1_tilde, W1))
        h2_aug = tf.concat([h2, e1], 1)
        xi2 = tf.random_normal(shape=tf.shape(h2_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h2_tilde = h2_aug + xi2
        y_p = tf.matmul(h2_tilde, W2)

        #Compute unperturbed output
        h2_0 = tf.sigmoid(tf.matmul(h1_aug, W1))
        h2_0_aug = tf.concat([h2_0, e1], 1)
        y_p_0 = tf.matmul(h2_0_aug, W2)

        self.trainable = [A, W1, W2, V1, V2, S1, S2]

        with tf.name_scope("loss"):
            #mean squared error
            self.loss_p = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss = tf.reduce_sum(tf.pow(y_p_0-self.y, 2))/2
            loss_p = tf.reduce_sum(tf.pow(y_p-self.y, 2),1)/2
            loss = tf.reduce_sum(tf.pow(y_p_0-self.y, 2),1)/2
            e = (y_p_0 - self.y)
            h1_prime_0 = tf.multiply(h1_aug, 1-h1_aug)[:,0:m]
            h2_prime_0 = tf.multiply(h2_0_aug, 1-h2_0_aug)[:,0:j]

            #Compute updates for W and A (based on B)
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            lmda2 = tf.matmul(e, tf.transpose(B2[0:j,:]))
            d2 = np.multiply(h2_prime_0, lmda2)
            grad_W1 = tf.matmul(tf.transpose(h1_aug), d2)
            lmda1 = tf.matmul(d2, tf.transpose(B1[0:m,:]))
            d1 = np.multiply(h1_prime_0, lmda1)
            grad_A = tf.matmul(tf.transpose(x_aug), d1)

            np_est1 = tf.matmul(tf.transpose(xi1),tf.diag(loss_p - loss)/var_xi/var_xi)
            np_est2 = tf.matmul(tf.transpose(xi2),tf.diag(loss_p - loss)/var_xi/var_xi)
            #grad_B1 = tf.matmul(tf.matmul(B1, tf.transpose(d2)) - np_est1, d2)
            #grad_B2 = tf.matmul(tf.matmul(B2, tf.transpose(e)) - np_est2, e)

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_A = A.assign(A - self.config.learning_rate*grad_A)

            #Train with exact least squares solution
            grad_V1 = tf.matmul(tf.transpose(d2), d2)
            grad_V2 = tf.matmul(tf.transpose(e), e)
            grad_S1 = tf.matmul(np_est1, d2)
            grad_S2 = tf.matmul(np_est2, e)

            #Update V and S
            new_V1 = V1.assign(V1 + grad_V1)                                            
            new_V2 = V2.assign(V2 + grad_V2)            
            new_S1 = S1.assign(S1 + grad_S1)
            new_S2 = S2.assign(S2 + grad_S2)            
            self.train_step = [new_W1, new_A, new_V1, new_S1, new_W2, new_V2, new_S2]
            self.train_step_warmup = [new_V1, new_S1, new_V2, new_S2]

            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Save training metrics
            Bs = [B1, B2]
            Ws = [W1, W2]
            es = [d2, e]
            #gradBs = [tf.norm(grad_B1), tf.norm(grad_B2)]
            self._set_training_metrics(Ws, Bs, es)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _set_training_metrics(self, Ws, Bs, es):
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
            delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
            alignment = tf.abs(tf_align(delta_fa, delta_bp))
            norm = tf.norm(Ws[idx] - Bs[idx])/tf.norm(Ws[idx])
            sgn_cong = tf.reduce_mean((tf.sign(Ws[idx])*tf.sign(Bs[idx])+1)/2)
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)
            self.training_metric_tags.append('norm_W%d_B%d'%(idx+2, idx+2))
            self.training_metrics.append(norm)
            self.training_metric_tags.append('sign_cong%d'%(idx+2))
            self.training_metrics.append(sgn_cong)



#####################################################################################
#####################################################################################


class AENPModel5_ExactLsq_BPAuto(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(AENPModel5_ExactLsq_BPAuto, self).__init__(config)

        self.m = 200
        self.j = 2
        self.n = 200
        #200,2,200,784

        self.build_model()
        #Whether to save or not....
        #self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        #Set initial feedforward and feedback weights
        p = self.config.state_size[0]
        #m = 512
        #j = 200
        m = self.m                    # 200 (by default... could change)
        j = self.j                    # 2
        n = self.n                    # 200
        o = self.config.state_size[0] # 784

        #activation = tf.sigmoid
        activation = tf.nn.tanh

        var_xi = self.config.var_xi
        gamma = self.config.gamma

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha2a = np.sqrt(2.0/n)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)
        W3 = tf.Variable(rng.randn(n+1,o)*alpha2a, name="output_weights", dtype=tf.float32)

        V1 = tf.Variable(gamma*np.eye(j), dtype=tf.float32)
        V2 = tf.Variable(gamma*np.eye(n), dtype=tf.float32)
        V3 = tf.Variable(gamma*np.eye(o), dtype=tf.float32)
        S1 = tf.Variable(np.zeros((m+1,j)), dtype=tf.float32)
        S2 = tf.Variable(np.zeros((j+1,n)), dtype=tf.float32)
        S3 = tf.Variable(np.zeros((n+1,o)), dtype=tf.float32)

        #The exact least squares solution for synth grad estimation
        B1 = tf.matmul(S1, tf.matrix_inverse(V1))
        B2 = tf.matmul(S2, tf.matrix_inverse(V2))
        B3 = tf.matmul(S3, tf.matrix_inverse(V3))

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        e1 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = activation(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e1], 1)
        xi1 = tf.random_normal(shape=tf.shape(h1_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h1_tilde = h1_aug + xi1
        # Non linear middle layer
        #h2 = activation(tf.matmul(h1_tilde, W1))
        # Linear middle layer
        h2 = tf.matmul(h1_tilde, W1)
        h2_aug = tf.concat([h2, e1], 1)
        xi2 = tf.random_normal(shape=tf.shape(h2_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h2_tilde = h2_aug + xi2
        h3 = activation(tf.matmul(h2_tilde, W2))
        h3_aug = tf.concat([h3, e1], 1)
        xi3 = tf.random_normal(shape=tf.shape(h3_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h3_tilde = h3_aug + xi3
        #y_p = tf.matmul(h3_tilde, W3)
        y_p = tf.nn.relu(tf.matmul(h3_tilde, W3))

        #Compute unperturbed output
        #Non linear middle layer
        #h2_0 = activation(tf.matmul(h1_aug, W1))
        #Linear middle layer
        h2_0 = tf.matmul(h1_aug, W1)
        h2_0_aug = tf.concat([h2_0, e1], 1)
        h3_0 = activation(tf.matmul(h2_0_aug, W2))
        h3_0_aug = tf.concat([h3_0, e1], 1)
        #y_p_0 = tf.matmul(h3_0_aug, W3)
        y_p_0 = tf.nn.relu(tf.matmul(h3_0_aug, W3))
        self.y_p = y_p_0

        self.trainable = [A, W1, W2, W3, V1, V2, V3, S1, S2, S3]

        with tf.name_scope("loss"):
            #mean squared error
            self.loss_p = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss = tf.reduce_sum(tf.pow(y_p_0-self.y, 2))/2

            #primes for sigmoid non-linearity
            #h1_prime_0 = tf.multiply(h1_aug, 1-h1_aug)[:,0:m]
            #h2_prime_0 = tf.multiply(h2_0_aug, 1-h2_0_aug)[:,0:j]
            #h2_prime_0 = 1.0
            #h3_prime_0 = tf.multiply(h3_0_aug, 1-h3_0_aug)[:,0:n]
            #y_p_prime_0 = tf.maximum(0.0, tf.sign(y_p_0))

            #primes for tanh nonlinearity
            h1_prime_0 = 1.0 - tf.multiply(h1_aug, h1_aug)[:,0:m]
            #h2_prime_0 = tf.multiply(h2_0_aug, 1-h2_0_aug)[:,0:j]
            h2_prime_0 = 1.0
            h3_prime_0 = 1.0 - tf.multiply(h3_0_aug, h3_0_aug)[:,0:n]
            y_p_prime_0 = tf.maximum(0.0, tf.sign(y_p_0))

            e = tf.multiply((y_p_0 - self.y), y_p_prime_0)

            lmda3 = tf.matmul(e, tf.transpose(B3[0:n,:]))
            d3 = tf.multiply(h3_prime_0, lmda3)
            grad_W2 = tf.matmul(tf.transpose(h2_0_aug), d3)
            lmda2 = tf.matmul(d3, tf.transpose(B2[0:j,:]))
            d2 = tf.multiply(h2_prime_0, lmda2)
            grad_W1 = tf.matmul(tf.transpose(h1_aug), d2)
            lmda1 = tf.matmul(d2, tf.transpose(B1[0:m,:]))
            d1 = tf.multiply(h1_prime_0, lmda1)
            grad_A = tf.matmul(tf.transpose(x_aug), d1)

            #Compute updates for W and A (based on B)
            grad_W3 = tf.gradients(xs=W3, ys=self.loss)[0]
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            grad_A = tf.gradients(xs=A, ys=self.loss)[0]

            np_est1 = tf.transpose(xi1)*(self.loss_p - self.loss)/var_xi/var_xi
            np_est2 = tf.transpose(xi2)*(self.loss_p - self.loss)/var_xi/var_xi
            np_est3 = tf.transpose(xi3)*(self.loss_p - self.loss)/var_xi/var_xi

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W3 = W3.assign(W3 - self.config.learning_rate*grad_W3)
            new_A = A.assign(A - self.config.learning_rate*grad_A)

            #Train with exact least squares solution
            grad_V1 = tf.matmul(tf.transpose(d2), d2)
            grad_V2 = tf.matmul(tf.transpose(d3), d3)
            grad_V3 = tf.matmul(tf.transpose(e), e)
            grad_S1 = tf.matmul(np_est1, d2)
            grad_S2 = tf.matmul(np_est2, d3)
            grad_S3 = tf.matmul(np_est3, e)

            #Update V and S
            new_V1 = V1.assign(V1 + grad_V1)                                            
            new_V2 = V2.assign(V2 + grad_V2)            
            new_V3 = V3.assign(V3 + grad_V3)            
            new_S1 = S1.assign(S1 + grad_S1)
            new_S2 = S2.assign(S2 + grad_S2)            
            new_S3 = S3.assign(S3 + grad_S3)            

            #NP
            self.train_step = [new_W1, new_A, new_W2, new_W3]
            #FA
            #self.train_step = [new_W1, new_A, new_W2, new_W3]
            self.train_step_warmup = [new_V1, new_S1, new_V2, new_S2, new_V3, new_S3]

            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Save training metrics
            Bs = [B1, B2, B3]
            Ws = [W1, W2, W3]
            es = [d2, d3, e]
            #gradBs = [tf.norm(grad_B1), tf.norm(grad_B2)]
            self._set_training_metrics(Ws, Bs, es)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _set_training_metrics(self, Ws, Bs, es):
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
            delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
            alignment = tf.abs(tf_align(delta_fa, delta_bp))
            norm = tf.norm(Ws[idx] - Bs[idx])/tf.norm(Ws[idx])
            sgn_cong = tf.reduce_mean((tf.sign(Ws[idx])*tf.sign(Bs[idx])+1)/2)
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)
            self.training_metric_tags.append('norm_W%d_B%d'%(idx+2, idx+2))
            self.training_metrics.append(norm)
            self.training_metric_tags.append('sign_cong%d'%(idx+2))
            self.training_metrics.append(sgn_cong)


class AENPModel5_ExactLsq_FAAuto(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(AENPModel5_ExactLsq_FAAuto, self).__init__(config)

        self.m = 200
        self.j = 2
        self.n = 200
        #200,2,200,784

        self.build_model()
        #Whether to save or not....
        #self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        #Set initial feedforward and feedback weights
        p = self.config.state_size[0]
        #m = 512
        #j = 200
        m = self.m                    # 200 (by default... could change)
        j = self.j                    # 2
        n = self.n                    # 200
        o = self.config.state_size[0] # 784

        #activation = tf.sigmoid
        activation = tf.nn.tanh

        var_xi = self.config.var_xi
        gamma = self.config.gamma

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha2a = np.sqrt(2.0/n)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)
        W3 = tf.Variable(rng.randn(n+1,o)*alpha2a, name="output_weights", dtype=tf.float32)

        B1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)
        B3 = tf.Variable(rng.randn(n+1,o)*alpha2a, name="output_weights", dtype=tf.float32)

        V1 = tf.Variable(gamma*np.eye(j), dtype=tf.float32)
        V2 = tf.Variable(gamma*np.eye(n), dtype=tf.float32)
        V3 = tf.Variable(gamma*np.eye(o), dtype=tf.float32)
        S1 = tf.Variable(np.zeros((m+1,j)), dtype=tf.float32)
        S2 = tf.Variable(np.zeros((j+1,n)), dtype=tf.float32)
        S3 = tf.Variable(np.zeros((n+1,o)), dtype=tf.float32)


        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        e1 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = activation(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e1], 1)
        xi1 = tf.random_normal(shape=tf.shape(h1_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h1_tilde = h1_aug + xi1
        # Non linear middle layer
        #h2 = activation(tf.matmul(h1_tilde, W1))
        # Linear middle layer
        h2 = tf_matmul_r(h1_tilde, W1, B1)
        h2_aug = tf.concat([h2, e1], 1)
        xi2 = tf.random_normal(shape=tf.shape(h2_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h2_tilde = h2_aug + xi2
        h3 = activation(tf_matmul_r(h2_tilde, W2, B2))
        h3_aug = tf.concat([h3, e1], 1)
        xi3 = tf.random_normal(shape=tf.shape(h3_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h3_tilde = h3_aug + xi3
        #y_p = tf.matmul(h3_tilde, W3)
        y_p = tf.nn.relu(tf_matmul_r(h3_tilde, W3, B3))

        #Compute unperturbed output
        #Non linear middle layer
        #h2_0 = activation(tf.matmul(h1_aug, W1))
        #Linear middle layer
        h2_0 = tf_matmul_r(h1_aug, W1, B1)
        h2_0_aug = tf.concat([h2_0, e1], 1)
        h3_0 = activation(tf_matmul_r(h2_0_aug, W2, B2))
        h3_0_aug = tf.concat([h3_0, e1], 1)
        #y_p_0 = tf.matmul(h3_0_aug, W3)
        y_p_0 = tf.nn.relu(tf_matmul_r(h3_0_aug, W3, B3))
        self.y_p = y_p_0

        self.trainable = [A, W1, W2, W3, V1, V2, V3, S1, S2, S3]

        with tf.name_scope("loss"):
            #mean squared error
            self.loss_p = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss = tf.reduce_sum(tf.pow(y_p_0-self.y, 2))/2

            #primes for sigmoid non-linearity
            #h1_prime_0 = tf.multiply(h1_aug, 1-h1_aug)[:,0:m]
            #h2_prime_0 = tf.multiply(h2_0_aug, 1-h2_0_aug)[:,0:j]
            #h2_prime_0 = 1.0
            #h3_prime_0 = tf.multiply(h3_0_aug, 1-h3_0_aug)[:,0:n]
            #y_p_prime_0 = tf.maximum(0.0, tf.sign(y_p_0))

            #primes for tanh nonlinearity
            h1_prime_0 = 1.0 - tf.multiply(h1_aug, h1_aug)[:,0:m]
            #h2_prime_0 = tf.multiply(h2_0_aug, 1-h2_0_aug)[:,0:j]
            h2_prime_0 = 1.0
            h3_prime_0 = 1.0 - tf.multiply(h3_0_aug, h3_0_aug)[:,0:n]
            y_p_prime_0 = tf.maximum(0.0, tf.sign(y_p_0))

            e = tf.multiply((y_p_0 - self.y), y_p_prime_0)

            lmda3 = tf.matmul(e, tf.transpose(B3[0:n,:]))
            d3 = tf.multiply(h3_prime_0, lmda3)
            grad_W2 = tf.matmul(tf.transpose(h2_0_aug), d3)
            lmda2 = tf.matmul(d3, tf.transpose(B2[0:j,:]))
            d2 = tf.multiply(h2_prime_0, lmda2)
            grad_W1 = tf.matmul(tf.transpose(h1_aug), d2)
            lmda1 = tf.matmul(d2, tf.transpose(B1[0:m,:]))
            d1 = tf.multiply(h1_prime_0, lmda1)
            grad_A = tf.matmul(tf.transpose(x_aug), d1)

            #Compute updates for W and A (based on B)
            grad_W3 = tf.gradients(xs=W3, ys=self.loss)[0]
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            grad_A = tf.gradients(xs=A, ys=self.loss)[0]

            np_est1 = tf.transpose(xi1)*(self.loss_p - self.loss)/var_xi/var_xi
            np_est2 = tf.transpose(xi2)*(self.loss_p - self.loss)/var_xi/var_xi
            np_est3 = tf.transpose(xi3)*(self.loss_p - self.loss)/var_xi/var_xi

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W3 = W3.assign(W3 - self.config.learning_rate*grad_W3)
            new_A = A.assign(A - self.config.learning_rate*grad_A)

            #Train with exact least squares solution
            grad_V1 = tf.matmul(tf.transpose(d2), d2)
            grad_V2 = tf.matmul(tf.transpose(d3), d3)
            grad_V3 = tf.matmul(tf.transpose(e), e)
            grad_S1 = tf.matmul(np_est1, d2)
            grad_S2 = tf.matmul(np_est2, d3)
            grad_S3 = tf.matmul(np_est3, e)

            #Update V and S
            new_V1 = V1.assign(V1 + grad_V1)                                            
            new_V2 = V2.assign(V2 + grad_V2)            
            new_V3 = V3.assign(V3 + grad_V3)            
            new_S1 = S1.assign(S1 + grad_S1)
            new_S2 = S2.assign(S2 + grad_S2)            
            new_S3 = S3.assign(S3 + grad_S3)            

            #NP
            self.train_step = [new_W1, new_A, new_W2, new_W3]
            #FA
            #self.train_step = [new_W1, new_A, new_W2, new_W3]
            self.train_step_warmup = [new_V1, new_S1, new_V2, new_S2, new_V3, new_S3]

            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Save training metrics
            Bs = [B1, B2, B3]
            Ws = [W1, W2, W3]
            es = [d2, d3, e]
            #gradBs = [tf.norm(grad_B1), tf.norm(grad_B2)]
            self._set_training_metrics(Ws, Bs, es)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _set_training_metrics(self, Ws, Bs, es):
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
            delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
            alignment = tf.abs(tf_align(delta_fa, delta_bp))
            norm = tf.norm(Ws[idx] - Bs[idx])/tf.norm(Ws[idx])
            sgn_cong = tf.reduce_mean((tf.sign(Ws[idx])*tf.sign(Bs[idx])+1)/2)
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)
            self.training_metric_tags.append('norm_W%d_B%d'%(idx+2, idx+2))
            self.training_metrics.append(norm)
            self.training_metric_tags.append('sign_cong%d'%(idx+2))
            self.training_metrics.append(sgn_cong)

class AENPModel5(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(AENPModel5, self).__init__(config)

        self.m = 200
        self.j = 2
        self.n = 200
        #200,2,200,784

        self.build_model()
        #Whether to save or not....
        #self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        #Set initial feedforward and feedback weights
        p = self.config.state_size[0]
        #m = 512
        #j = 200
        m = self.m                    # 200 (by default... could change)
        j = self.j                    # 2
        n = self.n                    # 200
        o = self.config.state_size[0] # 784

        #activation = tf.sigmoid
        activation = tf.nn.tanh
        identity = lambda x: x

        var_xi = self.config.var_xi
        gamma = self.config.gamma

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha2a = np.sqrt(2.0/n)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)
        W3 = tf.Variable(rng.randn(n+1,o)*alpha2a, name="output_weights", dtype=tf.float32)

        #The exact least squares solution for synth grad estimation
        B1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)
        B3 = tf.Variable(rng.randn(n+1,o)*alpha2a, name="output_weights", dtype=tf.float32)

        # first fully connected layer with 50 neurons using tanh activation
        n1, A, _ = fa_layer(self.x, 28*28, 50, self.config)
        l1 = tf.nn.tanh(n1)
        # third fully connected layer with 2 neurons
        n2, W1, B1 = fa_layer(l1, 50, 2, self.config)
        l2 = n2
        # fourth fully connected layer with 50 neurons and tanh activation
        n3, W2, B2 = fa_layer(l2, 2, 50, self.config)
        l3 = tf.nn.tanh(n3)
        n4, W3, B3 = fa_layer(l3, 50, 28*28, self.config)
        y_p_0 = tf.nn.relu(n4)

        #Add noise to response
        # first fully connected layer with 50 neurons using tanh activation
        l1_p, n1_p, xi1 = fc_layer_noise_act(self.x, A, var_xi, self.config, activation)
        # second fully connected layer with 50 neurons using tanh activation
        l2_p, n2_p, xi2 = fc_layer_noise_act(l1_p, W1, var_xi, self.config, identity)
        # third fully connected layer with 2 neurons
        l3_p, n3_p, xi3 = fc_layer_noise_act(l2_p, W2, var_xi, self.config, activation)
        # fourth fully connected layer with 50 neurons and tanh activation
        y_p, n4_p, xi4 = fc_layer_noise_act(l3_p, W3, 0, self.config, tf.nn.relu)

        self.y_p = y_p_0
        self.trainable = [A, W1, W2, W3, B1, B2, B3]

        with tf.name_scope("loss"):
            #mean squared error
            self.loss_p = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss = tf.reduce_sum(tf.pow(y_p_0-self.y, 2))/2

            loss_p = tf.reduce_sum(tf.pow(y_p-self.y, 2),1)/2
            loss = tf.reduce_sum(tf.pow(y_p_0-self.y, 2),1)/2

            #Compute updates for W and A (based on B)
            grad_W3 = tf.gradients(xs=W3, ys=self.loss)[0]
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            grad_A = tf.gradients(xs=A, ys=self.loss)[0]

            d2 = tf.gradients(xs=n2, ys=loss)[0]
            d3 = tf.gradients(xs=n3, ys=loss)[0]
            d4 = tf.gradients(xs=n4, ys=loss)[0]

            #grad_B1 = tf.matmul(tf.transpose(tf.gradients(xs=l1, ys=self.loss)[0] - xi1*(self.loss_p - self.loss)/var_xi/var_xi),d2)
            #grad_B2 = tf.matmul(tf.transpose(tf.gradients(xs=l2, ys=self.loss)[0] - xi2*(self.loss_p - self.loss)/var_xi/var_xi),d3)
            #grad_B3 = tf.matmul(tf.transpose(tf.gradients(xs=l3, ys=self.loss)[0] - xi3*(self.loss_p - self.loss)/var_xi/var_xi),d4)

            grad_B1 = tf.matmul(tf.transpose(tf.gradients(xs=l1, ys=loss)[0] - tf.matmul(tf.diag(loss_p - loss)/var_xi/var_xi,xi1)),d2)
            grad_B2 = tf.matmul(tf.transpose(tf.gradients(xs=l2, ys=loss)[0] - tf.matmul(tf.diag(loss_p - loss)/var_xi/var_xi,xi2)),d3)
            grad_B3 = tf.matmul(tf.transpose(tf.gradients(xs=l3, ys=loss)[0] - tf.matmul(tf.diag(loss_p - loss)/var_xi/var_xi,xi3)),d4)

            grad_B1 = tf.concat([grad_B1, tf.zeros([1, 2], tf.float32)], 0)
            grad_B2 = tf.concat([grad_B2, tf.zeros([1, 50], tf.float32)], 0)
            grad_B3 = tf.concat([grad_B3, tf.zeros([1, 784], tf.float32)], 0)

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W3 = W3.assign(W3 - self.config.learning_rate*grad_W3)
            new_A = A.assign(A - self.config.learning_rate*grad_A)

            new_B1 = B1.assign(B1 - self.config.lmda_learning_rate*grad_B1)
            new_B2 = B2.assign(B2 - self.config.lmda_learning_rate*grad_B2)
            new_B3 = B3.assign(B3 - self.config.lmda_learning_rate*grad_B3)

            self.train_step = [new_W1, new_A, new_W2, new_W3, new_B1, new_B2, new_B3]
            self.train_step_warmup = [new_B1, new_B2, new_B3]

            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Save training metrics
            Bs = [B1, B2, B3]
            Ws = [W1, W2, W3]
            es = [d2, d3, d4]
            self._set_training_metrics(Ws, Bs, es)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _set_training_metrics(self, Ws, Bs, es):
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
            delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
            alignment = tf.abs(tf_align(delta_fa, delta_bp))
            norm = tf.norm(Ws[idx] - Bs[idx])/tf.norm(Ws[idx])
            sgn_cong = tf.reduce_mean((tf.sign(Ws[idx])*tf.sign(Bs[idx])+1)/2)
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)
            self.training_metric_tags.append('norm_W%d_B%d'%(idx+2, idx+2))
            self.training_metrics.append(norm)
            self.training_metric_tags.append('sign_cong%d'%(idx+2))
            self.training_metrics.append(sgn_cong)


###################################################################################
###################################################################################
