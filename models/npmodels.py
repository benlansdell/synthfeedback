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

def fc_layer_noise(prev, W, var_xi, config):
    e = tf.ones([config.batch_size, 1], tf.float32)
    prev_aug = tf.concat([prev, e], 1)
    n = tf.matmul(prev_aug, W)
    xi = tf.random_normal(shape=tf.shape(n), mean=0.0, stddev=var_xi, dtype=tf.float32)
    return n + xi, xi

class NPModel(BaseModel):
    def __init__(self, config):
        super(NPModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # set initial feedforward and feedback weights
        m = 512
        n = 10
        p = self.config.state_size[0]
        var_xi = self.config.var_xi

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
        W = tf.Variable(rng.randn(m+1,n)*alpha1, name="output_weights", dtype=tf.float32)
        B = tf.Variable(rng.randn(m+1,n)*alpha2, name="feedback_weights", dtype=tf.float32)

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        e1 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h = tf.sigmoid(tf.matmul(x_aug, A))
        #Make some noise
        h_aug = tf.concat([h, e1], 1)
        xi = tf.random_normal(shape=tf.shape(h_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h_tilde = h_aug + xi
        #Add noise to hidden layer
        y_p = tf.matmul(h_tilde, W)
        y_p_0 = tf.matmul(h_aug, W)

        self.trainable = [A, W, B]

        with tf.name_scope("loss"):
            #mean squared error
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss_0 = tf.reduce_sum(tf.pow(y_p_0-self.y, 2))/2
            e = (y_p - self.y)
            h_prime = tf.multiply(h_tilde, 1-h_tilde)[:,0:m]

            #Feedback data for saving
            #Only take first item in epoch
            delta_bp = tf.matmul(e, tf.transpose(W[0:m,:]))[0,:]
            delta_fa = tf.matmul(e, tf.transpose(B[0:m,:]))[0,:]
            norm_W = tf.norm(W)
            norm_B = tf.norm(B)
            error_FA = tf.norm(delta_bp - delta_fa)
            alignment = tf.reduce_sum(tf.multiply(delta_fa,delta_bp))/tf.norm(delta_fa)/tf.norm(delta_bp)
            eigs = tf_eigvals(tf.matmul(tf.transpose(B), W))

            #Compute updates for W and A (based on B)
            lmda = tf.matmul(e, tf.transpose(B[0:m,:]))
            grad_W = tf.gradients(xs=W, ys=self.loss)[0]
            grad_A = tf.matmul(tf.transpose(x_aug), tf.multiply(h_prime, lmda))
            grad_B = tf.matmul(tf.matmul(B, tf.transpose(e)) - tf.transpose(xi)*(self.loss - self.loss_0)/var_xi, e)

            new_W = W.assign(W - self.config.learning_rate*grad_W)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            new_B = B.assign(B - self.config.lmda_learning_rate
                             *grad_B)            
            self.train_step = [new_W, new_A, new_B]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Also need to add eigenvector stuff
            self.training_metrics = [alignment, norm_W, norm_B, error_FA, eigs[0]]

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

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
        m = 512
        j = 200
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
        B1 = tf.Variable(rng.randn(m+1,j)*alpha3, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,n)*alpha3, name="feedback_weights2", dtype=tf.float32)

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        e1 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = tf.sigmoid(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e1], 1)
        xi1 = tf.random_normal(shape=tf.shape(h1_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h1_tilde = h1_aug + xi1
        h2 = tf.sigmoid(tf_matmul_r(h1_tilde, W1, B1))
        h2_aug = tf.concat([h2, e1], 1)
        xi2 = tf.random_normal(shape=tf.shape(h2_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h2_tilde = h2_aug + xi2
        y_p = tf_matmul_r(h2_tilde, W2, B2)

        #Compute unperturbed output
        h2_0 = tf.sigmoid(tf_matmul_r(h1_aug, W1, B1))
        h2_0_aug = tf.concat([h2_0, e1], 1)
        y_p_0 = tf_matmul_r(h2_0_aug, W2, B2)

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
            grad_B1 = tf.matmul(tf.matmul(B1, tf.transpose(d2)) - tf.transpose(xi1)*(self.loss_p - self.loss)/var_xi, d2)
            grad_B2 = tf.matmul(tf.matmul(B2, tf.transpose(e)) - tf.transpose(xi2)*(self.loss_p - self.loss)/var_xi, e)

            #Feedback data for saving
            #Only take first item in epoch
            delta_bp2 = tf.matmul(e, tf.transpose(W2[0:m,:]))[0,:]
            delta_fa2 = tf.matmul(e, tf.transpose(B2[0:m,:]))[0,:]
            delta_bp1 = tf.matmul(d2, tf.transpose(W1[0:m,:]))[0,:]
            delta_fa1 = tf.matmul(d2, tf.transpose(B1[0:m,:]))[0,:]
            norm_W1 = tf.norm(W1)
            norm_W2 = tf.norm(W2)
            norm_B1 = tf.norm(B1)
            norm_B2 = tf.norm(B2)
            error_FA1 = tf.norm(delta_bp1 - delta_fa1)
            error_FA2 = tf.norm(delta_bp2 - delta_fa2)
            alignment1 = tf.reduce_sum(tf.multiply(delta_fa1,delta_bp1))/tf.norm(delta_fa1)/tf.norm(delta_bp1)
            alignment2 = tf.reduce_sum(tf.multiply(delta_fa2,delta_bp2))/tf.norm(delta_fa2)/tf.norm(delta_bp2)
            eigs1 = tf_eigvals(tf.matmul(tf.transpose(B1), W1))
            eigs2 = tf_eigvals(tf.matmul(tf.transpose(B2), W2))

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_A = A.assign(A - self.config.learning_rate*grad_A)
            new_B1 = B1.assign(B1 - self.config.lmda_learning_rate*grad_B1)
            new_B2 = B2.assign(B2 - self.config.lmda_learning_rate*grad_B2)
            self.train_step = [new_W1, new_A, new_B1, new_W2, new_B2]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Also need to add eigenvector stuff
            #self.training_metrics = [alignment, norm_W, norm_B, error_FA, eigs[0]]
            self.training_metrics = []
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

class DirectNPModel4(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(DirectNPModel4, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # set initial feedforward and feedback weights
        p = self.config.state_size[0]
        m = 512
        j = 200
        n = 10
        var_xi = self.config.var_xi

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0,  name="hidden_weights", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)
        B1 = tf.Variable(rng.randn(m+1,n)*alpha3, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,n)*alpha3, name="feedback_weights2", dtype=tf.float32)

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        e1 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = tf.sigmoid(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e1], 1)
        xi1 = tf.random_normal(shape=tf.shape(h1_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h1_tilde = h1_aug + xi1

        h2 = tf.sigmoid(tf_matmul_r(h1_tilde, W1, B1))
        h2_aug = tf.concat([h2, e1], 1)
        xi2 = tf.random_normal(shape=tf.shape(h2_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h2_tilde = h2_aug + xi2

        y_p = tf_matmul_r(h2_tilde, W2, B2)

        #Compute unperturbed output
        h2_0 = tf.sigmoid(tf_matmul_r(h1_aug, W1, B1))
        h2_0_aug = tf.concat([h2, e1], 1)
        y_p_0 = tf.matmul(h2_0_aug, W2)

        self.trainable = [A, W1, W2, B1, B2]

        with tf.name_scope("loss"):
            #mean squared error
            self.loss_p = tf.reduce_sum(tf.pow(y_p - self.y, 2))/2
            self.loss = tf.reduce_sum(tf.pow(y_p_0 - self.y, 2))/2
            e = (y_p - self.y)
            h1_prime = tf.multiply(h1_aug, 1-h1_aug)[:,0:m]
            h2_prime = tf.multiply(h2_aug, 1-h2_aug)[:,0:j]

            #Compute updates for W and A (based on B)
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            lmda2 = tf.matmul(e, tf.transpose(B2[0:j,:]))
            d2 = np.multiply(h2_prime, lmda2)

            grad_W1 = tf.matmul(tf.transpose(h1_aug), d2)
            lmda1 = tf.matmul(e, tf.transpose(B1[0:m,:]))
            d1 = np.multiply(h1_prime, lmda1)

            grad_A = tf.matmul(tf.transpose(x_aug), d1)
            grad_B1 = tf.matmul(tf.matmul(B1, tf.transpose(e)) - tf.transpose(xi1)*(self.loss_p - self.loss)/var_xi, e)
            grad_B2 = tf.matmul(tf.matmul(B2, tf.transpose(e)) - tf.transpose(xi2)*(self.loss_p - self.loss)/var_xi, e)

            #Feedback data for saving
            #Only take first item in epoch
            delta_bp2 = tf.matmul(e, tf.transpose(W2[0:m,:]))[0,:]
            delta_fa2 = tf.matmul(e, tf.transpose(B2[0:m,:]))[0,:]
            delta_bp1 = tf.matmul(d2, tf.transpose(W1[0:m,:]))[0,:]
            delta_fa1 = tf.matmul(e, tf.transpose(B1[0:m,:]))[0,:]
            norm_W1 = tf.norm(W1)
            norm_W2 = tf.norm(W2)
            norm_B1 = tf.norm(B1)
            norm_B2 = tf.norm(B2)
            error_FA1 = tf.norm(delta_bp1 - delta_fa1)
            error_FA2 = tf.norm(delta_bp2 - delta_fa2)
            alignment1 = tf.reduce_sum(tf.multiply(delta_fa1,delta_bp1))/tf.norm(delta_fa1)/tf.norm(delta_bp1)
            alignment2 = tf.reduce_sum(tf.multiply(delta_fa2,delta_bp2))/tf.norm(delta_fa2)/tf.norm(delta_bp2)
            eigs1 = tf_eigvals(tf.matmul(tf.transpose(B1), W1))
            eigs2 = tf_eigvals(tf.matmul(tf.transpose(B2), W2))

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_A = A.assign(A - self.config.learning_rate*grad_A)
            new_B1 = B1.assign(B1 - self.config.lmda_learning_rate*grad_B1)
            new_B2 = B2.assign(B2 - self.config.lmda_learning_rate*grad_B2)
            self.train_step = [new_W1, new_A, new_B1, new_W2, new_B2]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Also need to add eigenvector stuff
            #self.training_metrics = [alignment, norm_W, norm_B, error_FA, eigs[0]]
            self.training_metrics = []
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

#####################################################################################
#####################################################################################

class AENPModel(BaseModel):
    #Auto encoder NP model
    def __init__(self, config):
        super(AENPModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        var_xi = self.config.var_xi

        # first fully connected layer with 50 neurons using tanh activation
        n1, W1, B1 = fa_layer(self.x, 28*28, 50, self.config)
        l1 = tf.nn.tanh(n1)
        # second fully connected layer with 50 neurons using tanh activation
        n2, W2, B2 = fa_layer(l1, 50, 50, self.config)
        l2 = tf.nn.tanh(n2)
        # third fully connected layer with 2 neurons
        n3, W3, B3 = fa_layer(l2, 50, 2, self.config)
        l3 = n3
        # fourth fully connected layer with 50 neurons and tanh activation
        n4, W4, B4 = fa_layer(l3, 2, 50, self.config)
        l4 = tf.nn.tanh(n4)
        # fifth fully connected layer with 50 neurons and tanh activation
        n5, W5, B5 = fa_layer(l4, 50, 50, self.config)
        l5 = tf.nn.tanh(n5)
        y_p, W6, B6 = fa_layer(l5, 50, 28*28, self.config)

        #Add noise to response
        # first fully connected layer with 50 neurons using tanh activation
        n1_p, xi1 = fc_layer_noise(self.x, W1, var_xi, self.config)
        l1_p = tf.nn.tanh(n1_p)
        # second fully connected layer with 50 neurons using tanh activation
        n2_p, xi2 = fc_layer_noise(l1_p, W2, var_xi, self.config)
        l2_p = tf.nn.tanh(n2_p)
        # third fully connected layer with 2 neurons
        n3_p, xi3 = fc_layer_noise(l2_p, W3, var_xi, self.config)
        l3_p = n3_p
        # fourth fully connected layer with 50 neurons and tanh activation
        n4_p, xi4 = fc_layer_noise(l3_p, W4, var_xi, self.config)
        l4_p = tf.nn.tanh(n4_p)
        # fifth fully connected layer with 50 neurons and tanh activation
        n5_p, xi5 = fc_layer_noise(l4_p, W5, var_xi, self.config)
        l5_p = tf.nn.tanh(n5_p)
        y_pp, xi6 = fc_layer_noise(l5_p, W6, var_xi, self.config)

        self.y_p = y_p

        with tf.name_scope("loss"):
            #mean squared error
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss_p = tf.reduce_sum(tf.pow(y_pp-self.y, 2))/2

            #Updates to W matrices
            grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            grad_W3 = tf.gradients(xs=W3, ys=self.loss)[0]
            grad_W4 = tf.gradients(xs=W4, ys=self.loss)[0]
            grad_W5 = tf.gradients(xs=W5, ys=self.loss)[0]
            grad_W6 = tf.gradients(xs=W6, ys=self.loss)[0]

            #d variables
            d2 = tf.gradients(xs=n2, ys=self.loss)[0]
            d3 = tf.gradients(xs=n3, ys=self.loss)[0]
            d4 = tf.gradients(xs=n4, ys=self.loss)[0]
            d5 = tf.gradients(xs=n5, ys=self.loss)[0]
            d6 = tf.gradients(xs=y_p, ys=self.loss)[0]

            #Updates to B matrices. Here is the node perturbation updates
            #grad_B2 = tf.matmul((tf.gradients(xs=l1_aug, ys=self.loss)[0] - tf.transpose(xi1)*(self.loss_p - self.loss)/var_xi),d2)
            #grad_B3 = tf.matmul((tf.gradients(xs=l2_aug, ys=self.loss)[0] - tf.transpose(xi2)*(self.loss_p - self.loss)/var_xi),d3)
            #grad_B4 = tf.matmul((tf.gradients(xs=l3_aug, ys=self.loss)[0] - tf.transpose(xi3)*(self.loss_p - self.loss)/var_xi),d4)
            #grad_B5 = tf.matmul((tf.gradients(xs=l4_aug, ys=self.loss)[0] - tf.transpose(xi4)*(self.loss_p - self.loss)/var_xi),d5)
            #grad_B6 = tf.matmul((tf.gradients(xs=l5_aug, ys=self.loss)[0] - tf.transpose(xi5)*(self.loss_p - self.loss)/var_xi),d6)

            grad_B2 = tf.matmul(tf.transpose(tf.gradients(xs=l1, ys=self.loss)[0] - xi1*(self.loss_p - self.loss)/var_xi),d2)
            grad_B3 = tf.matmul(tf.transpose(tf.gradients(xs=l2, ys=self.loss)[0] - xi2*(self.loss_p - self.loss)/var_xi),d3)
            grad_B4 = tf.matmul(tf.transpose(tf.gradients(xs=l3, ys=self.loss)[0] - xi3*(self.loss_p - self.loss)/var_xi),d4)
            grad_B5 = tf.matmul(tf.transpose(tf.gradients(xs=l4, ys=self.loss)[0] - xi4*(self.loss_p - self.loss)/var_xi),d5)
            grad_B6 = tf.matmul(tf.transpose(tf.gradients(xs=l5, ys=self.loss)[0] - xi5*(self.loss_p - self.loss)/var_xi),d6)

            #Add zeros to extra row
            grad_B2 = tf.concat([grad_B2, tf.zeros([1, 50], tf.float32)], 0)
            grad_B3 = tf.concat([grad_B3, tf.zeros([1, 2], tf.float32)], 0)
            grad_B4 = tf.concat([grad_B4, tf.zeros([1, 50], tf.float32)], 0)
            grad_B5 = tf.concat([grad_B5, tf.zeros([1, 50], tf.float32)], 0)
            grad_B6 = tf.concat([grad_B6, tf.zeros([1, 784], tf.float32)], 0)

            #Also need to add eigenvector stuff
            #self.training_metrics = [alignment, norm_W, norm_B, error_FA]

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W3 = W3.assign(W3 - self.config.learning_rate*grad_W3)
            new_W4 = W4.assign(W4 - self.config.learning_rate*grad_W4)
            new_W5 = W5.assign(W5 - self.config.learning_rate*grad_W5)
            new_W6 = W6.assign(W6 - self.config.learning_rate*grad_W6)

            new_B2 = B2.assign(B2 - self.config.lmda_learning_rate*grad_B2)
            new_B3 = B3.assign(B3 - self.config.lmda_learning_rate*grad_B3)
            new_B4 = B4.assign(B4 - self.config.lmda_learning_rate*grad_B4)
            new_B5 = B5.assign(B5 - self.config.lmda_learning_rate*grad_B5)
            new_B6 = B6.assign(B6 - self.config.lmda_learning_rate*grad_B6)
            self.train_step = [new_W1, new_W2, new_W3, new_W4, new_W5, new_W6, 
                                new_B2, new_B3, new_B4, new_B5, new_B6]

            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

#####################################################################################
#####################################################################################

class AEDFANPModel(BaseModel):
    #Auto encoder NP model
    def __init__(self, config):
        super(AEDFANPModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        alpha = 1.0

        #Define random feedback matrices
        B2 = tf.Variable(rng.randn(50,28*28)*alpha, name="feedback_weights", dtype=tf.float32)
        B3 = tf.Variable(rng.randn(50,28*28)*alpha, name="feedback_weights", dtype=tf.float32)
        B4 = tf.Variable(rng.randn(2,28*28)*alpha, name="feedback_weights", dtype=tf.float32)
        B5 = tf.Variable(rng.randn(50,28*28)*alpha, name="feedback_weights", dtype=tf.float32)
        B6 = tf.Variable(rng.randn(50,28*28)*alpha, name="feedback_weights", dtype=tf.float32)

        var_xi = self.config.var_xi

        # first fully connected layer with 50 neurons using tanh activation
        n1, W1, x_aug = fc_layer(self.x, 28*28, 50, self.config)
        l1 = tf.nn.tanh(n1)
        # second fully connected layer with 50 neurons using tanh activation
        n2, W2, l1_aug = fc_layer(l1, 50, 50, self.config)
        l2 = tf.nn.tanh(n2)
        # third fully connected layer with 2 neurons
        n3, W3, l2_aug = fc_layer(l2, 50, 2, self.config)
        l3 = n3
        # fourth fully connected layer with 50 neurons and tanh activation
        n4, W4, l3_aug = fc_layer(l3, 2, 50, self.config)
        l4 = tf.nn.tanh(n4)
        # fifth fully connected layer with 50 neurons and tanh activation
        n5, W5, l4_aug = fc_layer(l4, 50, 50, self.config)
        l5 = tf.nn.tanh(n5)
        y_p, W6, l5_aug = fc_layer(l5, 50, 28*28, self.config)

        #Add noise to response
        # first fully connected layer with 50 neurons using tanh activation
        n1_p, xi1 = fc_layer_noise(self.x, W1, var_xi, self.config)
        l1_p = tf.nn.tanh(n1_p)
        # second fully connected layer with 50 neurons using tanh activation
        n2_p, xi2 = fc_layer_noise(l1_p, W2, var_xi, self.config)
        l2_p = tf.nn.tanh(n2_p)
        # third fully connected layer with 2 neurons
        n3_p, xi3 = fc_layer_noise(l2_p, W3, var_xi, self.config)
        l3_p = n3_p
        # fourth fully connected layer with 50 neurons and tanh activation
        n4_p, xi4 = fc_layer_noise(l3_p, W4, var_xi, self.config)
        l4_p = tf.nn.tanh(n4_p)
        # fifth fully connected layer with 50 neurons and tanh activation
        n5_p, xi5 = fc_layer_noise(l4_p, W5, var_xi, self.config)
        l5_p = tf.nn.tanh(n5_p)
        y_pp, xi6 = fc_layer_noise(l5_p, W6, var_xi, self.config)

        self.y_p = y_p

        with tf.name_scope("loss"):
            #mean squared error
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss_p = tf.reduce_sum(tf.pow(y_pp-self.y, 2))/2

            #Direct feedback alignment. 
            e = y_p - self.y

            l1_sigma = 1 - tf.multiply(l1, l1)
            l2_sigma = 1 - tf.multiply(l2, l2)
            l3_sigma = 1
            l4_sigma = 1 - tf.multiply(l4, l4)
            l5_sigma = 1 - tf.multiply(l5, l5)

            #Updates to W matrices (DFA)
            grad_W1 = tf.matmul(tf.transpose(x_aug), tf.multiply(tf.matmul(e, tf.transpose(B2)), l1_sigma))
            grad_W2 = tf.matmul(tf.transpose(l1_aug),tf.multiply(tf.matmul(e, tf.transpose(B3)), l2_sigma))
            grad_W3 = tf.matmul(tf.transpose(l2_aug),tf.multiply(tf.matmul(e, tf.transpose(B4)), l3_sigma))
            grad_W4 = tf.matmul(tf.transpose(l3_aug),tf.multiply(tf.matmul(e, tf.transpose(B5)), l4_sigma))
            grad_W5 = tf.matmul(tf.transpose(l4_aug),tf.multiply(tf.matmul(e, tf.transpose(B6)), l5_sigma))
            grad_W6 = tf.matmul(tf.transpose(l5_aug),e)

            d2 = tf.gradients(xs=n2, ys=self.loss)[0]
            d3 = tf.gradients(xs=n3, ys=self.loss)[0]
            d4 = tf.gradients(xs=n4, ys=self.loss)[0]
            d5 = tf.gradients(xs=n5, ys=self.loss)[0]
            d6 = tf.gradients(xs=y_p, ys=self.loss)[0]

            dl1 = tf.gradients(xs=l1, ys=self.loss)[0]
            dl2 = tf.gradients(xs=l2, ys=self.loss)[0]
            dl3 = tf.gradients(xs=l3, ys=self.loss)[0]
            dl4 = tf.gradients(xs=l4, ys=self.loss)[0]
            dl5 = tf.gradients(xs=l5, ys=self.loss)[0]

            #Updates to B matrices (NP)
            grad_B2 = tf.matmul(tf.transpose(tf.matmul(e, tf.transpose(B2)) - xi1*(self.loss_p - self.loss)/var_xi),e)
            grad_B3 = tf.matmul(tf.transpose(tf.matmul(e, tf.transpose(B3)) - xi2*(self.loss_p - self.loss)/var_xi),e)
            grad_B4 = tf.matmul(tf.transpose(tf.matmul(e, tf.transpose(B4)) - xi3*(self.loss_p - self.loss)/var_xi),e)
            grad_B5 = tf.matmul(tf.transpose(tf.matmul(e, tf.transpose(B5)) - xi4*(self.loss_p - self.loss)/var_xi),e)
            grad_B6 = tf.matmul(tf.transpose(tf.matmul(e, tf.transpose(B6)) - xi5*(self.loss_p - self.loss)/var_xi),e)

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W3 = W3.assign(W3 - self.config.learning_rate*grad_W3)
            new_W4 = W4.assign(W4 - self.config.learning_rate*grad_W4)
            new_W5 = W5.assign(W5 - self.config.learning_rate*grad_W5)
            new_W6 = W6.assign(W6 - self.config.learning_rate*grad_W6)

            new_B2 = B2.assign(B2 - self.config.lmda_learning_rate*grad_B2)
            new_B3 = B3.assign(B3 - self.config.lmda_learning_rate*grad_B3)
            new_B4 = B4.assign(B4 - self.config.lmda_learning_rate*grad_B4)
            new_B5 = B5.assign(B5 - self.config.lmda_learning_rate*grad_B5)
            new_B6 = B6.assign(B6 - self.config.lmda_learning_rate*grad_B6)
            self.train_step = [new_W1, new_W2, new_W3, new_W4, new_W5, new_W6, 
                                new_B2, new_B3, new_B4, new_B5, new_B6]

            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Save training metrics
            Bs = [B2, B3, B4, B5, B6]
            Ws = [W2, W3, W4, W5, W6]
            dls = [dl1, dl2, dl3, dl4, dl5]
            ls = [l1, l2, l3, l4, l5]
            self._set_training_metrics(e, Bs, Ws, dls, ls, self.config.learning_rate)

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


        #Alignment of l_i with top k singular values of W_{i+1}

        #Alignment of B_ie with top k eigenvalues of B_i^TW_i 

        #Alignment of B_ie with top k eigenvalues of (1 + \eta B_i^TW_i) 

        #eigs1 = tf_eigvals(tf.matmul(tf.transpose(B1), W1))
        #eigs2 = tf_eigvals(tf.matmul(tf.transpose(B2), W2))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


