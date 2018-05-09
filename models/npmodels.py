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

def fc_layer(prev, input_size, output_size):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(prev, W) + b

def fa_layer(prev, input_size, output_size):
    W = weight_variable([input_size, output_size])
    B = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf_matmul_r(prev, W, B) + b

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
            new_B = B.assign(B - self.config.lmda_learning_rate*grad_B)            
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
            self.loss_p = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss = tf.reduce_sum(tf.pow(y_p_0-self.y, 2))/2
            e = (y_p - self.y)
            h1_prime = tf.multiply(h1_aug, 1-h1_aug)[:,0:m]
            h2_prime = tf.multiply(h2_aug, 1-h2_aug)[:,0:j]

            #Compute updates for W and A (based on B)
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            lmda2 = tf.matmul(e, tf.transpose(B2[0:j,:]))
            d2 = np.multiply(h2_prime, lmda2)
            grad_W1 = tf.matmul(tf.transpose(h1_aug), d2)
            lmda1 = tf.matmul(d2, tf.transpose(B1[0:m,:]))
            d1 = np.multiply(h1_prime, lmda1)
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
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)
        B1 = tf.Variable(rng.randn(m+1,n)*alpha1, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights2", dtype=tf.float32)

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
            self.loss_p = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss = tf.reduce_sum(tf.pow(y_p_0-self.y, 2))/2
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

class FAModelLinear(BaseModel):
    def __init__(self, config):
        super(FAModelLinear, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # set initial feedforward and feedback weights
        m = 20
        n = 10
        p = self.config.state_size[0]

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
        h = tf.matmul(x_aug, A)
        h_aug = tf.concat([h, e1], 1)
        y_p = tf_matmul_r(h_aug, W, B)

        with tf.name_scope("loss"):
            #mean squared error
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            grad_W = tf.gradients(xs=W, ys=self.loss)[0]
            grad_A = tf.gradients(xs=A, ys=self.loss)[0]

            #Feedback data for saving
            e = (y_p - self.y)
            delta_bp = tf.matmul(e, tf.transpose(W[0:m,:]))[0,:]
            delta_fa = tf.matmul(e, tf.transpose(B))[0,:]
            norm_W = tf.norm(W)
            norm_B = tf.norm(B)
            error_FA = tf.norm(delta_bp - delta_fa)
            alignment = tf.reduce_sum(tf.multiply(delta_fa,delta_bp))/tf.norm(delta_fa)/tf.norm(delta_bp)

            #Also need to add eigenvector stuff
            self.training_metrics = [alignment, norm_W, norm_B, error_FA]

            new_W = W.assign(W - self.config.learning_rate*grad_W)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            self.train_step = [new_W, new_A]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

class AEFAModel(BaseModel):
    #Auto encoder FA model
    def __init__(self, config):
        super(AEFAModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        network_size = self.config.network_size
        n_layers = len(network_size)

        # first fully connected layer with 50 neurons using tanh activation
        l1 = tf.nn.tanh(fc_layer(self.x, 28*28, 50))
        # second fully connected layer with 50 neurons using tanh activation
        l2 = tf.nn.tanh(fc_layer(l1, 50, 50))
        # third fully connected layer with 2 neurons
        l3 = fc_layer(l2, 50, 2)
        # fourth fully connected layer with 50 neurons and tanh activation
        l4 = tf.nn.tanh(fa_layer(l3, 2, 50))
        # fifth fully connected layer with 50 neurons and tanh activation
        l5 = tf.nn.tanh(fc_layer(l4, 50, 50))
        y_p = fc_layer(l5, 50, 28*28)

        #W = weight_variable([50, 784])
        #B = weight_variable([50, 784])
        #b = bias_variable([784])
        #return tf_matmul_r(prev, W, B) + b
        #y_p = tf_matmul_r(l5, W, B)
        #y_p = tf.matmul(l5, W)

        #Build the network
        #layers = []
        #BP
        #layers.append(tf.sigmoid(fc_layer(self.x, self.config.state_size[0], network_size[0])))
        #FA
        #layers.append(tf.sigmoid(fa_layer(self.x, self.config.state_size[0], network_size[0])))
        #for idx in range(1, n_layers-1):
        #    #BP
        #    layers.append(tf.sigmoid(fc_layer(layers[idx-1], network_size[idx-1], network_size[idx])))
        #    #FA
        #    #layers.append(tf.sigmoid(fa_layer(layers[idx-1], network_size[idx-1], network_size[idx])))
        #y_p = fa_layer(layers[-1], network_size[-2], network_size[-1])

        with tf.name_scope("loss"):
            #mean squared error
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            e = (y_p - self.y)

            #grad_W = tf.gradients(xs=W, ys=self.loss)[0]

            #Feedback data for saving
            #Only take first item in epoch
            #delta_bp = tf.matmul(e, tf.transpose(W[0:m,:]))[0,:]
            #delta_fa = tf.matmul(e, tf.transpose(B))[0,:]
            #norm_W = tf.norm(W)
            #norm_B = tf.norm(B)
            #error_FA = tf.norm(delta_bp - delta_fa)
            #alignment = tf.reduce_sum(tf.multiply(delta_fa,delta_bp))/tf.norm(delta_fa)/tf.norm(delta_bp)

            #Also need to add eigenvector stuff
            #self.training_metrics = [alignment, norm_W, norm_B, error_FA]

            #new_W = W.assign(W - self.config.learning_rate*grad_W)
            #self.train_step = [new_W]

            self.train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                             global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
