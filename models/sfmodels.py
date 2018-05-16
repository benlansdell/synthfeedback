from base.base_model import BaseModel
import tensorflow as tf

from numpy import random as rng
import numpy as np 

from utils.utils import tf_matmul_r, tf_matmul_l, tf_eigvecs, tf_eigvals

#Network building functions
def weight_variable(shape):
    sigma = np.sqrt(2.0/shape[0])
    initial = tf.truncated_normal(shape, stddev=sigma)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fc_layer_w(prev, input_size, output_size, batch_size):
    W = weight_variable([input_size+1, output_size])
    e0 = tf.ones([batch_size, 1], tf.float32)
    prev_aug = tf.concat([prev, e0], 1)
    return tf.matmul(prev_aug, W), W

def fc_layer(prev, input_size, output_size):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(prev, W) + b

def fa_layer(prev, input_size, output_size):
    W = weight_variable([input_size, output_size])
    B = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf_matmul_r(prev, W, B) + b

def fa_layer_w(prev, input_size, output_size, batch_size):
    W = weight_variable([input_size+1, output_size])
    B = weight_variable([input_size+1, output_size])
    e0 = tf.ones([batch_size, 1], tf.float32)
    prev_aug = tf.concat([prev, e0], 1)
    return tf.matmul(prev_aug, W), W, B

def tf_align(x, y):
    return tf.reduce_sum(tf.multiply(x,y))/tf.norm(x)/tf.norm(y)

class BPModel(BaseModel):
    def __init__(self, config):
        super(BPModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        # network architecture
        d1 = tf.layers.dense(self.x, 512, activation=tf.sigmoid, name="dense1")
        d2 = tf.layers.dense(d1, 10, name="dense2")
        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum(tf.pow(d2-self.y, 2))/2
            self.train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


class FAModel(BaseModel):
    def __init__(self, config):
        super(FAModel, self).__init__(config)
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
        h_aug = tf.concat([h, e1], 1)
        y_p = tf_matmul_r(h_aug, W, B)

        self.trainable = [A, W]

        with tf.name_scope("loss"):
            #mean squared error
            #cost = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2/self.config.batch_size
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            grad_W = tf.gradients(xs=W, ys=self.loss)[0]
            grad_A = tf.gradients(xs=A, ys=self.loss)[0]

            #Feedback data for saving
            #Only take first item in epoch
            e = (y_p - self.y)
            delta_bp = tf.matmul(e, tf.transpose(W[0:m,:]))[0,:]
            delta_fa = tf.matmul(e, tf.transpose(B[0:m,:]))[0,:]
            norm_W = tf.norm(W)
            norm_B = tf.norm(B)
            error_FA = tf.norm(delta_bp - delta_fa)
            alignment = tf.reduce_sum(tf.multiply(delta_fa,delta_bp))/tf.norm(delta_fa)/tf.norm(delta_bp)
            eigs = tf_eigvals(tf.matmul(tf.transpose(B), W))

            #Also need to add eigenvector stuff
            self.training_metrics = [alignment, norm_W, norm_B, error_FA, eigs[0]]

            new_W = W.assign(W - self.config.learning_rate*grad_W)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            self.train_step = [new_W, new_A]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

class BPModel4(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(BPModel4, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # set initial feedforward and feedback weights
        m = 512
        j = 200
        n = 10
        p = self.config.state_size[0]

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
        h2 = tf.sigmoid(tf.matmul(h1_aug, W1))
        h2_aug = tf.concat([h2, e1], 1)
        y_p = tf.matmul(h2_aug, W2)

        with tf.name_scope("loss"):
            #mean squared error
            #cost = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2/self.config.batch_size
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            grad_A = tf.gradients(xs=A, ys=self.loss)[0]

            e = (y_p - self.y)
            #BP Tensorflow
            #grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            #grad_A = tf.gradients(xs=A, ys=self.loss)[0]
            
            #BP manually
            #d = tf.multiply(h2_prime, tf.matmul(e, tf.transpose(W2[0:j,:])))
            #grad_W1 = tf.matmul(tf.transpose(h1_aug), d)
            #grad_A = tf.matmul(tf.transpose(x_aug), tf.multiply(h1_prime, tf.matmul(d, tf.transpose(W1[0:m,:]))))
            
            #FA

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

            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            self.train_step = [new_W2, new_W1, new_A]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

class FAModel4(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(FAModel4, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # set initial feedforward and feedback weights
        m = 512
        j = 200
        n = 10
        p = self.config.state_size[0]

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
        h2 = tf.sigmoid(tf_matmul_r(h1_aug, W1, B1))
        h2_aug = tf.concat([h2, e1], 1)
        y_p = tf_matmul_r(h2_aug, W2, B2)

        with tf.name_scope("loss"):
            #mean squared error
            #cost = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2/self.config.batch_size
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            grad_A = tf.gradients(xs=A, ys=self.loss)[0]

            e = (y_p - self.y)
            h1_prime = tf.multiply(h1_aug, 1-h1_aug)[:,0:m]
            h2_prime = tf.multiply(h2_aug, 1-h2_aug)[:,0:j]

            #FA
            lmda2 = tf.matmul(e, tf.transpose(B2[0:j,:]))
            d2 = np.multiply(h2_prime, lmda2)

            #Feedback data for saving
            #Only take first item in epoch
            delta_bp2 = tf.matmul(e, tf.transpose(W2[0:m,:]))[0,:]
            delta_fa2 = tf.matmul(e, tf.transpose(B2[0:m,:]))[0,:]
            delta_bp1 = tf.matmul(d2, tf.transpose(W1[0:m,:]))[0,:]
            delta_fa1 = tf.matmul(d2, tf.transpose(B1[0:m,:]))[0,:]
            norm_W1 = tf.norm(W1)
            norm_B1 = tf.norm(B1)
            norm_W2 = tf.norm(W2)
            norm_B2 = tf.norm(B2)
            error_FA1 = tf.norm(delta_bp1 - delta_fa1)
            error_FA2 = tf.norm(delta_bp2 - delta_fa2)
            alignment1 = tf_align(delta_fa1, delta_bp1)
            alignment2 = tf_align(delta_fa2, delta_bp2)

            #evals = tf_eigvals(tf.matmul(tf.transpose(B), W))
            #evecs = tf_eigvecs(tf.matmul(tf.transpose(B), W))

            #self.training_metrics = [alignment1, norm_W1, norm_B1, error_FA1, alignment2, norm_W2, norm_B2, error_FA2]
            #for idx in range(n):
                #Compute alignment with evecs of 

            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            self.train_step = [new_W2, new_W1, new_A]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

class DirectFAModel4(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(DirectFAModel4, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # set initial feedforward and feedback weights
        m = 512
        j = 200
        n = 10
        p = self.config.state_size[0]

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)
        B1 = tf.Variable(rng.randn(m,n)*alpha3, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j,n)*alpha3, name="feedback_weights2", dtype=tf.float32)

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        e1 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = tf.sigmoid(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e1], 1)
        h2 = tf.sigmoid(tf.matmul(h1_aug, W1))
        h2_aug = tf.concat([h2, e1], 1)
        y_p = tf.matmul(h2_aug, W2)

        with tf.name_scope("loss"):
            #mean squared error
            #cost = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2/self.config.batch_size
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]

            e = (y_p - self.y)
            h1_prime = h1*(1-h1)
            h2_prime = h2*(1-h2)
            #BP Tensorflow
            #grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            #grad_A = tf.gradients(xs=A, ys=self.loss)[0]
            
            #BP manually
            #d = tf.multiply(h2_prime, tf.matmul(e, tf.transpose(W2[0:j,:])))
            #grad_W1 = tf.matmul(tf.transpose(h1_aug), d)
            #grad_A = tf.matmul(tf.transpose(x_aug), tf.multiply(h1_prime, tf.matmul(d, tf.transpose(W1[0:m,:]))))
            
            #Direct FA
            d2 = tf.multiply(h2_prime, tf.matmul(e, tf.transpose(B2)))
            d1 = tf.multiply(h1_prime, tf.matmul(e, tf.transpose(B1)))
            grad_W1 = tf.matmul(tf.transpose(h1_aug), d2)
            grad_A = tf.matmul(tf.transpose(x_aug), d1)

            #Feedback data for saving
            #Only take first item in epoch
            delta_bp1 = tf.matmul(e, tf.transpose(W1[0:m,:]))[0,:]
            delta_fa1 = tf.matmul(e, tf.transpose(B1))[0,:]
            delta_bp2 = tf.matmul(e, tf.transpose(W2[0:m,:]))[0,:]
            delta_fa2 = tf.matmul(e, tf.transpose(B2))[0,:]
            norm_W1 = tf.norm(W1)
            norm_B1 = tf.norm(B1)
            norm_W2 = tf.norm(W2)
            norm_B2 = tf.norm(B2)
            error_FA1 = tf.norm(delta_bp1 - delta_fa1)
            error_FA2 = tf.norm(delta_bp2 - delta_fa2)
            alignment1 = tf_align(delta_fa1, delta_bp1)
            alignment2 = tf_align(delta_fa2, delta_bp2)

            evals = tf_eigvals(tf.matmul(tf.transpose(B), W))
            evecs = tf_eigvecs(tf.matmul(tf.transpose(B), W))

            self.training_metrics = [alignment1, norm_W1, norm_B1, error_FA1, alignment2, norm_W2, norm_B2, error_FA2]
            #Compute alignment between e and evecs of B^TW


            #Compute alignment between input at layer and singular values of W, B

            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            self.train_step = [new_W2, new_W1, new_A]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

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
        l4 = tf.nn.tanh(fc_layer(l3, 2, 50))
        # fifth fully connected layer with 50 neurons and tanh activation
        l5 = tf.nn.tanh(fc_layer(l4, 50, 50))
        y_p = tf.nn.relu(fc_layer(l5, 50, 28*28))

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
        l2 = tf.nn.tanh(fa_layer(l1, 50, 50))
        # third fully connected layer with 2 neurons
        l3 = fa_layer(l2, 50, 2)
        # fourth fully connected layer with 50 neurons and tanh activation
        l4 = tf.nn.tanh(fa_layer(l3, 2, 50))
        # fifth fully connected layer with 50 neurons and tanh activation
        l5 = tf.nn.tanh(fa_layer(l4, 50, 50))
        y_p = tf.nn.relu(fa_layer(l5, 50, 28*28))
        self.y_p = y_p

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

class AEBPModel(BaseModel):
    #Auto encoder FA model
    def __init__(self, config):
        super(AEBPModel, self).__init__(config)
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
        l4 = tf.nn.tanh(fc_layer(l3, 2, 50))
        # fifth fully connected layer with 50 neurons and tanh activation
        l5 = tf.nn.tanh(fc_layer(l4, 50, 50))
        y_p = tf.nn.relu(fc_layer(l5, 50, 28*28))

        self.y_p = y_p

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

class AEDFAModel(BaseModel):
    #Auto encoder FA model
    def __init__(self, config):
        super(AEDFAModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        network_size = self.config.network_size
        n_layers = len(network_size)
        batch_size = self.config.batch_size

        alpha1 = np.sqrt(2.0/28/28)

        e0 = tf.ones([batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)

        # first fully connected layer with 50 neurons using tanh activation
        h1, W1 = fc_layer_w(self.x, 28*28, 50, batch_size)
        h1_aug = tf.concat([h1, e0], 1)
        l1 = tf.nn.tanh(h1)
        # second fully connected layer with 50 neurons using tanh activation
        h2, W2 = fc_layer_w(l1, 50, 50, batch_size)
        h2_aug = tf.concat([h2, e0], 1)
        l2 = tf.nn.tanh(h2)
        # third fully connected layer with 2 neurons
        h3, W3 = fc_layer_w(l2, 50, 2, batch_size)
        h3_aug = tf.concat([h3, e0], 1)
        l3 = tf.nn.tanh(h3)
        # fourth fully connected layer with 50 neurons and tanh activation
        h4, W4 = fc_layer_w(l3, 2, 50, batch_size)
        h4_aug = tf.concat([h4, e0], 1)
        l4 = tf.nn.tanh(h4)
        # fifth fully connected layer with 50 neurons and tanh activation
        h5, W5 = fc_layer_w(l4, 50, 50, batch_size)
        h5_aug = tf.concat([h5, e0], 1)
        l5 = tf.nn.tanh(h5)

        h6, W6 = fc_layer_w(l5, 50, 28*28, batch_size)
        h6_aug = tf.concat([h6, e0], 1)
        y_p = tf.nn.relu(h6)

        self.y_p = y_p

        B2 = tf.Variable(rng.randn(network_size[-6],network_size[-1])*alpha1, name="feedback_weights1", dtype=tf.float32)
        B3 = tf.Variable(rng.randn(network_size[-5],network_size[-1])*alpha1, name="feedback_weights2", dtype=tf.float32)
        B4 = tf.Variable(rng.randn(network_size[-4],network_size[-1])*alpha1, name="feedback_weights3", dtype=tf.float32)
        B5 = tf.Variable(rng.randn(network_size[-3],network_size[-1])*alpha1, name="feedback_weights4", dtype=tf.float32)
        B6 = tf.Variable(rng.randn(network_size[-2],network_size[-1])*alpha1, name="feedback_weights5", dtype=tf.float32)

        with tf.name_scope("loss"):
            #mean squared error
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            e = (y_p - self.y)

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

            grad_W1 = tf.matmul(tf.transpose(x_aug), tf.matmul(e, tf.transpose(B2)))
            grad_W2 = tf.matmul(tf.transpose(h1_aug), tf.matmul(e, tf.transpose(B3)))
            grad_W3 = tf.matmul(tf.transpose(h2_aug), tf.matmul(e, tf.transpose(B4)))
            grad_W4 = tf.matmul(tf.transpose(h3_aug), tf.matmul(e, tf.transpose(B5)))
            grad_W5 = tf.matmul(tf.transpose(h4_aug), tf.matmul(e, tf.transpose(B6)))
            grad_W6 = tf.matmul(tf.transpose(h5_aug), e)

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W3 = W3.assign(W3 - self.config.learning_rate*grad_W3)
            new_W4 = W4.assign(W4 - self.config.learning_rate*grad_W4)
            new_W5 = W5.assign(W5 - self.config.learning_rate*grad_W5)
            new_W6 = W6.assign(W6 - self.config.learning_rate*grad_W6)
            self.train_step = [new_W1, new_W2, new_W3, new_W4, new_W5, new_W6]

            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

##########################################################################################################
##########################################################################################################

class FAModel10(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(FAModel10, self).__init__(config)
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

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights1", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights3", dtype=tf.float32)
        W3 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights4", dtype=tf.float32)
        W4 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights5", dtype=tf.float32)
        W5 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights5", dtype=tf.float32)
        W6 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights6", dtype=tf.float32)
        W7 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights7", dtype=tf.float32)
        W8 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights8", dtype=tf.float32)
        W9 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)

        B1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights2", dtype=tf.float32)
        B3 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights3", dtype=tf.float32)
        B4 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights4", dtype=tf.float32)
        B5 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights5", dtype=tf.float32)
        B6 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights6", dtype=tf.float32)
        B7 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights7", dtype=tf.float32)
        B8 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights8", dtype=tf.float32)
        B9 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights9", dtype=tf.float32)

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = tf.nn.relu(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e0], 1)
        h2 = tf.nn.relu(tf_matmul_r(h1_aug, W1, B1))
        h2_aug = tf.concat([h2, e0], 1)
        h3 = tf.nn.relu(tf_matmul_r(h2_aug, W2, B2))
        h3_aug = tf.concat([h3, e0], 1)
        h4 = tf.nn.relu(tf_matmul_r(h3_aug, W3, B3))
        h4_aug = tf.concat([h4, e0], 1)
        h5 = tf.nn.relu(tf_matmul_r(h4_aug, W4, B4))
        h5_aug = tf.concat([h5, e0], 1)
        h6 = tf.nn.relu(tf_matmul_r(h5_aug, W5, B5))
        h6_aug = tf.concat([h6, e0], 1)
        h7 = tf.nn.relu(tf_matmul_r(h6_aug, W6, B6))
        h7_aug = tf.concat([h7, e0], 1)
        h8 = tf.nn.relu(tf_matmul_r(h7_aug, W7, B7))
        h8_aug = tf.concat([h8, e0], 1)
        h9 = tf.nn.relu(tf_matmul_r(h8_aug, W8, B8))
        h9_aug = tf.concat([h9, e0], 1)
        y_p = tf_matmul_r(h9_aug, W9, B9)

        with tf.name_scope("loss"):
            #mean squared error
            #cost = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2/self.config.batch_size
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            grad_W9 = tf.gradients(xs=W9, ys=self.loss)[0]
            grad_W8 = tf.gradients(xs=W8, ys=self.loss)[0]
            grad_W7 = tf.gradients(xs=W7, ys=self.loss)[0]
            grad_W6 = tf.gradients(xs=W6, ys=self.loss)[0]
            grad_W5 = tf.gradients(xs=W5, ys=self.loss)[0]
            grad_W4 = tf.gradients(xs=W4, ys=self.loss)[0]
            grad_W3 = tf.gradients(xs=W3, ys=self.loss)[0]
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            grad_A = tf.gradients(xs=A, ys=self.loss)[0]

            e = (y_p - self.y)
            #BP Tensorflow
            #grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            #grad_A = tf.gradients(xs=A, ys=self.loss)[0]
            
            #BP manually
            #d = tf.multiply(h2_prime, tf.matmul(e, tf.transpose(W2[0:j,:])))
            #grad_W1 = tf.matmul(tf.transpose(h1_aug), d)
            #grad_A = tf.matmul(tf.transpose(x_aug), tf.multiply(h1_prime, tf.matmul(d, tf.transpose(W1[0:m,:]))))
            
            #FA

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
            new_W9 = W9.assign(W9 - self.config.learning_rate*grad_W9)
            new_W8 = W8.assign(W8 - self.config.learning_rate*grad_W8)
            new_W7 = W7.assign(W7 - self.config.learning_rate*grad_W7)
            new_W6 = W6.assign(W6 - self.config.learning_rate*grad_W6)
            new_W5 = W5.assign(W5 - self.config.learning_rate*grad_W5)
            new_W4 = W4.assign(W4 - self.config.learning_rate*grad_W4)
            new_W3 = W3.assign(W3 - self.config.learning_rate*grad_W3)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            self.train_step = [new_W9, new_W8, new_W7, new_W6, new_W5, new_W4, new_W3, new_W2, new_W1, new_A]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

class BPModel10(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(BPModel10, self).__init__(config)
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

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights1", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights3", dtype=tf.float32)
        W3 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights4", dtype=tf.float32)
        W4 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights5", dtype=tf.float32)
        W5 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights5", dtype=tf.float32)
        W6 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights6", dtype=tf.float32)
        W7 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights7", dtype=tf.float32)
        W8 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights8", dtype=tf.float32)
        W9 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)

        B1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights2", dtype=tf.float32)
        B3 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights3", dtype=tf.float32)
        B4 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights4", dtype=tf.float32)
        B5 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights5", dtype=tf.float32)
        B6 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights6", dtype=tf.float32)
        B7 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights7", dtype=tf.float32)
        B8 = tf.Variable(rng.randn(j+1,j)*alpha2, name="feedback_weights8", dtype=tf.float32)
        B9 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights9", dtype=tf.float32)

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = tf.nn.relu(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e0], 1)
        h2 = tf.nn.relu(tf.matmul(h1_aug, W1))
        h2_aug = tf.concat([h2, e0], 1)
        h3 = tf.nn.relu(tf.matmul(h2_aug, W2))
        h3_aug = tf.concat([h3, e0], 1)
        h4 = tf.nn.relu(tf.matmul(h3_aug, W3))
        h4_aug = tf.concat([h4, e0], 1)
        h5 = tf.nn.relu(tf.matmul(h4_aug, W4))
        h5_aug = tf.concat([h5, e0], 1)
        h6 = tf.nn.relu(tf.matmul(h5_aug, W5))
        h6_aug = tf.concat([h6, e0], 1)
        h7 = tf.nn.relu(tf.matmul(h6_aug, W6))
        h7_aug = tf.concat([h7, e0], 1)
        h8 = tf.nn.relu(tf.matmul(h7_aug, W7))
        h8_aug = tf.concat([h8, e0], 1)
        h9 = tf.nn.relu(tf.matmul(h8_aug, W8))
        h9_aug = tf.concat([h9, e0], 1)
        y_p = tf.matmul(h9_aug, W9)
        #y_p = tf.matmul(h2_aug, W9)

        with tf.name_scope("loss"):
            #mean squared error
            #cost = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2/self.config.batch_size
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            grad_W9 = tf.gradients(xs=W9, ys=self.loss)[0]
            grad_W8 = tf.gradients(xs=W8, ys=self.loss)[0]
            grad_W7 = tf.gradients(xs=W7, ys=self.loss)[0]
            grad_W6 = tf.gradients(xs=W6, ys=self.loss)[0]
            grad_W5 = tf.gradients(xs=W5, ys=self.loss)[0]
            grad_W4 = tf.gradients(xs=W4, ys=self.loss)[0]
            grad_W3 = tf.gradients(xs=W3, ys=self.loss)[0]
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            grad_A = tf.gradients(xs=A, ys=self.loss)[0]

            e = (y_p - self.y)
            #BP Tensorflow
            #grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            #grad_A = tf.gradients(xs=A, ys=self.loss)[0]
            
            #BP manually
            #d = tf.multiply(h2_prime, tf.matmul(e, tf.transpose(W2[0:j,:])))
            #grad_W1 = tf.matmul(tf.transpose(h1_aug), d)
            #grad_A = tf.matmul(tf.transpose(x_aug), tf.multiply(h1_prime, tf.matmul(d, tf.transpose(W1[0:m,:]))))
            
            #FA

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
            new_W9 = W9.assign(W9 - self.config.learning_rate*grad_W9)
            new_W8 = W8.assign(W8 - self.config.learning_rate*grad_W8)
            new_W7 = W7.assign(W7 - self.config.learning_rate*grad_W7)
            new_W6 = W6.assign(W6 - self.config.learning_rate*grad_W6)
            new_W5 = W5.assign(W5 - self.config.learning_rate*grad_W5)
            new_W4 = W4.assign(W4 - self.config.learning_rate*grad_W4)
            new_W3 = W3.assign(W3 - self.config.learning_rate*grad_W3)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            self.train_step = [new_W9, new_W8, new_W7, new_W6, new_W5, new_W4, new_W3, new_W2, new_W1, new_A]
            #self.train_step = [new_W9, new_W1, new_A]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

##########################################################################################################
##########################################################################################################

class DFAModel10(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(DFAModel10, self).__init__(config)
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

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights1", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights3", dtype=tf.float32)
        W3 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights4", dtype=tf.float32)
        W4 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights5", dtype=tf.float32)
        W5 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights5", dtype=tf.float32)
        W6 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights6", dtype=tf.float32)
        W7 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights7", dtype=tf.float32)
        W8 = tf.Variable(rng.randn(j+1,j)*alpha2, name="hidden_weights8", dtype=tf.float32)
        W9 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)

        B1 = tf.Variable(rng.randn(m+1,n)*alpha1, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights2", dtype=tf.float32)
        B3 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights3", dtype=tf.float32)
        B4 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights4", dtype=tf.float32)
        B5 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights5", dtype=tf.float32)
        B6 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights6", dtype=tf.float32)
        B7 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights7", dtype=tf.float32)
        B8 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights8", dtype=tf.float32)
        B9 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights9", dtype=tf.float32)

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = tf.nn.relu(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e0], 1)
        h2 = tf.nn.relu(tf.matmul(h1_aug, W1))
        h2_aug = tf.concat([h2, e0], 1)
        h3 = tf.nn.relu(tf.matmul(h2_aug, W2))
        h3_aug = tf.concat([h3, e0], 1)
        h4 = tf.nn.relu(tf.matmul(h3_aug, W3))
        h4_aug = tf.concat([h4, e0], 1)
        h5 = tf.nn.relu(tf.matmul(h4_aug, W4))
        h5_aug = tf.concat([h5, e0], 1)
        h6 = tf.nn.relu(tf.matmul(h5_aug, W5))
        h6_aug = tf.concat([h6, e0], 1)
        h7 = tf.nn.relu(tf.matmul(h6_aug, W6))
        h7_aug = tf.concat([h7, e0], 1)
        h8 = tf.nn.relu(tf.matmul(h7_aug, W7))
        h8_aug = tf.concat([h8, e0], 1)
        h9 = tf.nn.relu(tf.matmul(h8_aug, W8))
        h9_aug = tf.concat([h9, e0], 1)
        y_p = tf.matmul(h9_aug, W9)

        with tf.name_scope("loss"):
            #mean squared error
            #cost = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2/self.config.batch_size
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            grad_W9 = tf.gradients(xs=W9, ys=self.loss)[0]
            grad_W8 = tf.gradients(xs=W8, ys=self.loss)[0]
            grad_W7 = tf.gradients(xs=W7, ys=self.loss)[0]
            grad_W6 = tf.gradients(xs=W6, ys=self.loss)[0]
            grad_W5 = tf.gradients(xs=W5, ys=self.loss)[0]
            grad_W4 = tf.gradients(xs=W4, ys=self.loss)[0]
            grad_W3 = tf.gradients(xs=W3, ys=self.loss)[0]
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            grad_A = tf.gradients(xs=A, ys=self.loss)[0]

            e = (y_p - self.y)
            #BP Tensorflow
            #grad_W1 = tf.gradients(xs=W1, ys=self.loss)[0]
            #grad_A = tf.gradients(xs=A, ys=self.loss)[0]
            
            #BP manually
            #d = tf.multiply(h2_prime, tf.matmul(e, tf.transpose(W2[0:j,:])))
            #grad_W1 = tf.matmul(tf.transpose(h1_aug), d)
            #grad_A = tf.matmul(tf.transpose(x_aug), tf.multiply(h1_prime, tf.matmul(d, tf.transpose(W1[0:m,:]))))
            
            #FA

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
            new_W9 = W9.assign(W9 - self.config.learning_rate*grad_W9)
            new_W8 = W8.assign(W8 - self.config.learning_rate*grad_W8)
            new_W7 = W7.assign(W7 - self.config.learning_rate*grad_W7)
            new_W6 = W6.assign(W6 - self.config.learning_rate*grad_W6)
            new_W5 = W5.assign(W5 - self.config.learning_rate*grad_W5)
            new_W4 = W4.assign(W4 - self.config.learning_rate*grad_W4)
            new_W3 = W3.assign(W3 - self.config.learning_rate*grad_W3)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            self.train_step = [new_W9, new_W8, new_W7, new_W6, new_W5, new_W4, new_W3, new_W2, new_W1, new_A]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
