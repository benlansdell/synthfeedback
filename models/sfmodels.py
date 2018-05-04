from base.base_model import BaseModel
import tensorflow as tf

from numpy import random as rng
import numpy as np 

from utils.utils import tf_matmul_r, tf_matmul_l

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