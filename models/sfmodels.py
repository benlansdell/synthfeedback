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
    return tf.matmul(prev_aug, W), W, prev_aug

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
    #Check they have the right dimensions...
    #if x.get_shape() != y.get_shape():
    #print "Vectors different shape"
    #print x.get_shape(), y.get_shape()
    theta = 180/np.pi*tf.abs(tf.acos(tf.reduce_sum(tf.multiply(x,y))/tf.norm(x)/tf.norm(y)))
    #if (theta > 90) and (theta < 180):
    #    theta = 90 - theta
    return tf.cond(tf.logical_and(tf.less(90.0,theta), tf.less(theta,180.0)), lambda: 180.0 - theta, lambda: theta)

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

class BPModel4_Small(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(BPModel4_Small, self).__init__(config)
        self.build_model()
        #self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # set initial feedforward and feedback weights
        m = 50
        j = 20
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
        B1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights2", dtype=tf.float32)

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
        #self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # set initial feedforward and feedback weights
        m = 512
        j = 200
        #m = 50
        #j = 20
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
        B1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights2", dtype=tf.float32)

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

            #True gradients...
            #dl1 = tf.gradients(xs=l1, ys=self.loss)[0]
            #dl2 = tf.gradients(xs=l2, ys=self.loss)[0]

            #Save training metrics
            Bs = [B1, B2]
            Ws = [W1, W2]
            es = [d2, e]
            #dls = [dl1, dl2, dl3, dl4, dl5]
            #ls = [l1, l2, l3, l4, l5]
            dls = []
            ls = [h1_aug, h2_aug]
            self._set_training_metrics(es, Bs, Ws, dls, ls, self.config.learning_rate)

    def _set_training_metrics(self, es, Bs, Ws, dls, ls, eta):

        k = 6
        #Alignment of B_{i+1}e with dl_i (feedback alignment)
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
            delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
            error_fa = tf.norm(delta_fa - delta_bp)/tf.norm(delta_fa)
            alignment = tf.abs(tf_align(delta_fa, delta_bp))

            #Eigenvector alignment (B^TW)
            evecs = tf.cast(tf_eigvecs(tf.matmul(Bs[idx], tf.transpose(Ws[idx]))), tf.float32)
            #Save each evec
            for j in range(k):
                ev = evecs[:,j]
                eva = tf.abs(tf_align(delta_fa, ev))
                self.training_metric_tags.append('ev_%d_%d'%(idx+1,j))
                self.training_metrics.append(eva)

            #Eigenvector alignment (I + eta B^TW)
            X = tf.matmul(Bs[idx], tf.transpose(Ws[idx]))
            evecs = tf.cast(tf_eigvecs(tf.eye(tf.shape(X)[0]) + eta*X), tf.float32)
            #save each evec
            for j in range(k):
                ev = evecs[:,j]
                eva = tf.abs(tf_align(delta_fa, ev))
                self.training_metric_tags.append('ev_I_eta_%d_%d'%(idx+1,j))
                self.training_metrics.append(eva)

            #Compute the SVD
            s,u,v = tf.svd(Ws[idx])
            for j in range(k):
                uj = u[:,j]
                uj_a = tf.abs(tf_align(uj, ls[idx]))
                self.training_metric_tags.append('uj_a_%d_%d'%(idx+1,j))
                self.training_metrics.append(uj_a)

            self.training_metric_tags.append('align_B%d'%(idx+1))
            self.training_metrics.append(alignment)

            self.training_metric_tags.append('error_fa_%d'%(idx+1))
            self.training_metrics.append(error_fa)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


class FAModel4_Small(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(FAModel4_Small, self).__init__(config)
        self.build_model()
        #self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # set initial feedforward and feedback weights
        m = 50
        j = 20
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
        B1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights2", dtype=tf.float32)

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

            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            self.train_step = [new_W2, new_W1, new_A]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Save training metrics
            Bs = [B1, B2]
            Ws = [W1, W2]
            es = [d2, e]
            dls = []
            ls = [h1_aug, h2_aug]
            self._set_training_metrics(es, Bs, Ws)

    # def _set_training_metrics(self, Ws, Bs, es):
    #     for idx in range(len(Bs)):
    #         delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
    #         delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
    #         alignment = tf.abs(tf_align(delta_fa, delta_bp))
    #         norm = tf.norm(Ws[idx] - Bs[idx])/tf.norm(Ws[idx])
    #         sgn_cong = tf.reduce_mean((tf.sign(Ws[idx])*tf.sign(Bs[idx])+1)/2)
    #         self.training_metric_tags.append('align_B%d'%(idx+2))
    #         self.training_metrics.append(alignment)
    #         self.training_metric_tags.append('norm_W%d_B%d'%(idx+2, idx+2))
    #         self.training_metrics.append(norm)
    #         self.training_metric_tags.append('sign_cong%d'%(idx+2))
    #         self.training_metrics.append(sgn_cong)

    #         #delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
    #         #delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
    #         #alignment = tf.abs(tf_align(delta_fa, delta_bp))


    def _set_training_metrics(self, es, Bs, Ws):

        k = 6
        #Alignment of B_{i+1}e with dl_i (feedback alignment)
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
            delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
            error_fa = tf.norm(delta_fa - delta_bp)/tf.norm(delta_fa)
            alignment = tf.abs(tf_align(delta_fa, delta_bp))
            norm = tf.norm(Ws[idx] - Bs[idx])/tf.norm(Ws[idx])
            sgn_cong = tf.reduce_mean((tf.sign(Ws[idx])*tf.sign(Bs[idx])+1)/2)
            # self.training_metric_tags.append('align_B%d'%(idx+1))
            # self.training_metrics.append(alignment)
            # self.training_metric_tags.append('error_fa_%d'%(idx+1))
            # self.training_metrics.append(error_fa)
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)
            self.training_metric_tags.append('norm_W%d_B%d'%(idx+2, idx+2))
            self.training_metrics.append(norm)
            self.training_metric_tags.append('sign_cong%d'%(idx+2))
            self.training_metrics.append(sgn_cong)


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

class FAModel4Linear(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(FAModel4Linear, self).__init__(config)
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
        B1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="feedback_weights1", dtype=tf.float32)
        B2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="feedback_weights2", dtype=tf.float32)

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        e1 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = tf.matmul(x_aug, A)
        h1_aug = tf.concat([h1, e1], 1)
        h2 = tf_matmul_r(h1_aug, W1, B1)
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

            #FA
            lmda2 = tf.matmul(e, tf.transpose(B2[0:j,:]))
            d2 = lmda2

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

            #True gradients...
            #dl1 = tf.gradients(xs=l1, ys=self.loss)[0]
            #dl2 = tf.gradients(xs=l2, ys=self.loss)[0]

            #Save training metrics
            Bs = [B1, B2]
            Ws = [W1, W2]
            es = [d2, e]
            #dls = [dl1, dl2, dl3, dl4, dl5]
            #ls = [l1, l2, l3, l4, l5]
            dls = []
            ls = [h1_aug, h2_aug]
            self._set_training_metrics(es, Bs, Ws, dls, ls, self.config.learning_rate)

    def _set_training_metrics(self, es, Bs, Ws, dls, ls, eta):

        k = 6
        #Alignment of B_{i+1}e with dl_i (feedback alignment)
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
            delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
            error_fa = tf.norm(delta_fa - delta_bp)/tf.norm(delta_fa)
            alignment = tf.abs(tf_align(delta_fa, delta_bp))

            #Eigenvector alignment (B^TW)
            evecs = tf.cast(tf_eigvecs(tf.matmul(Bs[idx], tf.transpose(Ws[idx]))), tf.float32)
            #Save each evec
            for j in range(k):
                ev = evecs[:,j]
                eva = tf.abs(tf_align(delta_fa, ev))
                self.training_metric_tags.append('ev_%d_%d'%(idx+1,j))
                self.training_metrics.append(eva)

            #Eigenvector alignment (I + eta B^TW)
            X = tf.matmul(Bs[idx], tf.transpose(Ws[idx]))
            evecs = tf.cast(tf_eigvecs(tf.eye(tf.shape(X)[0]) + eta*X), tf.float32)
            #save each evec
            for j in range(k):
                ev = evecs[:,j]
                eva = tf.abs(tf_align(delta_fa, ev))
                self.training_metric_tags.append('ev_I_eta_%d_%d'%(idx+1,j))
                self.training_metrics.append(eva)

            #Compute the SVD
            s,u,v = tf.svd(Ws[idx])
            for j in range(k):
                uj = u[:,j]
                uj_a = tf.abs(tf_align(uj, ls[idx]))
                self.training_metric_tags.append('uj_a_%d_%d'%(idx+1,j))
                self.training_metrics.append(uj_a)

            self.training_metric_tags.append('align_B%d'%(idx+1))
            self.training_metrics.append(alignment)

            self.training_metric_tags.append('error_fa_%d'%(idx+1))
            self.training_metrics.append(error_fa)

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

            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            self.train_step = [new_W2, new_W1, new_A]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            dl1 = tf.gradients(xs=h1_aug, ys=self.loss)[0]
            dl2 = tf.gradients(xs=h2_aug, ys=self.loss)[0]

            #Save training metrics
            Bs = [B1, B2]
            Ws = [W1, W2]
            dls = [dl1, dl2]
            ls = [h1_aug, h2_aug]
            self._set_training_metrics(e, Bs, Ws, dls, ls, self.config.learning_rate)

    def _set_training_metrics(self, e, Bs, Ws, dls, ls, eta):

        k = 6
        #Alignment of B_{i+1}e with dl_i (feedback alignment)
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(e, tf.transpose(Bs[idx]))[0,:]
            delta_bp = dls[idx][0,:]

            error_fa = tf.norm(delta_fa - delta_bp[0:-1])
            alignment = tf_align(delta_fa, delta_bp[0:-1])

            #Compute the SVD
            s,u,v = tf.svd(Ws[idx])
            for j in range(k):
                uj = u[:,j]
                uj_a = tf_align(uj, ls[idx])
                self.training_metric_tags.append('uj_a_%d_%d'%(idx+1,j))
                self.training_metrics.append(uj_a)

            self.training_metric_tags.append('align_B%d'%(idx+1))
            self.training_metrics.append(alignment)

            self.training_metric_tags.append('error_fa_%d'%(idx+1))
            self.training_metrics.append(error_fa)

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
        alpha0 = 0.01
        alpha1 = 0.01
        alpha2 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.rand(p,m)*alpha0-alpha0/2, name="hidden_weights", dtype=tf.float32)
        W = tf.Variable(rng.rand(m,n)*alpha1-alpha1/2, name="output_weights", dtype=tf.float32)
        B = tf.Variable(rng.rand(m,n)*alpha2, name="feedback_weights", dtype=tf.float32)

        # network architecture with ones added for bias terms
        h = tf.matmul(self.x, A)
        y_p = tf_matmul_r(h, W, B)

        with tf.name_scope("loss"):
            #mean squared error
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            grad_W = tf.gradients(xs=W, ys=self.loss)[0]
            grad_A = tf.gradients(xs=A, ys=self.loss)[0]

            #Feedback data for saving
            e = (y_p - self.y)

            new_W = W.assign(W - self.config.learning_rate*grad_W)
            new_A = A.assign(A - self.config.learning_rate*grad_A)            
            self.train_step = [new_W, new_A]
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #Save training metrics
            Bs = [B]
            Ws = [W]
            dls = [tf.matmul(e, tf.transpose(W))]
            ls = [h]
            self._set_training_metrics(e, Bs, Ws, dls, ls, self.config.learning_rate)

    def _set_training_metrics(self, e, Bs, Ws, dls, ls, eta):

        k = 6
        #Alignment of B_{i+1}e with dl_i (feedback alignment)
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(e, tf.transpose(Bs[idx]))[0,:]
            delta_bp = dls[idx][0,:]

            error_fa = tf.norm(delta_fa - delta_bp)/tf.norm(delta_fa)
            alignment = tf_align(delta_fa, delta_bp)

            #Eigenvector alignment (B^TW)
            evecs = tf.cast(tf_eigvecs(tf.matmul(Bs[idx], tf.transpose(Ws[idx]))), tf.float32)
            #Save each evec
            for j in range(k):
                ev = evecs[:,j]
                eva = tf_align(tf.transpose(delta_fa), ev)
                self.training_metric_tags.append('ev_%d_%d'%(idx+1,j))
                self.training_metrics.append(eva)

            #Eigenvector alignment (I + eta B^TW)
            X = tf.matmul(Bs[idx], tf.transpose(Ws[idx]))
            evecs = tf.cast(tf_eigvecs(tf.eye(tf.shape(X)[0]) + eta*X), tf.float32)
            #save each evec
            for j in range(k):
                ev = evecs[:,j]
                eva = tf_align(tf.transpose(delta_fa), ev)
                self.training_metric_tags.append('ev_I_eta_%d_%d'%(idx+1,j))
                self.training_metrics.append(eva)

            #Compute the SVD
            s,u,v = tf.svd(Ws[idx])
            for j in range(k):
                uj = u[:,j]
                uj_a = tf_align(uj, ls[idx])
                self.training_metric_tags.append('uj_a_%d_%d'%(idx+1,j))
                self.training_metrics.append(uj_a)

            self.training_metric_tags.append('align_B%d'%(idx+1))
            self.training_metrics.append(alignment)

            self.training_metric_tags.append('error_fa_%d'%(idx+1))
            self.training_metrics.append(error_fa)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

####################################################################################################
####################################################################################################

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

        # first fully connected layer with 50 neurons using tanh activation
        l1_bp = tf.nn.tanh(fc_layer(self.x, 28*28, 50))
        # second fully connected layer with 50 neurons using tanh activation
        l2_bp = tf.nn.tanh(fc_layer(l1_bp, 50, 50))
        # third fully connected layer with 2 neurons
        l3_bp = fc_layer(l2_bp, 50, 2)
        # fourth fully connected layer with 50 neurons and tanh activation
        l4_bp = tf.nn.tanh(fc_layer(l3_bp, 2, 50))
        # fifth fully connected layer with 50 neurons and tanh activation
        l5_bp = tf.nn.tanh(fc_layer(l4_bp, 50, 50))
        y_p_bp = tf.nn.relu(fc_layer(l5_bp, 50, 28*28))
        loss_bp = tf.reduce_sum(tf.pow(y_p_bp-self.y, 2))/2

        with tf.name_scope("loss"):
            #mean squared error
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            e = (y_p - self.y)

            self.train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                             global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #FA gradients...
            fa_l1 = tf.gradients(xs=l1, ys=self.loss)[0]
            fa_l2 = tf.gradients(xs=l2, ys=self.loss)[0]
            fa_l3 = tf.gradients(xs=l3, ys=self.loss)[0]
            fa_l4 = tf.gradients(xs=l4, ys=self.loss)[0]
            fa_l5 = tf.gradients(xs=l5, ys=self.loss)[0]

            #True gradients...
            dl1 = tf.gradients(xs=l1_bp, ys=loss_bp)[0]
            dl2 = tf.gradients(xs=l2_bp, ys=loss_bp)[0]
            dl3 = tf.gradients(xs=l3_bp, ys=loss_bp)[0]
            dl4 = tf.gradients(xs=l4_bp, ys=loss_bp)[0]
            dl5 = tf.gradients(xs=l5_bp, ys=loss_bp)[0]

            #Save training metrics
            Bs = []
            Ws = []
            dls = [dl1, dl2, dl3, dl4, dl5]
            fas = [fa_l1, fa_l2,fa_l3,fa_l4,fa_l5]
            ls = [l1, l2, l3, l4, l5]
            self._set_training_metrics(e, Bs, Ws, fas, dls, ls, self.config.learning_rate)

    def _set_training_metrics(self, e, Bs, Ws, fas, dls, ls, eta):

        k = 6
        #Alignment of B_{i+1}e with dl_i (feedback alignment)
        for idx in range(len(dls)):
            delta_fa = fas[idx][0,:]
            delta_bp = dls[idx][0,:]
            #error_fa = tf.norm(delta_fa - dls[idx])
            alignment = tf.reduce_sum(tf.multiply(delta_fa,delta_bp))/tf.norm(delta_fa)/tf.norm(delta_bp)
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)

        k = 6
        #Alignment of B_{i+1}e with dl_i (feedback alignment)
        for idx in range(len(Bs)):
            delta_fa = tf.matmul(es[idx], tf.transpose(Bs[idx]))[0,:]
            delta_bp = tf.matmul(es[idx], tf.transpose(Ws[idx]))[0,:]
            error_fa = tf.norm(delta_fa - delta_bp)
            alignment = tf_align(delta_fa, delta_bp)

            #Eigenvector alignment (B^TW)
            evecs = tf.cast(tf_eigvecs(tf.matmul(Bs[idx], tf.transpose(Ws[idx]))), tf.float32)
            #Save each evec
            for j in range(k):
                ev = evecs[:,j]
                eva = tf_align(delta_fa, ev)
                self.training_metric_tags.append('ev_%d_%d'%(idx+1,j))
                self.training_metrics.append(eva)

            #Eigenvector alignment (I + eta B^TW)
            X = tf.matmul(Bs[idx], tf.transpose(Ws[idx]))
            evecs = tf.cast(tf_eigvecs(tf.eye(tf.shape(X)[0]) + eta*X), tf.float32)
            #save each evec
            for j in range(k):
                ev = evecs[:,j]
                eva = tf_align(delta_fa, ev)
                self.training_metric_tags.append('ev_I_eta_%d_%d'%(idx+1,j))
                self.training_metrics.append(eva)

            #Compute the SVD
            s,u,v = tf.svd(Ws[idx])
            for j in range(k):
                uj = u[:,j]
                uj_a = tf_align(uj, ls[idx])
                self.training_metric_tags.append('uj_a_%d_%d'%(idx+1,j))
                self.training_metrics.append(uj_a)

            self.training_metric_tags.append('align_B%d'%(idx+1))
            self.training_metrics.append(alignment)

            self.training_metric_tags.append('error_fa_%d'%(idx+1))
            self.training_metrics.append(error_fa)
            
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

        with tf.name_scope("loss"):
            #mean squared error
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            e = (y_p - self.y)
            self.train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                             global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

class AEBPModel5(BaseModel):
    #Auto encoder FA model
    def __init__(self, config):
        super(AEBPModel5, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        network_size = self.config.network_size
        n_layers = len(network_size)

        # first fully connected layer with 50 neurons using tanh activation
        l1 = tf.nn.tanh(fc_layer(self.x, 28*28, 200))
        # third fully connected layer with 2 neurons
        l2 = fc_layer(l1, 200, 2)
        # fourth fully connected layer with 50 neurons and tanh activation
        l3 = tf.nn.tanh(fc_layer(l2, 2, 200))
        # fifth fully connected layer with 50 neurons and tanh activation
        y_p = tf.nn.relu(fc_layer(l3, 200, 28*28))
        self.y_p = y_p

        with tf.name_scope("loss"):
            #mean squared error
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            e = (y_p - self.y)
            self.train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                             global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

class AEFAModel5(BaseModel):
    #Auto encoder FA model
    def __init__(self, config):
        super(AEFAModel5, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)

        network_size = self.config.network_size
        n_layers = len(network_size)

        # first fully connected layer with 50 neurons using tanh activation
        l1 = tf.nn.tanh(fa_layer(self.x, 28*28, 200))
        # third fully connected layer with 2 neurons
        l2 = fa_layer(l1, 200, 2)
        # fourth fully connected layer with 50 neurons and tanh activation
        l3 = tf.nn.tanh(fa_layer(l2, 2, 200))
        # fifth fully connected layer with 50 neurons and tanh activation
        y_p = tf.nn.relu(fa_layer(l3, 200, 28*28))
        self.y_p = y_p

        with tf.name_scope("loss"):
            #mean squared error
            self.loss = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            e = (y_p - self.y)
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
        h1, W1, x_aug = fc_layer_w(self.x, 28*28, 50, batch_size)
        h1_aug = tf.concat([h1, e0], 1)
        l1 = tf.nn.tanh(h1)
        # second fully connected layer with 50 neurons using tanh activation
        h2, W2, l1_aug = fc_layer_w(l1, 50, 50, batch_size)
        h2_aug = tf.concat([h2, e0], 1)
        l2 = tf.nn.tanh(h2)
        # third fully connected layer with 2 neurons
        h3, W3, l2_aug = fc_layer_w(l2, 50, 2, batch_size)
        h3_aug = tf.concat([h3, e0], 1)
        l3 = tf.nn.tanh(h3)
        # fourth fully connected layer with 50 neurons and tanh activation
        h4, W4, l3_aug = fc_layer_w(l3, 2, 50, batch_size)
        h4_aug = tf.concat([h4, e0], 1)
        l4 = tf.nn.tanh(h4)
        # fifth fully connected layer with 50 neurons and tanh activation
        h5, W5, l4_aug = fc_layer_w(l4, 50, 50, batch_size)
        h5_aug = tf.concat([h5, e0], 1)
        l5 = tf.nn.tanh(h5)

        h6, W6, l5_aug = fc_layer_w(l5, 50, 28*28, batch_size)
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
            l1_sigma = 1 - tf.multiply(l1, l1)
            l2_sigma = 1 - tf.multiply(l2, l2)
            l3_sigma = 1
            l4_sigma = 1 - tf.multiply(l4, l4)
            l5_sigma = 1 - tf.multiply(l5, l5)

            grad_W1 = tf.matmul(tf.transpose(x_aug), tf.multiply(tf.matmul(e, tf.transpose(B2)), l1_sigma))
            grad_W2 = tf.matmul(tf.transpose(l1_aug),tf.multiply(tf.matmul(e, tf.transpose(B3)), l2_sigma))
            grad_W3 = tf.matmul(tf.transpose(l2_aug),tf.multiply(tf.matmul(e, tf.transpose(B4)), l3_sigma))
            grad_W4 = tf.matmul(tf.transpose(l3_aug),tf.multiply(tf.matmul(e, tf.transpose(B5)), l4_sigma))
            grad_W5 = tf.matmul(tf.transpose(l4_aug),tf.multiply(tf.matmul(e, tf.transpose(B6)), l5_sigma))
            grad_W6 = tf.matmul(tf.transpose(l5_aug),e)

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_W3 = W3.assign(W3 - self.config.learning_rate*grad_W3)
            new_W4 = W4.assign(W4 - self.config.learning_rate*grad_W4)
            new_W5 = W5.assign(W5 - self.config.learning_rate*grad_W5)
            new_W6 = W6.assign(W6 - self.config.learning_rate*grad_W6)
            self.train_step = [new_W1, new_W2, new_W3, new_W4, new_W5, new_W6]

            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #True gradients...
            dl1 = tf.gradients(xs=l1, ys=self.loss)[0]
            dl2 = tf.gradients(xs=l2, ys=self.loss)[0]
            dl3 = tf.gradients(xs=l3, ys=self.loss)[0]
            dl4 = tf.gradients(xs=l4, ys=self.loss)[0]
            dl5 = tf.gradients(xs=l5, ys=self.loss)[0]

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
            alignment = tf.reduce_sum(tf.multiply(delta_fa,delta_bp))/tf.norm(delta_fa)/tf.norm(delta_bp)
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

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        h1_bp = tf.nn.relu(tf.matmul(x_aug, A))
        h1_aug_bp = tf.concat([h1_bp, e0], 1)
        h2_bp = tf.nn.relu(tf.matmul(h1_aug_bp, W1))
        h2_aug_bp = tf.concat([h2_bp, e0], 1)
        h3_bp = tf.nn.relu(tf.matmul(h2_aug_bp, W2))
        h3_aug_bp = tf.concat([h3_bp, e0], 1)
        h4_bp = tf.nn.relu(tf.matmul(h3_aug_bp, W3))
        h4_aug_bp = tf.concat([h4_bp, e0], 1)
        h5_bp = tf.nn.relu(tf.matmul(h4_aug_bp, W4))
        h5_aug_bp = tf.concat([h5_bp, e0], 1)
        h6_bp = tf.nn.relu(tf.matmul(h5_aug_bp, W5))
        h6_aug_bp = tf.concat([h6_bp, e0], 1)
        h7_bp = tf.nn.relu(tf.matmul(h6_aug_bp, W6))
        h7_aug_bp = tf.concat([h7_bp, e0], 1)
        h8_bp = tf.nn.relu(tf.matmul(h7_aug_bp, W7))
        h8_aug_bp = tf.concat([h8_bp, e0], 1)
        h9_bp = tf.nn.relu(tf.matmul(h8_aug_bp, W8))
        h9_aug_bp = tf.concat([h9_bp, e0], 1)
        y_p_bp = tf.matmul(h9_aug_bp, W9)
        loss_bp = tf.reduce_sum(tf.pow(y_p_bp-self.y, 2))/2
        
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

            #FA gradients...
            fa_l1 = tf.gradients(xs=h1_aug, ys=self.loss)[0]
            fa_l2 = tf.gradients(xs=h2_aug, ys=self.loss)[0]
            fa_l3 = tf.gradients(xs=h3_aug, ys=self.loss)[0]
            fa_l4 = tf.gradients(xs=h4_aug, ys=self.loss)[0]
            fa_l5 = tf.gradients(xs=h5_aug, ys=self.loss)[0]
            fa_l6 = tf.gradients(xs=h6_aug, ys=self.loss)[0]
            fa_l7 = tf.gradients(xs=h7_aug, ys=self.loss)[0]
            fa_l8 = tf.gradients(xs=h8_aug, ys=self.loss)[0]
            fa_l9 = tf.gradients(xs=h9_aug, ys=self.loss)[0]

            #True gradients...
            bp_l1 = tf.gradients(xs=h1_aug_bp, ys=loss_bp)[0]
            bp_l2 = tf.gradients(xs=h2_aug_bp, ys=loss_bp)[0]
            bp_l3 = tf.gradients(xs=h3_aug_bp, ys=loss_bp)[0]
            bp_l4 = tf.gradients(xs=h4_aug_bp, ys=loss_bp)[0]
            bp_l5 = tf.gradients(xs=h5_aug_bp, ys=loss_bp)[0]
            bp_l6 = tf.gradients(xs=h6_aug_bp, ys=loss_bp)[0]
            bp_l7 = tf.gradients(xs=h7_aug_bp, ys=loss_bp)[0]
            bp_l8 = tf.gradients(xs=h8_aug_bp, ys=loss_bp)[0]
            bp_l9 = tf.gradients(xs=h9_aug_bp, ys=loss_bp)[0]

            #Save training metrics
            Bs = []
            Ws = []
            bps = [bp_l1, bp_l2, bp_l3, bp_l4, bp_l5, bp_l6, bp_l7, bp_l8, bp_l9]
            fas = [fa_l1, fa_l2, fa_l3, fa_l4, fa_l5, fa_l6, fa_l7, fa_l8, fa_l9]
            ls = []
            self._set_training_metrics(e, Bs, Ws, fas, bps, ls, self.config.learning_rate)

    def _set_training_metrics(self, e, Bs, Ws, fas, bps, ls, eta):

        k = 6
        #Alignment of B_{i+1}e with dl_i (feedback alignment)
        for idx in range(len(bps)):
            delta_fa = fas[idx][0,:]
            delta_bp = bps[idx][0,:]
            #error_fa = tf.norm(delta_fa - dls[idx])
            alignment = tf.reduce_sum(tf.multiply(delta_fa,delta_bp))/tf.norm(delta_fa)/tf.norm(delta_bp)
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)

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
