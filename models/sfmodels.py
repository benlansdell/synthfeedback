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
