from base.base_model import BaseModel
import tensorflow as tf

from numpy import random as rng
import numpy as np 

from utils.utils import tf_matmul_r, tf_matmul_l, tf_eigvecs, tf_eigvals

from npmodels import weight_variable,bias_variable,weight_w_bias,fc_layer, fa_layer,fc_layer_noise,tf_align

class WMModel4(BaseModel):
    #Four layers version
    def __init__(self, config):
        super(WMModel4, self).__init__(config)
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

        #Compute unperturbed output
        h2_0 = tf.sigmoid(tf.matmul(h1_aug, W1))
        h2_0_aug = tf.concat([h2_0, e1], 1)
        y_p_0 = tf.matmul(h2_0_aug, W2)

        self.trainable = [A, W1, W2, B1, B2]

        with tf.name_scope("loss"):
            #mean squared error
            
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
            p1=tf.random_normal(shape=tf.shape(h1_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
            p2=tf.random_normal(shape=tf.shape(h2_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)

            p12= tf.sigmoid(tf.matmul(p1, W1))#perturbation of layer 1 going to 2
            p12_mean=p12-tf.mean(p12,axis=0)
            
            p2y=tf.sigmoid(tf.matmul(p2, W2))#perturbation of layer 2 going to y
            p2y_mean=p2y-tf.mean(p2y,axis=0)
            
            grad_B1 = tf.matmul(tf.transpose(p1),p12_mean)-self.config.lambda*B1
            grad_B2 = tf.matmul(tf.transpose(p2),p2y_mean)-self.config.lambda*B2

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_A = A.assign(A - self.config.learning_rate*grad_A)

            #Train with SGD
            new_B1 = B1.assign(B1 - self.config.lmda_learning_rate*grad_B1)
            new_B2 = B2.assign(B2 - self.config.lmda_learning_rate*grad_B2)

            self.train_step = [new_W1, new_A, new_B1, new_W2, new_B2]
            
            self.train_step_mirror = [new_B1, new_B2]

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
            self.training_metric_tags.append('align_B%d'%(idx+2))
            self.training_metrics.append(alignment)
            self.training_metric_tags.append('norm_W%d_B%d'%(idx+2, idx+2))
            self.training_metrics.append(norm)
            self.training_metric_tags.append('norm_gradB%d'%(idx+2))
            self.training_metrics.append(gradBs[idx])
