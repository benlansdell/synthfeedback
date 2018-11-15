import os
os.environ["CUDA_VISIBLE_DEVICES"]=''

from data_loader.data_generator import MNISTDataGenerator, LinearDataGenerator
from models.npmodels import NPModel4,DirectNPModel4,AENPModel,AEDFANPModel
from trainers.sf_trainer import SFTrainer, AESFTrainer
from utils.config import process_config
import shutil
import tensorflow as tf
import numpy as np
import numpy.random as rng
from data_loader.data_generator import MNISTDataGenerator, LinearDataGenerator
from models.npmodels import NPModel4,DirectNPModel4,AENPModel,AEDFANPModel
from trainers.sf_trainer import SFTrainer, AESFTrainer
from utils.config import process_config
import shutil
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import operator
from utils.utils import tf_matmul_r, tf_matmul_l, tf_eigvecs, tf_eigvals
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#p = self.config.state_size[0]
p=784# inshape 
m =1000# hiddenshap
j = 10#outshpae
#n = 10
var_xi = 0.1
learning_rate=6.30957344e-05
lmda_learning_rate=0.00025
#lmda_learning_rate=0
#Training data inputs
x=tf.placeholder(tf.float32,[None,p], name = 'x')
y=tf.placeholder(tf.float32,[None,j], name = 'y')

#Scale weight initialization
alpha0 = np.sqrt(2.0/p)
alpha1 = np.sqrt(2.0/m)
alpha2 = np.sqrt(2.0/j)
alpha3 = 1

A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
W = tf.Variable(rng.randn(m+1,j)*alpha1, name="output_weights", dtype=tf.float32)
B = tf.Variable(rng.randn(m+1,j)*alpha1, name="feedback_weights", dtype=tf.float32)

# network architecture with ones added for bias terms
#0 = tf.ones([batch_size, 1], tf.float32)
#1 = tf.ones([batch_size, 1], tf.float32)
e0 = tf.ones([tf.shape(x)[0], 1], tf.float32)
e1 = tf.ones([tf.shape(x)[0], 1], tf.float32)
# e0 = tf.ones([1,batch_size], tf.float32)
# e1 = tf.ones([1,batch_size], tf.float32)

x_aug = tf.concat([x, e0], 1)
h = tf.sigmoid(tf.matmul(x_aug, A))
#Make some noise
h_aug = tf.concat([h, e1], 1)
xi = tf.random_normal(shape=tf.shape(h_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
h_tilde = h_aug + xi
#Add noise to hidden layer
#y_p = tf.sigmoid(tf.matmul(h_tilde, W))
y_p = tf.matmul(h_tilde, W)
y_p_0 = tf.matmul(h_aug, W)

trainable = [A, W, B]

#mean squared error
loss = tf.reduce_sum(tf.pow(y_p-y, 2))/2
loss_0 = tf.reduce_sum(tf.pow(y_p_0-y, 2))/2
e = (y_p - y)

h_prime = tf.multiply(h_tilde, 1-h_tilde)[:,0:m]

#Feedback data for saving
#Only take first item in epoch
delta_bp = tf.matmul(e, tf.transpose(W[0:m,:]))[0,:]
delta_fa = tf.matmul(e, tf.transpose(B[0:m,:]))[0,:]
norm_W = tf.norm(W)
norm_B = tf.norm(B)
error_FA = tf.norm(delta_bp - delta_fa)
alignment = tf.reduce_sum(tf.multiply(delta_fa,delta_bp))/tf.norm(delta_fa)/tf.norm(delta_bp)
norm_diff = tf.norm(W - B)
eigs = tf_eigvals(tf.matmul(tf.transpose(B), W))

#Compute updates for W and A (based on B)
#Node pert
lmda = tf.matmul(e, tf.transpose(B[0:m,:]))
#Backprop
#lmda = tf.matmul(e, tf.transpose(W[0:m,:]))
grad_W = tf.gradients(xs=W, ys=loss)[0]
grad_A = tf.matmul(tf.transpose(x_aug), tf.multiply(h_prime, lmda))
grad_B = tf.matmul(tf.matmul(B, tf.transpose(e)) - tf.transpose(xi)*(loss - loss_0)/var_xi, e)

new_W = W.assign(W - learning_rate*grad_W)
new_A = A.assign(A - learning_rate*grad_A)            
new_B = B.assign(B - lmda_learning_rate
                 *grad_B)            
train_step = [new_W, new_A, new_B]
correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Also need to add eigenvector stuff
training_metrics = [alignment, norm_W, norm_B, error_FA, eigs[0]]



init = tf.global_variables_initializer()
iteration= 100000
epoch=10
store_al=np.zeros((epoch,iteration))
store_df=np.zeros((epoch,iteration))
store_err=np.zeros((epoch,iteration))
store_acc=np.zeros((epoch,iteration))
# store_out=np.zeros((N, 4))
# x_in=[[0,0],[0,1],[1,0],[1,1]]
batch_size=50
with tf.Session() as sess:

    sess.run(init)
    for epoch_no in range(epoch):
        for idx in range(iteration):
            (train_x, train_y) = mnist.train.next_batch(batch_size) 
            _,align,diff,err,acc=sess.run([train_step,alignment,norm_diff,loss_0,accuracy],feed_dict={x: train_x, y: train_y})
            
            if np.isnan(err)==True:
        
                print("\n\tModel does not converge!!!\n")
                break
#             store_out[idx,:] = out[0][:,0]
            store_err[epoch_no,idx]=err
            store_al[epoch_no,idx]=align
            store_acc[epoch_no,idx]=acc
        print("Run No:%d completed"%epoch_no)
#             store_df.append(diff)
        #print(align)
 
with open("MNIST_NP.pkl",'wb') as f:
    pickle.dump([store_err,store_al,store_acc,iteration,epoch],f)