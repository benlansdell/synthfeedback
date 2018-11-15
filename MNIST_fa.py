import os
os.environ["CUDA_VISIBLE_DEVICES"]=''

import tensorflow as tf
import numpy as np
import numpy.random as rng
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import operator
from utils.utils import tf_matmul_r, tf_matmul_l, tf_eigvecs, tf_eigvals
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


p=784 #input size
m=1000 # No of hidden units 
n=10 # Output size 


#Training parameters 
batch_size = 50
eta=tf.constant(0.00001,dtype=tf.float32)

#Training data inputs
'''x=tf.placeholder(tf.float64,[None,inshape], name = 'x')
y=tf.placeholder(tf.float64,[None,outshape], name = 'y')
'''
x=tf.placeholder(tf.float32,[batch_size,784])
y=tf.placeholder(tf.float32,[batch_size,10])

#Scale weight initialization
alpha0 = np.sqrt(2.0/p)
alpha1 = np.sqrt(2.0/m)
alpha2 = 1

#parameters for feedback alignment
A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
W = tf.Variable(rng.randn(m+1,n)*alpha1, name="output_weights", dtype=tf.float32)
B = tf.Variable(rng.randn(m+1,n)*alpha2, name="feedback_weights", dtype=tf.float32)

e0 = tf.ones([batch_size, 1], tf.float32)
e1 = tf.ones([batch_size, 1], tf.float32)
x_aug = tf.concat([x, e0], 1)


h=tf.sigmoid(tf.matmul(x_aug,A))
h_aug=tf.concat([h,e1],1)
y_pred=tf_matmul_r(h_aug,W,B) 

trainable=[A,W]
          
e=y_pred-y
loss = tf.reduce_sum(tf.pow(e, 2))/2
grad_W=tf.gradients(xs=W,ys=loss)[0]
grad_A=tf.gradients(xs=A,ys=loss)[0]

delta_bp = tf.matmul(e, tf.transpose(W))[0,:]
delta_fa = tf.matmul(e, tf.transpose(B))[0,:]

alignment = tf.reduce_sum(tf.multiply(delta_fa,delta_bp))/tf.norm(delta_fa)/tf.norm(delta_bp)
# eigs = tf_eigvals(tf.matmul(tf.transpose(B), W))

new_W = W.assign(W - eta*grad_W)
new_A = A.assign(A - eta*grad_A)            
train_step = [new_W, new_A]
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''def updatefa(A,W):    
    #gradients for a loss L 
    #Update
    #w_1=tf.subtract(w_1,tf.multiply(eta,tf.transpose(dLw1)))
    #w_2=tf.subtract(w_2,tf.multiply(eta,tf.transpose(dLw2))) 
    new_A = A.assign(A-tf.multiply(eta,grad_A))
    new_W = W.assign(W-tf.multiply(eta,grad_W)) 
    return new_A, new_W 
'''
init = tf.global_variables_initializer()
iteration= 100000
epoch=10
store_al=np.zeros((epoch,iteration))
store_err=np.zeros((epoch,iteration))
store_acc=np.zeros((epoch,iteration))
with tf.Session() as sess:
    sess.run(init)
    for epoch_no in range(epoch):
        for idx in range(iteration):
            (train_x, train_y) = mnist.train.next_batch(batch_size)

            #sess.run(updatefa(A,W), feed_dict={x: train_x, y: train_y})
            _,err,align,acc = sess.run([train_step,loss,alignment,accuracy], feed_dict={x: train_x, y: train_y})
            if np.isnan(err)==True:
                print("\n\tModel does not converge!!!\n")
                break        
            store_err[epoch_no,idx]=err
            store_al[epoch_no,idx]=align
            store_acc[epoch_no,idx]=acc
        print("Run no: %d completed"%epoch_no)
        
with open("MNIST-FA.pkl",'wb') as f:
    pickle.dump([store_err,store_al,store_acc,iteration,epoch],f)

