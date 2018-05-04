import argparse
import numpy as np
import numpy.random as rng
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def get_args():
	argparser = argparse.ArgumentParser(description=__doc__)
	argparser.add_argument(
		'-c', '--config',
		metavar='C',
		default='None',
		help='The Configuration file')
	args = argparser.parse_args()
	return args

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

	# Need to generate a unique name to avoid duplicates:
	rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

	tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": rnd_name}):
		return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def np_matmul(A_f,x,A_b):
	return np.dot(A_f,x).astype(np.float32)

def matmul_l_grad(op, grad):
	A_f = op.inputs[0]
	x = op.inputs[1]
	A_b = op.inputs[2]
	input_rows = array_ops.shape(x)[0]
	input_cols = array_ops.shape(x)[1]
	output_rows = array_ops.shape(A_f)[0]
	
	#This is incorrect...
	#grad_A = x
	#grad_A = tf.multiply(tf.transpose(grad), tf.reshape(tf.tile(tf.reshape(x,\
	#					 [input_rows]), [output_rows]), [output_rows, -1]))*0
	grad_A = tf.matmul(grad, tf.transpose(x))

	#Backprop
	#grad_x = tf.matmul(tf.transpose(A_f), grad)
	#Feedback
	grad_x = tf.matmul(tf.transpose(A_b), grad)

	grad_Ab = grad
	
	return [grad_A, grad_x, grad_Ab]
		
def tf_matmul_l(A_f,x,A_b, name=None):
	#A_f = tf.Variable(rng.randn(31,20), name="hidden_weights", dtype=tf.float32)
	#A_b = tf.Variable(rng.randn(31,20), name="hidden_weights", dtype=tf.float32)
	#x = tf.Variable(rng.randn(31,1), name="hidden_weights", dtype=tf.float32)

	with ops.name_scope(name, "matmul_fa", [A_f, x, A_b]) as name:
		a = py_func(np_matmul,
						[A_f,x,A_b],
						[tf.float32],
						name=name,
						grad=matmul_l_grad)
		return a[0]

def tf_matmul_r(x,A_f,A_b, name=None):
	return tf.transpose(tf_matmul_l(tf.transpose(A_f), tf.transpose(x), tf.transpose(A_b)))