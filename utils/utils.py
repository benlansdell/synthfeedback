import argparse

import tensorflow as tf

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def matmul_fa_l(x_f, y_f, x_b):
	return tf.matmul(x_b,y_f) + tf.stop_gradient(tf.matmul(x_f,y_f) - tf.matmul(x_b,y_f))

def matmul_fa_r(x_f, y_f, y_b):
	return tf.matmul(x_f,y_b) + tf.stop_gradient(tf.matmul(x_f,y_f) - tf.matmul(x_f,y_b))

def matmul_fa(x_f, y_f, b):
	if b.shape == x_f.shape:
		return matmul_fa_l(x_f, y_f, b)
	elif b.shape == y_f.shape:
		return matmul_fa_r(x_f, y_f, b)
	else:
		raise ValueError