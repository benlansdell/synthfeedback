#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf

from data_loader.data_generator import MNISTDataGenerator, LinearDataGenerator
from models.sfmodels import BPModel, FAModel, FAModelLinear, DirectFAModel4, \
														FAModel4, AEFAModel
from trainers.sf_trainer import SFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

import numpy.random as rng
import numpy as np

def select_random_hyperparameters(config, attr, ranges, log_scale):
	params = np.zeros(len(attr))
	for idx in range(len(params)):
		val = (ranges[idx][1]-ranges[idx][0])*rng.rand()+ranges[idx][0]
		if log_scale[idx]:
			val = np.power(10, val)
		params[idx] = val
		setattr(config, attr[idx], val)
	return params

def main():
	#Load config file from command line
	try:
		args = get_args()
		config = process_config(args.config)
		model_name = process_config(args.modelname)
		#Select models:
		#model_name = 'feedbackalignment'
	except:
		print("Missing or invalid arguments.")
		exit()

	model_name = 'backprop4'

	if model_name == 'feedbackalignment':
		Model = FAModel
		Data = MNISTDataGenerator
	elif model_name == 'feedbackalignment4':
		Model = FAModel4
		Data = MNISTDataGenerator
	elif model_name == 'directfeedbackalignment':
		Model = DirectFAModel4
		Data = MNISTDataGenerator
	elif model_name == 'backprop':
		Model = BPModel
		Data = MNISTDataGenerator
	elif model_name == 'backprop4':
		Model = BPModel4
		Data = MNISTDataGenerator
	elif model_name == 'feedbackalignment_linear':
		Model = FAModelLinear
		Data = LinearDataGenerator
	elif model_name == 'feedbackalignment_autoencoder':
		Model = AEFAModel
		Data = MNISTDataGenerator

	config = process_config('./configs/sf.json', model_name)
	create_dirs([config.summary_dir, config.checkpoint_dir])

	#Param search parameters
	attr = ['learning_rate']
	N = 50
	M = len(attr)
	ranges = [[-6, -3]]
	log10_scale = [True]

	test_losses = np.zeros(N)
	hyperparams = np.zeros((N, M))
	isnan = np.zeros(N)

	for n in range(N):
		with tf.Session() as sess:	
			sess = tf.Session()
			hyperparam = select_random_hyperparameters(config, attr, ranges,
																log10_scale)
			model = Model(config)
			model.load(sess)
			data = Data(config)
			trainer = SFTrainer(sess, model, data, config, None)
			print 'Hyperparameters: ' + ' '.join([attr[i] + ' = %e'%hyperparam[i]\
			 for i in range(M)])
			try:
				trainer.train()
			except ValueError:
				print "Method fails to converge for these parameters"
				isnan[n] = 1
			loss, acc = trainer.test()
			hyperparams[n,:] = hyperparam
			test_losses[n] = loss

	#Save data for analysis
	fn = os.path.join(config.summary_dir) + "hyperparameter_search.npz"
	np.savez(fn, test_losses=test_losses, hyperparams=hyperparams, attr = attr,
		ranges = ranges, log10_scale = log10_scale, isnan = isnan)

if __name__ == '__main__':
	main()