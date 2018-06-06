#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf

from data_loader.data_generator import MNISTDataGenerator, LinearDataGenerator
from models.sfmodels import BPModel, FAModel, FAModelLinear, DirectFAModel4, \
														BPModel4, FAModel4, \
														AEFAModel, AEBPModel,\
														AEDFAModel, BPModel10, FAModel10,\
														FAModel4Linear
from trainers.sf_trainer import SFTrainer, AESFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.logger import LoggerNumpy, Logger

import shutil
import numpy.random as rng
import numpy as np

def main():
	args = get_args()
	model_name =args.modelname
	N = args.nreps
	rmdirs = args.rmdirs

	if model_name == 'feedbackalignment':
		Model = FAModel
		Data = MNISTDataGenerator
		Trainer = SFTrainer
	elif model_name == 'feedbackalignment4':
		Model = FAModel4
		Data = MNISTDataGenerator
		Trainer = SFTrainer
	elif model_name == 'feedbackalignment4linear':
		Model = FAModel4Linear
		Data = MNISTDataGenerator
		Trainer = SFTrainer
	elif model_name == 'feedbackalignment10':
		Model = FAModel10
		Data = MNISTDataGenerator
		Trainer = SFTrainer
	elif model_name == 'directfeedbackalignment':
		Model = DirectFAModel4
		Data = MNISTDataGenerator
		Trainer = SFTrainer
	elif model_name == 'backprop':
		Model = BPModel
		Data = MNISTDataGenerator
		Trainer = SFTrainer
	elif model_name == 'backprop4':
		Model = BPModel4
		Data = MNISTDataGenerator
		Trainer = SFTrainer
	elif model_name == 'backprop10':
		Model = BPModel10
		Data = MNISTDataGenerator
		Trainer = SFTrainer
	elif model_name == 'feedbackalignment_linear':
		Model = FAModelLinear
		Data = LinearDataGenerator
		Trainer = SFTrainer
	elif model_name == 'feedbackalignment_autoencoder':
		Model = AEFAModel
		Data = MNISTDataGenerator
		Trainer = AESFTrainer
	elif model_name == 'backprop_autoencoder':
		Model = AEBPModel
		Data = MNISTDataGenerator
		Trainer = AESFTrainer
	elif model_name == 'directfeedbackalignment_autoencoder':
		Model = AEDFAModel
		Data = MNISTDataGenerator
		Trainer = AESFTrainer

	#config = process_config('./configs/np_optimized.json', model_name)
	config = process_config('./configs/sf_optimized.json', model_name)

	#Remove summary dir, but not hyperparams
	if rmdirs:
		try:
			shutil.rmtree(config.summary_dir + '/test/')
			shutil.rmtree(config.summary_dir + '/train/')
			shutil.rmtree(config.checkpoint_dir)
		except OSError:
			print 'an error'
			pass 

	create_dirs([config.summary_dir, config.checkpoint_dir])

	model = Model(config)
	for idx in range(N):
		print 'Running %s, iteration %d/%d'%(model_name, idx+1, N)
		with tf.Session() as sess:	
			#sess = tf.Session()
			model.load(sess)
			data = Data(config)
			logger = Logger(sess, config)
			trainer = Trainer(sess, model, data, config, logger)

			try:
				trainer.train()
			except ValueError:
				print("Method fails to converge for these parameters")

if __name__ == '__main__':
	main()