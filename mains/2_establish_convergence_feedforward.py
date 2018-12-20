#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf

from data_loader.data_generator import MNISTDataGenerator, LinearDataGenerator
from models.npmodels import NPModel, NPModel4, DirectNPModel4, NPModel4_ExactLsq
from trainers.sf_trainer import SFTrainer, AESFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.logger import LoggerNumpy

import numpy.random as rng
import numpy as np

def set_hyperparameters(config, attr, vals):
    for idx, val in enumerate(vals):
        setattr(config, attr[idx], val)

#The exact model appears to behave more as expected... will use it instead.

def main():

    args = get_args()

    #Select models:
    model_name = 'nodepert4_fixedw_exact'

    #Model = NPModel4
    Model = NPModel4_ExactLsq
    Data = MNISTDataGenerator
    Trainer = SFTrainer

    config = process_config('./configs/np.json', model_name)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    #Param search parameters
    attr = ['var_xi']
    #var_vals = [1e-3, 1e-2, 1e-1, 1, 10]
    var_vals = [1e-1]
    N = len(var_vals)
    #M = 10
    M = 1
    T = config.num_epochs

    test_losses = np.zeros((N, M))
    isnan = np.zeros((N, M))

    norm_err1 = np.zeros((N, M, T))
    norm_err2 = np.zeros((N, M, T))
    alignment1 = np.zeros((N, M, T))
    alignment2 = np.zeros((N, M, T))

    n = m = 0
    #for n in range(N):
    var_val = [var_vals[n]]
    set_hyperparameters(config, attr, var_val)
    model = Model(config)
    data = Data(config)
        #print('Iter: %d Hyperparameters: '%n + ' '.join([attr[i] + ' = %e'%hyperparam[i]\
        # for i in range(M)]))
    #    for m in range(M):
    with tf.Session() as sess:
        logger = LoggerNumpy(sess, config, model)
        model.load(sess)
        trainer = Trainer(sess, model, data, config, logger)
        try:
            trainer.train()
        except ValueError:
            print("Method fails to converge for these parameters")
            isnan[n,m] = 1
        loss, acc = trainer.test()
        metrics = logger.get_data()
        test_losses[n,m] = loss
        #Compute and store ||W - B|| and alignment

    #Save data for analysis
    #fn = os.path.join(config.summary_dir) + "2_establish_convergence_feedforward_output.npz"
    #np.savez(fn, test_losses=test_losses, hyperparams=hyperparams, attr = attr,
    #    ranges = ranges, log10_scale = log10_scale, isnan = isnan)
    return metrics

metrics = main()
print(metrics[:,4:])

#if __name__ == '__main__':
#    main()