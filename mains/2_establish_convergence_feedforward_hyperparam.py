#!/usr/bin/env ipython
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#python -m cProfile -o 2_profile.txt ./mains/2_establish_convergence_feedforward.py

import tensorflow as tf
import numpy.random as rng
import numpy as np
import pickle

from data_loader.data_generator import MNISTDataGenerator
from models.npmodels import NPModel4_ExactLsq, NPModel4_ExactLsq_CorrectBatch
from trainers.sf_trainer import SFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.logger import LoggerNumpy

import cProfile
import re

#Add tensorflow profiling
from tensorflow.python.client import timeline

def set_hyperparameters(config, attr, vals):
    for idx, val in enumerate(vals):
        setattr(config, attr[idx], val)

def set_random_hyperparameters(config, attrs, ranges, log_scale):
    params = []
    for idx, attr in enumerate(attrs):
        val = rng.rand()*(ranges[idx][1]-ranges[idx][0])+ranges[idx][0]
        if log_scale[idx]:
            val = np.power(10, val)
        setattr(config, attrs[idx], val)
        params.append(val)
    return params

def main():
    args = get_args()
    model_name = 'nodepert4_fixedw_exact_largegamma'
    Model = NPModel4_ExactLsq_CorrectBatch
    Data = MNISTDataGenerator
    Trainer = SFTrainer

    config = process_config('./configs/np.json', model_name)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    #Param search parameters
    attr = ['var_xi', 'gamma']
    #print(attr[1])
    attr_ranges = [[-4, -1], [-3,3]]
    #var_vals = [1e-4, 1e-3, 1e-2, 1e-1]
    log_scale = [True, True]
    #var_vals = [1e-1]
    N = 20
    #N = len(var_vals)
    M = 2
    T = config.num_epochs+1

    n_tags = 10
    test_losses = np.zeros((N, M))
    isnan = np.zeros((N, M))
    metrics = np.zeros((N, M, T, n_tags))
    params = []

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True

    for n in range(N):
        #var_val = [var_vals[n]]
        param = set_random_hyperparameters(config, attr, attr_ranges, log_scale)
        params.append(param)
        tf.reset_default_graph()
        model = Model(config)
        data = Data(config)
        #print(param)
        print('Hyperparameters: ' + ' '.join([attr[ii] + ' = %f'%param[ii] for ii in range(len(attr))]))
        for m in range(M):
            with tf.Session(config=tfconfig) as sess:
                #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #run_metadata = tf.RunMetadata()
                logger = LoggerNumpy(sess, config, model)
                model.load(sess)
                #trainer = Trainer(sess, model, data, config, logger, options, run_metadata)
                trainer = Trainer(sess, model, data, config, logger)
                try:
                    trainer.train()
                except ValueError:
                    print("Method fails to converge for these parameters")
                    isnan[n,m] = 1
                loss, acc = trainer.test()
                metric = logger.get_data()
                tags = logger.get_tags()
                test_losses[n,m] = loss
                metrics[n,m,:,:] = metric

                #See here: https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d
                #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                #with open('./timeline_02_n_%d_m_%d.json'%(n,m), 'w') as f:
                #    f.write(chrome_trace)

        #Save after each run
        fn = os.path.join(config.summary_dir) + "2_establish_convergence_feedforward_output_correctbatches_hyperparamsearch.npz"
        to_save = {
            'attr': attr,
            'params': params,
            'test_losses': test_losses,
            'metrics': metrics,
            'isnan': isnan,
            'tags': tags
        }
        pickle.dump(to_save, open(fn, "wb"))

    return metrics

if __name__ == '__main__':
    
    main()