#!/usr/bin/env ipython
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#python -m cProfile -o 2_profile.txt ./mains/2_establish_convergence_feedforward.py

import tensorflow as tf
import numpy.random as rng
import numpy as np
import pickle

from data_loader.data_generator import MNISTDataGenerator
from models.npmodels import NPModel3_ExactLsq
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
def main():
    args = get_args()
    model_name = 'nodepert3_fixedw_exact'
    Model = NPModel3_ExactLsq
    Data = MNISTDataGenerator
    Trainer = SFTrainer

    config = process_config('/home/prashanth/synthfeedback/configs/np.json', model_name)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    #Param search parameters
    attr = ['neurons']
    #var_vals = [1e-3, 1e-2, 1e-1, 1, 10]
    var_vals = [30,100,300,1000]
    N = len(var_vals)
    M = 3
    T = config.num_epochs+1

    n_tags = 6
    test_losses = np.zeros((N, M))
    isnan = np.zeros((N, M))
    metrics = np.zeros((N, M, T, n_tags))

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True

    for n in range(N):
        var_val = [var_vals[n]]
        set_hyperparameters(config, attr, var_val)
        tf.reset_default_graph()
        model = Model(config)
        data = Data(config)
        print('Hyperparameters: ' + attr[0] + ' = %f'%var_vals[n])
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
                print("tags are here:",tags)
                print("metric",metric.shape)
                test_losses[n,m] = loss
                metrics[n,m,:,:] = metric

                #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                #with open('./timeline_02_n_%d_m_%d.json'%(n,m), 'w') as f:
                #    f.write(chrome_trace)

        #Save after each run
        fn = os.path.join(config.summary_dir) + "2_establish_convergence_feedforward_output_3layer.npz"
        to_save = {
            'test_losses': test_losses,
            'metrics': metrics,
            'isnan': isnan,
            'tags': tags
        }
        pickle.dump(to_save, open(fn, "wb"))

    #np.savez(fn, test_losses=test_losses, metrics = metrics, isnan = isnan, tags = tags)
    return metrics

if __name__ == '__main__':
    main()