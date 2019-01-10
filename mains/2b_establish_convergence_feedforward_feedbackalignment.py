#!/usr/bin/env ipython
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#python -m cProfile -o 2_profile.txt ./mains/2_establish_convergence_feedforward.py

import tensorflow as tf
import numpy.random as rng
import numpy as np
import pickle

from data_loader.data_generator import MNISTDataGenerator
from models.sfmodels import FAModel4_Small, FAModel4
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
    model_name = 'feedbackalignment4_small'
    Model = FAModel4_Small
    #Model = FAModel4
    Data = MNISTDataGenerator
    Trainer = SFTrainer

    config = process_config('./configs/sf.json', model_name)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    N = 1
    M = 1
    T = config.num_epochs+1

    n_tags = 10
    test_losses = np.zeros((N, M))
    isnan = np.zeros((N, M))
    metrics = np.zeros((N, M, T, n_tags))

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True

    n = 0
    model = Model(config)
    data = Data(config)
    for m in range(M):
        with tf.Session(config=tfconfig) as sess:
            logger = LoggerNumpy(sess, config, model)
            model.load(sess)
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

    #Save after each run
    fn = os.path.join(config.summary_dir) + "2b_establish_convergence_feedforward_feedbackalignment_output.npz"
    to_save = {
        'test_losses': test_losses,
        'metrics': metrics,
        'isnan': isnan,
        'tags': tags
    }
    pickle.dump(to_save, open(fn, "wb"))

    return metrics

if __name__ == '__main__':
    
    main()