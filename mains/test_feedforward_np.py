#!/usr/bin/env ipython
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy.random as rng
import numpy as np
import pickle

from data_loader.data_generator import MNISTDataGenerator
from models.npmodels import NPModel4_ExactLsq
from trainers.sf_trainer import SFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.logger import LoggerNumpy

def set_hyperparameters(config, attr, vals):
    for idx, val in enumerate(vals):
        setattr(config, attr[idx], val)

def main():
    args = get_args()
    model_name = 'nodepert4_fixedw_exact'
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
    M = 1
    T = config.num_epochs+1

    n_tags = 8
    test_losses = np.zeros((N, M))
    isnan = np.zeros((N, M))
    metrics = np.zeros((N, M, T, n_tags))

    for n in range(N):
        var_val = [var_vals[n]]
        set_hyperparameters(config, attr, var_val)
        model = Model(config)
        data = Data(config)
        print('Hyperparameters: ' + attr[0] + ' = %f'%var_vals[n])
        for m in range(M):
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
                metric = logger.get_data()
                tags = logger.get_tags()
                test_losses[n,m] = loss
                metrics[n,m,:] = metric

    fn = os.path.join(config.summary_dir) + "2_establish_convergence_feedforward_output.npz"
    to_save = {
        'test_losses': test_losses,
        'metrics': metrics,
        'isnan': isnan,
        'tags': tags
    }
    pickle.dump(to_save, open(fn, "wb"))

    #np.savez(fn, test_losses=test_losses, metrics = metrics, isnan = isnan, tags = tags)
    return metrics

metrics = main()

if __name__ == '__main__':
    main()