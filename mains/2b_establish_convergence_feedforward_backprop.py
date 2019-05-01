#!/usr/bin/env ipython
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy.random as rng
import numpy as np
import pickle

from data_loader.data_generator import MNISTDataGenerator
from models.sfmodels import BPModel4_Small, BPModel4
from trainers.sf_trainer import SFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.logger import LoggerNumpy

def main():
    args = get_args()
    model_name = 'backprop4_small'
    Model = BPModel4_Small
    Data = MNISTDataGenerator
    Trainer = SFTrainer

    config = process_config('./configs/sf.json', model_name)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    N = 1
    M = 5
    T = config.num_epochs+1

    n_tags = 4
    test_losses = np.zeros((N, M))
    isnan = np.zeros((N, M))
    metrics = np.zeros((N, M, T, n_tags))

    n = 0
    model = Model(config)
    data = Data(config)
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
            metrics[n,m,:,:] = metric

    #Save after each run
    fn = os.path.join(config.summary_dir) + "2b_establish_convergence_feedforward_backprop_output.npz"
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