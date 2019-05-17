#!/usr/bin/env ipython
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy.random as rng
import numpy as np
import pickle
from data_loader.data_generator import MNISTDataGenerator
from models.npmodels import AENPModel5_ExactLsq_BPAuto, AENPModel5_ExactLsq_FAAuto, AENPModel5
from trainers.sf_trainer import AESFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.logger import LoggerNumpy, Logger

def set_hyperparameters(config, attr, vals):
    for idx, val in enumerate(vals):
        setattr(config, attr[idx], val)

def main():
    args = get_args()
    #model_name = 'nodepert_ae5_sgd_correctgeom'
    model_name = 'nodepert_ae5_faauto'
    #model_name = 'nodepert_ae5_bpauto'

    #Model = AENPModel5
    #Model = AENPModel5_ExactLsq
    #Model = AENPModel5_ExactLsq_BPAuto
    #Model = AENPModel5_ExactLsq_BPSelf
    Model = AENPModel5_ExactLsq_FAAuto
    #Model = AENPModel5_ExactLsq_FASelf
    Data = MNISTDataGenerator
    Trainer = AESFTrainer

    config = process_config('./configs/np_optimized.json', model_name)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    #var_vals = [1e-2]
    N = 1
    #M = 5
    M = 10
    T = config.num_epochs+1
    n_tags = 13
    test_losses = np.zeros((N, M))
    isnan = np.zeros((N, M))
    metrics = np.zeros((N, M, T, n_tags))

    for n in range(N):
        tf.reset_default_graph()
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
        fn = os.path.join(config.summary_dir) + "3_autoencoder_correctbatch.npz"
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