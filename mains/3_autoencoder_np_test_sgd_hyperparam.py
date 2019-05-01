#!/usr/bin/env ipython
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

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
    model_name = 'nodepert_ae5_sgd'
    #model_name = 'nodepert_ae5_bpauto'
    #model_name = 'nodepert_ae5_bpself'
    #model_name = 'nodepert_ae5_faauto'
    #model_name = 'nodepert_ae5_faself'
    Model = AENPModel5
    #Model = AENPModel5_ExactLsq
    #Model = AENPModel5_ExactLsq_BPAuto
    #Model = AENPModel5_ExactLsq_BPSelf
    #Model = AENPModel5_ExactLsq_FAAuto
    #Model = AENPModel5_ExactLsq_FASelf
    Data = MNISTDataGenerator
    Trainer = AESFTrainer

    config = process_config('./configs/np.json', model_name)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    #Param search parameters
    attr = ['var_xi', 'learning_rate', 'lmda_learning_rate']
    attr_ranges = [[-4, -1], [-6,-3], [-6, -3]]
    log_scale = [True, True, True]
    N = 20
    #M = 5
    M = 1
    T = config.num_epochs+1
    n_tags = 13
    test_losses = np.zeros((N, M))
    isnan = np.zeros((N, M))
    metrics = np.zeros((N, M, T, n_tags))
    params = []

    for n in range(N):
        param = set_random_hyperparameters(config, attr, attr_ranges, log_scale)
        params.append(param)
        tf.reset_default_graph()
        model = Model(config)
        data = Data(config)
        print('Hyperparameters: ' + ' '.join([attr[ii] + ' = %f'%param[ii] for ii in range(len(attr))]))
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
        fn = os.path.join(config.summary_dir) + "3_autoencoder_correctbatch_hyperparam.npz"
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