#!/usr/bin/env ipython
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#python -m cProfile -o 2_profile.txt ./mains/2_establish_convergence_feedforward.py

import tensorflow as tf
import numpy.random as rng
import numpy as np
import pickle

from data_loader.data_generator import MNISTDataGenerator
from models.wmmodels import WMModel4
from trainers.sf_trainer import SFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.logger import LoggerNumpy
import cProfile
import re

#Add tensorflow profiling
from tensorflow.python.client import timeline

def set_random_hyperparameters(config, attrs, ranges, log_scale):
    params = []
    for idx, attr in enumerate(attrs):
        val = rng.rand()*(ranges[idx][1]-ranges[idx][0])+ranges[idx][0]
        if log_scale[idx]:
            val = np.power(10, val)
        setattr(config, attrs[idx], val)
        params.append(val)
        print(attr,params)
    return params

def set_hyperparameters(config, attr, vals):
    for idx, val in enumerate(vals):
        setattr(config, attr[idx], val)
def main():
    args = get_args()
    model_name = 'WM_4layer'
    Model = WMModel4
    Data = MNISTDataGenerator
    Trainer = SFTrainer

    config = process_config('/home/prashanth/synthfeedback/configs/wm.json', model_name)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    #Param search parameters
    #attr = ['']
    #var_vals = [1e-3, 1e-2, 1e-1, 1, 10]
    #var_vals = [30,100]
    #learning_Rate=1e-1 to 1e-5
    #lmda=0.1 to .99
    attr = ['learning_rate','learning_rate_b','lamba']
    attr_ranges = [[-1, -5], [-1,-5], [-2, -1]]
    log_scale = [True, True, True] 
    N = 3
    M = 1
    #config.num_epoch=5
    #print("\n\n",config.num_epoch)
    T = config.num_epochs+1
    #config.gamma=250000
    
    n_tags = 10
    test_losses = np.zeros((N, M))
    isnan = np.zeros((N, M))
    metrics = np.zeros((N, M, T, n_tags))

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    rando=dict()
    learning_rate=[]
    learning_rate_b=[]
    lamba=[]
    for n in range(N):
        param = set_random_hyperparameters(config, attr, attr_ranges, log_scale) 
        print("Look Here",param)

        learning_rate.append(param[0])
        learning_rate_b.append(param[1])
        lamba.append(param[2])
    rando['lamba']=lamba
    rando['l']=learning_rate
    rando['l_b']=learning_rate_b
    with open("Ignore.p","wb") as f:
        pickle.dump([rando],f)
        #model = Model(config)
        ##print("\n\n",config.num_epoch)
        #data = Data(config)
         #   #print('Hyperparameters: ' + attr[0] + ' = %f'%var_vals[n])
        #for m in range(M):
          #  with tf.Session(config=tfconfig) as sess:
           #     #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #    #run_metadata = tf.RunMetadata()
             #   logger = LoggerNumpy(sess, config, model)
              #  model.load(sess)
               # #trainer = Trainer(sess, model, data, config, logger, options, run_metadata)
                #trainer = Trainer(sess, model, data, config, logger)
                #try:
                 #   trainer.train()
               # except ValueError:
                #    print("Method fails to converge for these parameters")
                #    isnan[n,m] = 1
                #loss, acc = trainer.test()
                        #metric = logger.get_data()
                #tags = logger.get_tags()
                #test_losses[n,m] = loss
                #metrics[n,m,:,:] = metric

            #    #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
             #   #chrome_trace = fetched_timeline.generate_chrome_trace_format()
              #  #with open('./timeline_02_n_%d_m_%d.json'%(n,m), 'w') as f:
               # #    f.write(chrome_trace)

   #     #Save after each run
    #    fn = os.path.join(config.summary_dir) + "WM_4layer(condnum).npz"
     #   to_save = {
      #      'test_losses': test_losses,
       #     'metrics': metrics,
        #    'isnan': isnan,
         #   'tags': tags,
          #  'params':param
       # }
        #pickle.dump(to_save, open(fn, "wb"))

    #np.savez(fn, test_losses=test_losses, metrics = metrics, isnan = isnan, tags = tags)
    #return metrics

if __name__ == '__main__':
    main()