#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf

from data_loader.data_generator import MNISTDataGenerator, LinearDataGenerator
from models.sfmodels import BPModel, FAModel, FAModelLinear, DirectFAModel4, BPModel4, FAModel4, AEFAModel, AEBPModel,\
                            AEDFAModel, BPModel10, FAModel10,\
                            FAModel4Linear
from trainers.sf_trainer import SFTrainer, AESFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.logger import LoggerNumpy, Logger
from models.npmodels import NPModel, NPModel4, DirectNPModel4, AEDFANPModel, AENPModel
import shutil
import numpy.random as rng
import numpy as np







def crossvalidate(Model,Data,Trainer,model_name,rmdirs,N):
        
    first_rates=np.logspace(-3,-6,num=6)
    second_rates=np.logspace(-3,-6,num=6)
    print("The following are the outer learning rates:",first_rates)
    print("The following are the inner learning rates:",second_rates)
    count=0
    #allocate GPU memory 
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    for eta1 in first_rates:
        print("Current outer learning rate:",eta1)
        config = process_config('./configs/np_optimized.json', model_name)
        config.learning_rate=eta1
        for eta2 in second_rates:
        #print("this is config:\n",config)
            print("\nInner learning rate:",eta2)
            config.lmda_learning_rate=eta2
            #Remove summary dir, but not hyperparams               
            #config.num_epochs=5
            if rmdirs:
                try:

                    #shutil.rmtree(config.summary_dir + '/test/')
                    #shutil.rmtree(config.summary_dir + '/train/')
                    shutil.rmtree(config.checkpoint_dir)
                except (OSError, FileNotFoundError) :
                    print ('an error')
                    pass 
            create_dirs([config.summary_dir, config.checkpoint_dir])
            model = Model(config)
            print("model:",model)
            
            
            #mem_config = tf.ConfigProto(device_count={'CPU': 4})
            #mem_config.intra_op_parallelism_threads = 1
            #mem_config.inter_op_parallelism_threads = 1
            
            for idx in range(N):
                print ('Running %s, iteration %d/%d'%(model_name, idx+1, N))
                #with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4,
                  #inter_op_parallelism_threads=4)) as sess:
                with tf.Session() as sess:
                    model.load(sess)
                    data = Data(config)
                    logger = Logger(sess, config)
                    trainer = Trainer(sess, model, data, config, logger)
                    #print("this is trainer:\n",trainer)
                    try:
                        trainer.train()
                    except ValueError:
                        print("Method fails to converge for these parameters")   
            count=count+1


    
def main():
    print("MaIn bEgiNs hErE!!!")
    print(FAModel4)
    args = get_args()
    model_name = args.modelname
    N = args.nreps
    rmdirs = args.rmdirs

    if model_name == 'backprop':
        Model = BPModel
        Data = MNISTDataGenerator
        Trainer = SFTrainer
    elif model_name=='nodepert':
        Model=NPModel
        Data=MNISTDataGenerator
        Trainer=SFTrainer
    elif model_name=='nodepert4':
        Model=NPModel4
        Data=MNISTDataGenerator
        Trainer=SFTrainer
    elif model_name=='directnodepert4':
        Model=DirectNPModel4
        Data=MNISTDataGenerator
        Trainer=SFTrainer
    elif model_name=='nodepert_autoencoder':
        Model=AENPModel
        Data=MNISTDataGenerator
        Trainer=AESFTrainer
    elif model_name=='nodepert_dfa_autoencoder':
        Model=AEDFANPModel
        Data=MNISTDataGenerator
        Trainer=AESFTrainer
    
    crossvalidate(Model,Data,Trainer,model_name,rmdirs,N)
    
    
if __name__ == '__main__':
    main()