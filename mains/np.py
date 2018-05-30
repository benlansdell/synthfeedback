import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf

from data_loader.data_generator import MNISTDataGenerator, LinearDataGenerator
from models.npmodels import NPModel, NPModel4, DirectNPModel4, AENPModel, AEDFANPModel#, AEFAModel
from trainers.sf_trainer import SFTrainer, AESFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import LoggerNumpy, Logger
from utils.utils import get_args

#def main():
#    #Load config file from command line
#    try:
#        args = get_args()
#        config = process_config(args.config)
#    except:
#        print("missing or invalid arguments")
#        exit(0)

#Select models:
model_name = 'nodepert_dfa_autoencoder'

if model_name == 'nodepert':
    Model = NPModel
    Data = MNISTDataGenerator
    Trainer = SFTrainer
elif model_name == 'nodepert4':
    Model = NPModel4
    Data = MNISTDataGenerator
    Trainer = SFTrainer
elif model_name == 'directnodepert4':
    Model = DirectNPModel4
    Data = MNISTDataGenerator
    Trainer = SFTrainer
elif model_name == 'nodepert_sanity':
    Model = NPModel4
    Data = MNISTDataGenerator
    Trainer = SFTrainer
elif model_name == 'nodepert_autoencoder':
    Model = AENPModel
    Data = MNISTDataGenerator
    Trainer = AESFTrainer
elif model_name == 'nodepert_dfa_autoencoder':
    Model = AEDFANPModel
    Data = MNISTDataGenerator
    Trainer = AESFTrainer
#elif model_name == 'directfeedbackalignment':
#    Model = DirectFAModel4
#    Data = MNISTDataGenerator
#elif model_name == 'backprop':
#    Model = BPModel
#    Data = MNISTDataGenerator
#elif model_name == 'feedbackalignment_linear':
#    Model = FAModelLinear
#    Data = LinearDataGenerator
#elif model_name == 'feedbackalignment_autoencoder':
#    Model = AEFAModel
#    Data = MNISTDataGenerator

config = process_config('./configs/np.json', model_name)
create_dirs([config.summary_dir, config.checkpoint_dir])
sess = tf.Session()
model = Model(config)
model.load(sess)
data = Data(config)
#logger = LoggerNumpy(sess, config, model)
logger = Logger(sess, config)
trainer = Trainer(sess, model, data, config, logger)
trainer.train()

#if __name__ == '__main__':
#    main()