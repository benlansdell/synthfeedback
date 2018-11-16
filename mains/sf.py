import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
#os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf

from data_loader.data_generator import MNISTDataGenerator, LinearDataGenerator
from models.sfmodels import BPModel, FAModel, FAModelLinear, DirectFAModel4, FAModel4, AEFAModel,\
                            BPModel10, FAModel10, AEDFAModel, AEBPModel
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
model_name = 'feedbackalignment4'

if model_name == 'feedbackalignment':
    Model = FAModel
    Data = MNISTDataGenerator
    Trainer = SFTrainer
elif model_name == 'feedbackalignment4':
    Model = FAModel4
    Data = MNISTDataGenerator
    Trainer = SFTrainer
elif model_name == 'directfeedbackalignment':
    Model = DirectFAModel4
    Data = MNISTDataGenerator
    Trainer = SFTrainer
elif model_name == 'backprop':
    Model = BPModel
    Data = MNISTDataGenerator
    Trainer = SFTrainer
elif model_name == 'backprop_slower':
    Model = BPModel
    Data = MNISTDataGenerator
    Trainer = SFTrainer
elif model_name == 'backprop10':
    Model = BPModel10
    Data = MNISTDataGenerator
    Trainer = SFTrainer
elif model_name == 'feedbackalignment10':
    Model = FAModel10
    Data = MNISTDataGenerator
    Trainer = SFTrainer
elif model_name == 'feedbackalignment_linear':
    Model = FAModelLinear
    Data = LinearDataGenerator
    Trainer = SFTrainer
elif model_name == 'feedbackalignment_autoencoder':
    Model = AEFAModel
    Data = MNISTDataGenerator
    Trainer = AESFTrainer
elif model_name == 'directfeedbackalignment_autoencoder':
    Model = AEDFAModel
    Data = MNISTDataGenerator
    Trainer = AESFTrainer
elif model_name == 'backprop_autoencoder':
    Model = AEBPModel
    Data = MNISTDataGenerator
    Trainer = AESFTrainer

config = process_config('./configs/sf_optimized.json', model_name)
create_dirs([config.summary_dir, config.checkpoint_dir])
sess = tf.Session()
model = Model(config)
model.load(sess)
data = Data(config)
#logger = LoggerNumpy(sess, config, model)
logger = Logger(sess, config)
#logger = None
trainer = Trainer(sess, model, data, config, logger)
#trainer = AESFTrainer(sess, model, data, config, logger)
trainer.train()

#if __name__ == '__main__':
#    main()