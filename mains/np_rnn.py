import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf

from data_loader.data_generator import SmallXORDataGenerator
from models.rnnmodels import BPTTModel#, FARNNModel, NPRNNModel, 
from trainers.rnn_trainer import RNNTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import LoggerNumpy, Logger
from utils.utils import get_args

#Select models:
model_name = 'bptt_short'
Model = BPTTModel
Data = SmallXORDataGenerator
Trainer = RNNTrainer

#if model_name == 'bptt':
#    Model = BPTTModel
#    Data = XORDataGenerator
#    Trainer = RNNTrainer
#elif model_name == 'fa':
#    Model = FARNNModel
#    Data = XORDataGenerator
#    Trainer = RNNTrainer
#elif model_name == 'nodepert':
#    Model = NPRNNModel
#    Data = XORDataGenerator
#    Trainer = RNNTrainer

config = process_config('./configs/rnn.json', model_name)
create_dirs([config.summary_dir, config.checkpoint_dir])
sess = tf.Session()
model = Model(config)
model.load(sess)
data = Data(config)
#logger = LoggerNumpy(sess, config, model)
logger = Logger(sess, config)
trainer = Trainer(sess, model, data, config, logger)
trainer.train()

loss, pred, x, y = trainer.test()



#Test on 

#Test XOR data generator
#next(data.next_batch())