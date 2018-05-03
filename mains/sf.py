import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf

from data_loader.data_generator import MNISTDataGenerator, LinearDataGenerator
from models.sfmodels import BPModel, FAModel, FAModelLinear, DirectFAModel4, FAModel4
from trainers.sf_trainer import SFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import LoggerNumpy
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
model_name = 'directfeedbackalignment'

if model_name == 'feedbackalignment':
	Model = FAModel
	Data = MNISTDataGenerator
elif model_name == 'feedbackalignment4':
	Model = FAModel4
	Data = MNISTDataGenerator
elif model_name == 'directfeedbackalignment':
	Model = DirectFAModel4
	Data = MNISTDataGenerator
elif model_name == 'backprop':
	Model = BPModel
	Data = MNISTDataGenerator
elif model_name == 'feedbackalignment_linear':
	Model = FAModelLinear
	Data = LinearDataGenerator

config = process_config('./configs/sf.json', model_name)
create_dirs([config.summary_dir, config.checkpoint_dir])
sess = tf.Session()
model = Model(config)
model.load(sess)
data = Data(config)
logger = LoggerNumpy(sess, config, model)
trainer = SFTrainer(sess, model, data, config, logger)
trainer.train()

#if __name__ == '__main__':
#    main()