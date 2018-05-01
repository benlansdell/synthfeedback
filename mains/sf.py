import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.sfmodels import BPModel
from trainers.sf_trainer import SFTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    create_dirs([config.summary_dir, config.checkpoint_dir])
    sess = tf.Session()
    model = BPModel(config)
    model.load(sess)
    data = DataGenerator(config)
    logger = Logger(sess, config)
    trainer = SFTrainer(sess, model, data, config, logger)
    trainer.train()

if __name__ == '__main__':
    main()