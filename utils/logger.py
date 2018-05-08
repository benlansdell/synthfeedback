import tensorflow as tf
import os

import numpy as np

class LoggerNumpy:
    #Save numpy arrays with data in them for later plotting, etc
    def __init__(self, sess, config, model):
        self.sess = sess
        self.config = config

        with tf.variable_scope("loss"):
            nrows = config.num_epochs+1
            ncols = len(model.training_metrics)+2
            self.data = np.zeros((nrows, ncols))

    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        if summaries_dict is not None:
            with tf.variable_scope("loss"):
                if 'metrics' in summaries_dict.keys():
                    self.data[step,:] = [summaries_dict['loss'], summaries_dict['acc']] + summaries_dict['metrics']
                else:
                    self.data[step,:] = [summaries_dict['loss'], summaries_dict['acc']]

    def save(self):
        #Save self.data
        fn = os.path.join(self.config.summary_dir) + "train.npy"
        np.save(fn, self.data)

class Logger:
    def __init__(self, sess,config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"),
                                                          self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "test"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
        with tf.variable_scope(scope):
            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]), name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()



#BATCH_SIZE = 32
#GRID_ROWS = 5
#GRID_COLS = 10
#USE_RELU = False
#
#def layer_grid_summary(name, var, image_dims):
#    prod = np.prod(image_dims)
#    grid = form_image_grid(tf.reshape(var, [BATCH_SIZE, prod]), [GRID_ROWS, 
#        GRID_COLS], image_dims, 1)
#    return tf.summary.image(name, grid)
#
#
#def create_summaries(loss, x, latent, output):
#    writer = tf.summary.FileWriter("./logs")
#    tf.summary.scalar("Loss", loss)
#    layer_grid_summary("Input", x, [28, 28])
#    layer_grid_summary("Encoder", latent, [2, 1])
#    layer_grid_summary("Output", output, [28, 28])
#    return writer, tf.summary.merge_all()
#
#
#def make_image(name, var, image_dims):
#    prod = np.prod(image_dims)
#    grid = form_image_grid(tf.reshape(var, [BATCH_SIZE, prod]), [GRID_ROWS, 
#        GRID_COLS], image_dims, 1)
#    s_grid = tf.squeeze(grid, axis=0)
#
#    # This reproduces the code in: tensorflow/core/kernels/summary_image_op.cc
#    im_min = tf.reduce_min(s_grid)
#    im_max = tf.reduce_max(s_grid)
#
#    kZeroThreshold = tf.constant(1e-6)
#    max_val = tf.maximum(tf.abs(im_min), tf.abs(im_max))
#
#    offset = tf.cond(
#            im_min < tf.constant(0.0),
#            lambda: tf.constant(128.0),
#            lambda: tf.constant(0.0)
#            )
#    scale = tf.cond(
#            im_min < tf.constant(0.0),
#            lambda: tf.cond(
#                max_val < kZeroThreshold,
#                lambda: tf.constant(0.0),
#                lambda: tf.div(127.0, max_val)
#                ),
#            lambda: tf.cond(
#                im_max < kZeroThreshold,
#                lambda: tf.constant(0.0),
#                lambda: tf.div(255.0, im_max)
#                )
#            )
#    s_grid = tf.cast(tf.add(tf.multiply(s_grid, scale), offset), tf.uint8)
#    enc = tf.image.encode_jpeg(s_grid)
#
#    fwrite = tf.write_file(name, enc)
#    return fwrite