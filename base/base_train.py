import tensorflow as tf
import numpy as np

class BaseTrain(object):
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

        #Save logger info if defined
        if self.logger:
            save_fn = getattr(self.logger, 'save', None)
            if callable(save_fn):
                save_fn()

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def training_metrics(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        return self.model.training_metric_tags, self.sess.run(self.model.training_metrics, feed_dict=feed_dict)

    def test(self):
        batch_x, batch_y = next(self.data.test_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        losses, accs = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed_dict)
        loss = np.mean(losses)
        acc = np.mean(accs)
        return loss, acc
