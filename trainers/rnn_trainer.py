from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf

class RNNTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(RNNTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        losses_test = []
        for _ in loop:
            loss = self.train_step()
            loss_test = self.test()
            losses.append(loss)
            losses_test.append(loss_test)

        loss = np.mean(losses)
        loss_test = np.mean(losses_test)

        #Check for convergence issues...
        for x in self.model.trainable:
            if self.sess.run(tf.is_nan(x)).any():
                raise ValueError("nan encountered. Model does not converge.")
        if np.isnan(loss):
            raise ValueError("nan encountered. Model does not converge.")

        metric_tags, metrics = self.training_metrics()

        cur_ep = self.model.cur_epoch_tensor.eval(self.sess)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'loss_test': loss_test,
        }

        for idx in range(len(metrics)):
            #summaries_dict['metrics'] = np.array(metrics)
            summaries_dict[metric_tags[idx]] = metrics[idx]

        print("Epoch: %d Train loss: %f Train accuracy: %f Test loss: %f Test accuracy: %f"%(cur_ep, loss, loss_test))
        if self.logger:
            self.logger.summarize(cur_ep, summaries_dict=summaries_dict)
            self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch())
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, \
                                                self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_step, self.model.loss,\
                                    ], feed_dict=feed_dict)
        return loss

    def test(self):
        batch_x, batch_y = next(self.data.test_batch())
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, \
                                                self.model.is_training: True}
        loss = self.sess.run([self.model.loss,\
                                    ], feed_dict=feed_dict)
        return loss
