from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf

class SFTrainer(BaseTrain):
    #def __init__(self, sess, model, data, config, logger, options, run_metadata):
    def __init__(self, sess, model, data, config, logger):
        super(SFTrainer, self).__init__(sess, model, data, config, logger)

        #self.options = options
        #self.run_metadata = run_metadata

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        losses_test = []
        accs_test = []
        for _ in loop:
            loss, acc = self.train_step()
            loss_test, acc_test = self.test()
            losses.append(loss)
            accs.append(acc)
            losses_test.append(loss_test)
            accs_test.append(acc_test)

        loss = np.mean(losses)
        acc = np.mean(accs)
        loss_test = np.mean(losses_test)
        acc_test = np.mean(accs_test)

        #Check for convergence issues...
        for x in self.model.trainable:
            if np.isnan(self.sess.run(x)).any():
                raise ValueError("nan encountered. Model does not converge.")
        if np.isnan(loss):
            raise ValueError("nan encountered. Model does not converge.")

        metric_tags, metrics = self.training_metrics()

        cur_ep = self.model.cur_epoch_tensor.eval(self.sess)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
            'loss_test': loss_test,
            'acc_test': acc_test
        }

        for idx in range(len(metrics)):
            #summaries_dict['metrics'] = np.array(metrics)
            summaries_dict[metric_tags[idx]] = metrics[idx]

        print("Epoch: %d Train loss: %f Train accuracy: %f Test loss: %f Test accuracy: %f"%(cur_ep, loss, acc, loss_test, acc_test))
        if self.logger:
            self.logger.summarize(cur_ep, summaries_dict=summaries_dict)
            self.model.save(self.sess)

    def train_step(self):

        curr_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        train_step = self.model.train_step

        if hasattr(self.config, 'warmup_epoch'):
            if curr_epoch < self.config.warmup_epoch:
                train_step = self.model.train_step_warmup

        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, \
                                                self.model.is_training: True}
        _, loss, acc = self.sess.run([train_step, self.model.loss,\
                                    self.model.accuracy], feed_dict=feed_dict)

        return loss, acc

    def test(self):
        batch_x, batch_y = next(self.data.test_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, \
                                                self.model.is_training: True}
        loss, acc = self.sess.run([self.model.loss,\
                                    self.model.accuracy], feed_dict=feed_dict)
        return loss, acc

class AESFTrainer(SFTrainer):
    def __init__(self, sess, model, data, config, logger):
        super(AESFTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        losses_test = []
        accs_test = []
        for _ in loop:
            loss, acc = self.train_step()
            loss_test, acc_test = self.test()
            losses.append(loss)
            accs.append(acc)
            losses_test.append(loss_test)
            accs_test.append(acc_test)

        loss = np.mean(losses)
        acc = np.mean(accs)
        loss_test = np.mean(losses_test)
        acc_test = np.mean(accs_test)

        #Check for convergence issues...
        for x in self.model.trainable:
            if np.isnan(self.sess.run(x)).any():
                raise ValueError("nan encountered. Model does not converge.")
        if np.isnan(loss):
            raise ValueError("nan encountered. Model does not converge.")
        metric_tags, metrics = self.training_metrics()
        ae_input, ae_output = self.images()

        cur_ep = self.model.cur_epoch_tensor.eval(self.sess)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
            #'input': ae_input,
            #'output': ae_output,
            'loss_test': loss_test,
            'acc_test': acc_test
        }
        for idx in range(len(metrics)):
            #summaries_dict['metrics'] = np.array(metrics)
            summaries_dict[metric_tags[idx]] = metrics[idx]

        print("Epoch: %d Train loss: %f Train accuracy: %f Test loss: %f Test accuracy: %f"%(cur_ep, loss, acc, loss_test, acc_test))
        if self.logger:
            self.logger.summarize(cur_ep, summaries_dict=summaries_dict)
            self.model.save(self.sess)

    def images(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_x, \
                                                self.model.is_training: True}
        ae_output = self.sess.run(self.model.y_p, feed_dict=feed_dict)
        return batch_x.reshape((-1, 28, 28, 1)), ae_output.reshape((-1,28,28,1))
        #np.zeros((1,2,2,1)), np.zeros((1,2,2,1))#batch_x[0,:], ae_output[0,:].reshape((28,28))

    def train_step(self):
        curr_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        train_step = self.model.train_step

        if hasattr(self.config, 'warmup_epoch'):
            if curr_epoch < self.config.warmup_epoch:
                train_step = self.model.train_step_warmup

        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_x, \
                                                self.model.is_training: True}
        _, loss, acc = self.sess.run([train_step, self.model.loss,\
                                    self.model.accuracy], feed_dict=feed_dict)
        return loss, acc

    def training_metrics(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_x, self.model.is_training: True}
        return self.model.training_metric_tags, self.sess.run(self.model.training_metrics, feed_dict=feed_dict)

    def test(self):
        batch_x, batch_y = next(self.data.test_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_x, self.model.is_training: True}
        losses, accs = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed_dict)
        loss = np.mean(losses)
        acc = np.mean(accs)
        return loss, acc