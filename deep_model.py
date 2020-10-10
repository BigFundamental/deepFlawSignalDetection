#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import math
import tensorflow as tf
from sklearn import metrics

__all__ = ['LSTM_NER', 'Bi_LSTM_NER', 'CNN_Bi_LSTM_NER', 'DICNN_NER']

def sizeof_fmt(num, suffix='B'):
    ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def batch_index(length, batch_size, n_iter=100, shuffle=True):
    index = list(range(length))
    rd = int(length / batch_size)
    if length % batch_size != 0:
        rd += 1
    for j in range(n_iter):
        if shuffle:
            np.random.shuffle(index)
        for i in range(rd):
            # print("batch_index:", i * batch_size)
            yield index[i * batch_size: (i + 1) * batch_size]
            # yield index[6429:6430]
        # yield index[7072:7073]

def init_variable(shape, name=None):
    initial = tf.random_uniform(shape, -0.01, 0.01)
    return tf.Variable(initial, name=name)

class neural_detector(object):
    """
    A tensorflow signal flaw detector.
    Use the 'build' method to customize the network structure.

    Example:
    1. LSTM + CRF: see class 'LSTM_DETECTOR'
    2. Bi-LSTM + CRF: see class 'Bi_LSTM_DETECTOR'
    3. CNN + Bi-LSTM + CRF: see class 'CNN_Bi_LSTM_DETECTOR'

    Use the 'inference' method to define how to calculate
    unary scores of given signal sequences

    Inherit this class and overwrite 'build' & 'inference', you can customize
    structure of your flaw detectors.

    Then use 'run' method to training.
    """

    def __init__(self, hidden_dim, nb_classes, drop_rate=1.0, batch_size=None,
                 time_steps=0, l2_reg=0., fea_dim=17, nb_channel=4):
        self.hidden_dim = hidden_dim
        self.nb_classes = nb_classes
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.l2_reg = l2_reg
        # self.fea_dim = 17 + 4
        self.fea_dim = fea_dim
        self.nb_channel = nb_channel

        with tf.name_scope('inputs'):
            self.X = tf.placeholder(
                tf.float32, shape=[None, self.time_steps, self.nb_channel],
                name='X_placeholder')
            self.Y = tf.placeholder(
                tf.int32, shape=[None, self.nb_classes],
                name='Y_placeholder')
            self.X_LEN = tf.placeholder(
                tf.int32, shape=[None, ], name='X_len_placeholder')
            self.keep_prob = tf.placeholder(tf.float32, name='output_dropout')
            self.X_FEA = tf.placeholder(
                tf.float32, shape=[None, self.fea_dim],
                name="X_FEA_placeholder"
            )
        self.build()
        return

    def __str__(self):
        return "detector"

    def build(self):
        pass

    def inference(self, x, x_len, reuse=None):
        pass

    def get_batch_data(self, x, y, l, batch_size, fea, keep_prob=1.0, shuffle=True):
        for index in batch_index(len(y), batch_size, 1, shuffle):
            feed_dict = {
                self.X: x[index],
                self.Y: y[index],
                self.X_LEN: l[index],
                self.keep_prob: keep_prob,
                self.X_FEA: fea[index]
            }
            yield feed_dict, len(index)

    def loss(self, pred):
        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.Y, pred))
            return cost

    def evaluate(self, preds, labels):
        p = metrics.precision_score(np.argmax(labels, 1), np.argmax(preds, 1), average='macro')
        r = metrics.recall_score(np.argmax(labels, 1), np.argmax(preds, 1), average='macro')
        f = metrics.f1_score(np.argmax(labels, 1), np.argmax(preds, 1), average='macro')
        return (p, r, f)

    def run(
        self,
        train_x, train_y, train_lens,
        valid_x, valid_y, valid_lens,
        test_x, test_y, test_lens,
        fea_train, fea_valid, fea_test,
        FLAGS=None
    ):
        if FLAGS is None:
            print("FLAGS ERROR")
            sys.exit(0)

        self.lr = FLAGS.lr
        self.training_iter = FLAGS.train_steps
        self.train_file_path = FLAGS.train_data
        self.test_file_path = FLAGS.valid_data
        self.display_step = FLAGS.display_step

        # unary_scores & loss
        pred = self.inference(self.X, self.X_LEN, self.X_FEA)
        cost = self.loss(pred)
        tf.add_to_collection('pred', pred)
        # decay_learning_rate = tf.train.exponential_decay(learning_rate=self.lr, global_step=global_step, decay_steps=200, decay_rate=0.98, staircase=False)
        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            # decay_learning_rate = tf.train.exponential_decay(learning_rate=self.lr, global_step=global_step,
            #                                                  decay_steps=200, decay_rate=0.8, staircase=False)
            # optimizer = tf.train.AdamOptimizer(
            #     learning_rate=self.lr).minimize(cost, global_step=global_step)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost, global_step=global_step)

        with tf.name_scope('summary'):
            if FLAGS.log:
                localtime = time.strftime("%Y%m%d-%X", time.localtime())
                Summary_dir = FLAGS.log_dir + localtime

                info = 'batch:{}, lr:{}, l2_reg:{}'.format(
                    self.batch_size, self.lr, self.l2_reg)
                info += ';' + self.train_file_path + ';' + os.path.sep +\
                    self.test_file_path + ';' + 'Method:%s' % self.__str__()
                train_acc = tf.placeholder(tf.float32)
                train_loss = tf.placeholder(tf.float32)
                summary_acc = tf.summary.scalar('ACC ' + info, train_acc)
                summary_loss = tf.summary.scalar('LOSS ' + info, train_loss)
                summary_op = tf.summary.merge([summary_loss, summary_acc])

                valid_acc = tf.placeholder(tf.float32)
                valid_loss = tf.placeholder(tf.float32)
                summary_valid_acc = tf.summary.scalar('ACC ' + info, valid_acc)
                summary_valid_loss = tf.summary.scalar(
                    'LOSS ' + info, valid_loss)
                summary_valid = tf.summary.merge(
                    [summary_valid_loss, summary_valid_acc])

                train_summary_writer = tf.summary.FileWriter(
                    Summary_dir + '/train')
                valid_summary_writer = tf.summary.FileWriter(
                    Summary_dir + '/valid')

        with tf.name_scope('saveModel'):
            localtime = time.strftime("%X-%Y-%m-%d", time.localtime()).replace(':', '_')
            saver = tf.train.Saver()
            save_dir = FLAGS.model_dir + localtime + os.path.sep
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        with tf.Session() as sess:
            max_acc, bestIter = 0., 0

            if FLAGS.restore_model != None or self.training_iter == 0:
                saver.restore(sess, FLAGS.restore_model)
                print("[+] Model restored from %s" % FLAGS.restore_model)
            else:
                sess.run(tf.global_variables_initializer())

            for epoch in range(self.training_iter):
                training_loop = 0
                for train, num in self.get_batch_data(train_x, train_y, train_lens, self.batch_size, fea_train, (1 - self.drop_rate)):
                    training_loop += 1
                    _, step, loss, predication = sess.run(
                        [optimizer, global_step, cost, pred],
                        feed_dict=train)
                    if math.isnan(loss):
                        print("Pred:", predication, "Cost: ", loss)
                        print(train)
                        raise ValueError("nan loss detected")
                    accuracy, recall, f1 = self.evaluate(predication, train[self.Y])
                    if FLAGS.log:
                        summary = sess.run(summary_op, feed_dict={
                            train_loss: loss, train_acc: accuracy})
                        train_summary_writer.add_summary(summary, step)
                    print(step, loss, accuracy)
                    print('Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, accuracy))
                save_path = saver.save(sess, save_dir, global_step=step)
                print("[+] Model saved in file: %s" % save_path)

                if epoch % self.display_step == 0:
                    rd, loss, acc = 0, 0., 0.
                    for valid, num in self.get_batch_data(valid_x, valid_y, valid_lens, self.batch_size, fea_valid):
                        _loss, predication = sess.run(
                            [cost, pred], feed_dict=valid)
                        loss += _loss
                        # rewrite evaluate
                        f, p, f1 = self.evaluate(predication, valid[self.Y])
                        acc += f
                        rd += 1
                    loss /= rd
                    acc /= rd
                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step
                    if FLAGS.log:
                        summary = sess.run(summary_valid, feed_dict={
                            valid_loss: loss, valid_acc: acc})
                        valid_summary_writer.add_summary(summary, step)
                    print('----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))
                    print('Iter {}: valid loss(avg)={:.6f}, acc(avg)={:.6f}'.format(step, loss, acc))
                    print('round {}: max_acc={} BestIter={}\n'.format(epoch, max_acc, bestIter))

            print('Optimization Finished!')
            pred_test_y = []
            acc, loss, rd = 0., 0., 0
            for test, num in self.get_batch_data(test_x, test_y, test_lens, self.batch_size, fea_test, shuffle=False):
                _loss, prediction = sess.run(
                    [cost, pred], feed_dict=test)
                loss += _loss
                rd += 1
                f, recall, f1 = self.evaluate(prediction, test[self.Y])
                acc += f
                pred_test_y.extend(prediction)
            acc /= rd
            loss /= rd
            return pred_test_y, loss, acc

# class MATT_DETECTOR(neural_detector):
#     def __str__(self):
#         return "Multi-head Attention Detector"
#
#     def build(self):
#         pass
#
#     def loss(self, pred):
#         with tf.name_scope('loss'):
#             cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=pred))
#             return cost
#
#     def

class SEPERABLE_CNN_DETECTOR(neural_detector):
    def __str__(self):
        return "1D SEPERABLE CNN DETECTOR"

    def build(self):
        pass

    def loss(self, pred):
        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=pred))
            return cost

    def inference(self, X, X_LEN, _):
        FILTER_PARAMS = [
            [128, 256],
            [128, 3],
            [128, 5]
        ]
        with tf.name_scope('weights'):
            outputs = tf.layers.separable_conv1d(
                inputs=X,
                filters=FILTER_PARAMS[0][0],
                kernel_size=FILTER_PARAMS[0][1],
                data_format='channels_first',
                use_bias=True
            )
            outputs = tf.layers.max_pooling1d(
                inputs=outputs,
                pool_size=20,
                strides=1
            )
        for i in range(1, len(FILTER_PARAMS)):
            filter_num, kernel_size = FILTER_PARAMS[i]
            outputs = tf.layers.separable_conv1d(
                inputs=outputs,
                filters=filter_num,
                kernel_size=kernel_size,
                use_bias=True
            )
            outputs = tf.layers.max_pooling1d(
                inputs=outputs,
                pool_size=20,
                strides=1
            )
        # dense layers
        outputs = tf.layers.dense(
            inputs=tf.layers.flatten(outputs),
            units=100,
            # activation=tf.nn.softmax,
            use_bias=True
        )
        outputs = tf.layers.dense(
            inputs=outputs,
            units=2,
            use_bias=True
        )
        return outputs

class CNN_DETECTOR(neural_detector):

    def __str__(self):
        return "1D CNN DETECTOR"

    def build(self):
        pass

    def loss(self, pred):
        print("pred size:", tf.rank(pred))
        print("label size:", tf.rank(self.Y))
        with tf.name_scope('loss'):
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=pred))
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred))
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=pred))
            return cost

    def inference(self, X, X_LEN, _):
        FILTER_PARAMS = [
            [128, 2],
            [128, 3],
            [128, 5]
        ]
        print("xxxx", tf.shape(X))
        with tf.name_scope('weights'):

            # branch 1
            outputs = tf.layers.conv1d(
                inputs=X,
                filters=FILTER_PARAMS[0][0],
                kernel_size=FILTER_PARAMS[0][1],
                # activation=tf.nn.relu,
                use_bias=True
            )
            outputs = tf.layers.max_pooling1d(
                inputs=outputs,
                pool_size=20,
                strides=1
            )
            for i in range(1, len(FILTER_PARAMS)):
                filter_num, kernel_size = FILTER_PARAMS[i]
                outputs = tf.layers.conv1d(
                    inputs=outputs,
                    filters=filter_num,
                    kernel_size=kernel_size,
                    # activation=tf.nn.relu,
                    use_bias=True
                )


            # #branch2
            # branch2 = tf.layers.conv1d(
            #     inputs=X,
            #     filters=128,
            #     kernel_size=3,
            #     strides=1,
            #     use_bias=True
            # )
            # branch2 = tf.layers.max_pooling1d(
            #     inputs=branch2,
            #     pool_size=10,
            #     strides=1
            # )
            # #
            # outputs = tf.concat([outputs, branch2], 1)

            # dropout layer
            outputs = tf.layers.dropout(
                inputs=outputs,
                rate=self.drop_rate)
            # max pooling layers
            outputs = tf.layers.max_pooling1d(
                inputs=outputs,
                pool_size=20,
                strides=1
            )
            # dense layers
            outputs = tf.layers.dense(
                inputs=tf.layers.flatten(outputs),
                units=100,
                # activation=tf.nn.softmax,
                use_bias=True
            )
            outputs = tf.layers.dense(
                inputs=outputs,
                units=2,
                use_bias=True
            )
        return outputs

class CNN_FEA_DETECTOR(neural_detector):

    def __str__(self):
        return "1D CNN FEA DETECTOR"

    def build(self):
        pass

    def loss(self, pred):
        with tf.name_scope('loss'):
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=pred))
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=pred))
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=pred))
            # print("Label:", self.Y.eval())
            # print("Pred:", pred.eval())
            # cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y, logits=pred))
            return cost

    def inference(self, X, X_LEN, X_FEA):
        FILTER_PARAMS = [
            [128*3, 2],
            [128*2, 3],
            [128, 5]
        ]
        with tf.name_scope('weights'):
            outputs = tf.layers.conv1d(
                inputs=X,
                filters=FILTER_PARAMS[0][0],
                kernel_size=FILTER_PARAMS[0][1],
                # activation=tf.nn.relu,
                use_bias=True
            )
            outputs = tf.layers.max_pooling1d(
               inputs=outputs,
               pool_size=20,
               strides=1
            )
            for i in range(1, len(FILTER_PARAMS)):
                filter_num, kernel_size = FILTER_PARAMS[i]
                outputs = tf.layers.conv1d(
                    inputs=outputs,
                    filters=filter_num,
                    kernel_size=kernel_size,
                    use_bias=True
                )

            # outputs = tf.layers.dropout(
            #     inputs=outputs,
            #     rate=self.drop_rate)
            # max pooling layers
            outputs = tf.layers.max_pooling1d(
                inputs=outputs,
                pool_size=20,
                strides=1
            )
            # # dense layers
            # outputs = tf.layers.dense(
            #     inputs=tf.layers.flatten(outputs),
            #     units=100,
            #     # activation=tf.nn.softmax,
            #     use_bias=True
            # )
            #
            # #dense layers with features
            # fea_outputs = tf.layers.dense(
            #     inputs=X_FEA,
            #     units=100,
            #     use_bias=True
            # )
            mix_concat_outputs = tf.concat([tf.layers.flatten(outputs), X_FEA], 1)
            mix_outputs = tf.layers.batch_normalization(
                inputs=mix_concat_outputs
            )
            mix_outputs = tf.layers.dense(
                inputs=tf.layers.flatten(mix_outputs),
                units=100,
                use_bias=True
            )
            mix_outputs = tf.layers.batch_normalization(
                inputs=mix_outputs
            )
            outputs = tf.layers.dense(
                inputs=mix_outputs,
                units=2,
                use_bias=True
            )
        return outputs


class Bi_LSTM_DETECTOR(neural_detector):
    def __str__(self):
        return "BiLSTM DETECTOR"

    def loss(self, pred):
        print("pred size:", tf.rank(pred))
        print("label size:", tf.rank(self.Y))
        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=pred))
            return cost

    def build(self):
        with tf.name_scope('weigths'):
            self.W = tf.get_variable(
                shape=[self.hidden_dim * 2, self.nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='weights'
            )
            self.lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            self.lstm_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)

        with tf.name_scope('biases'):
            self.b = tf.Variable(tf.zeros([self.nb_classes], name="bias"))
        return

    def inference(self, X, X_LEN, reuse=None):
        outputs = tf.layers.dropout(X, self.drop_rate)
        outputs = tf.reshape(outputs, [-1, self.time_steps, 1])
        with tf.variable_scope('label_inference', reuse=reuse):
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.lstm_fw,
                cell_bw=self.lstm_bw,
                inputs=outputs,
                dtype=tf.float32,
                sequence_length=X_LEN
            )
            outputs = tf.concat([output_fw, output_bw], 2)
            # outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])
            # outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)
            # dense layers
            outputs = tf.layers.dense(
                inputs=tf.layers.flatten(outputs),
                units=100,
                # activation=tf.nn.softmax,
                use_bias=True
            )
            outputs = tf.layers.dense(
                inputs=outputs,
                units=2,
                use_bias=True
            )
        return outputs

class CNN_ATTN_DETECTOR(neural_detector):

    def __str__(self):
        return "1D CNN ATTN DETECTOR"

    def build(self):
        pass

    def loss(self, pred):
        print("pred size:", tf.rank(pred))
        print("label size:", tf.rank(self.Y))
        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=pred))
            return cost

    def inference(self, X, X_LEN, X_FEA):
        FILTER_PARAMS = [
            [128, 2],
            [128, 3],
            [128, 5]
        ]
        with tf.name_scope('weights'):
            # branch 1
            outputs = tf.layers.conv1d(
                inputs=X,
                filters=FILTER_PARAMS[0][0],
                kernel_size=FILTER_PARAMS[0][1],
                # activation=tf.nn.relu,
                use_bias=True
            )
            tf.layers.separable_conv1d()
            outputs = tf.layers.max_pooling1d(
                inputs=outputs,
                pool_size=20,
                strides=1
            )
            for i in range(1, len(FILTER_PARAMS)):
                filter_num, kernel_size = FILTER_PARAMS[i]
                outputs = tf.layers.o;)SXZ$frL"@!#jihlo0;-p9l 1d(
                    inputs=outputs,
                    filters=filter_num,
                    kernel_size=kernel_size,
                    # activation=tf.nn.relu,
                    use_bias=True
                )

            # max pooling layers
            outputs = tf.layers.max_pooling1d(
                inputs=outputs,
                pool_size=20,
                strides=1
            )
            print("DEBUG: shape", outputs.get_shape())


            # Building MultiAttentionLayers
            head_n = 10
            attn_size = 24
            output_size = 128
            input_emb_size = X.get_shape()[-1]
            mattn_layer1 = MultiHeadAttention(head_n, input_emb_size, attn_size, 1)

            attn_outputs = mattn_layer1(head_n, X, attn_size, output_size)
            print("attn_output ", attn_outputs.get_shape())

            # merge attentions & cnn outputs
            outputs = tf.concat([tf.layers.flatten(outputs), attn_outputs], axis=-1)

            # dense layers
            outputs = tf.layers.dense(
                inputs=tf.layers.flatten(outputs),
                units=100,
                # activation=tf.nn.softmax,
                use_bias=True
            )
            outputs = tf.layers.dense(
                inputs=outputs,
                units=2,
                use_bias=True
            )
        return outputs

class MultiheadAttentionDetector(neural_detector):
    def __str__(self):
        return "Multihead Attention Detector"

    def loss(self, pred):
        with tf.name_scope('loss'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=pred))
            return cost

    def build(self):
        # with tf.name_scope('attn_weights'):
        #     self.W = tf.get_variable(
        #         shape=[self.hidden_dim * 2, self.nb_classes],
        #         initializer=tf.truncated_normal_initializer(stddev=0.01),
        #         name='weights'
        #     )
        #     self.lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
        #     self.lstm_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
        #
        # with tf.name_scope('biases'):
        #     self.b = tf.Variable(tf.zeros([self.nb_classes], name="bias"))
        return

    def inference(self, X, X_LEN, X_FEA, reuse=None):
        head_n = 10
        attn_size = 24
        output_size = 128
        print("reuse", reuse)
        input_emb_size = X.get_shape()[-1]
        # query_size = X.get_shape()[-2]
        mattn_layer1 = MultiHeadAttention(head_n, input_emb_size, attn_size, 1)

        with tf.variable_scope('label_inference'):
            attn_outputs = mattn_layer1(head_n, X, attn_size, output_size)
            print("attn_output ", attn_outputs.get_shape())
            attn_outputs = mattn_layer1(head_n, attn_outputs, attn_size, output_size)
            outputs = tf.layers.dense(
                inputs=tf.layers.flatten(attn_outputs),
                units=2,
                use_bias=True
            )
            print("outputs ", outputs.get_shape())
        return outputs

class MultiHeadAttention(object):
    """
    Multi-head Attention structures    
    """
    def __init__(self, n_head, model_size, attn_size, index):
        self.n_head = n_head
        # defining Query / Key / Value transformation
        # Query、Key、Value的值应当一致

        # Attention Linear Transformations Variables
        self.WQ = []
        self.WK = []
        self.WV = []

        for hi in range(self.n_head):
            # queryname
            qw_name = 'qweights_' + str(hi) + '_' + str(index)
            self.WQ.append(tf.get_variable(
                shape=[model_size, attn_size],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name=qw_name)
            )

            # keyname
            kw_name = 'kweihgts_' + str(hi) + '_' + str(index)
            self.WK.append(tf.get_variable(
                shape=[model_size, attn_size],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name=kw_name)
            )

            vw_name = 'vweights_' + str(hi) + '_' + str(index)
            self.WV.append(tf.get_variable(
                shape=[model_size, attn_size],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name=vw_name)
            )


    def __call__(self, n_head, X, attn_size, output_size):
        """
        [batch_size, query_len, emb_size]
        [batch_size, value_len, emb_size]
        [batch_size, key_len, emb_size]
        :param X: Input Tensor, [batch_size, seq_len, emb_size]
        :param Query: if None, self-attention; otherwise, using Query to generate query informations
        :return: Tensor of Value * alignment
        """
        input_emb_size = X.get_shape()[-1]
        seq_len = X.get_shape()[-2]
        # query_size = X.get_shape()[-2]
        # key_size, value_size = query_size, query_size
        # self.__init__(n_head, query_size, key_size, value_size, attn_size)
        print("input_emb_size", input_emb_size)
        print("sequence length", seq_len)
        # Doing linear transforms, turning into keys
        attn_querys = []
        #[batch_size, query_len, emb_size] -> [batch_size, query_len, attn_size]
        # [5120, 12]
        X = tf.reshape(X, [-1, input_emb_size])
        for wq in self.WQ:
            query = tf.reshape(tf.matmul(X, wq), [-1, seq_len, attn_size])
            attn_querys.append(query)

        attn_values = []
        #[batch_size, value_len, emb_size] --> [batch_size, value_len, attn_size]
        for wv in self.WV:
            value = tf.reshape(tf.matmul(X, wv), [-1, seq_len, attn_size])
            attn_values.append(value)

        attn_keys = []
        # [batch_size, key_len, emb_size] --> [batch_size, key_len, attn_size]
        for wk in self.WK:
            key = tf.reshape(tf.matmul(X, wk), [-1, seq_len, attn_size])
            attn_keys.append(key)

        # alignment scores
        align_score = []
        for i in range(len(attn_querys)):
            # [batch_size, query_len, attn_size] * [batch_size, value_len, attn_size]^T -->
            # [batch_size, query_len, value_len]
            ascore = tf.matmul(attn_querys[i], attn_keys[i], transpose_b=True) \
                          / tf.math.sqrt(tf.dtypes.cast(attn_size, tf.dtypes.float32))
            ascore = tf.math.softmax(ascore, axis=-1)
            align_score.append(ascore)

        # attentionend values
        heads = []
        for i in range(len(attn_values)):
            # [batch_size, query_len, key_len] * [batch_size, value_len, attn_size]
            # key_length equals value_length
            context = tf.matmul(align_score[i], attn_values[i])
            heads.append(context)

        # single head: [batch_size, value_len, attn_size]
        # multi-heads: [batch_size, value_len, attn_size * n_heads]
        attn_outputs = tf.concat(heads, axis=-1)
        print("attn_outputs_shape ", attn_outputs.get_shape())

        dense_outputs = tf.layers.dense(inputs=tf.layers.flatten(attn_outputs),
                                        units=output_size,
                                        use_bias=True,
                                        activation=tf.nn.relu)
        print("dense_outputs ", dense_outputs.get_shape())
        # batchnormalization layers
        bn_outputs = tf.layers.batch_normalization(inputs=dense_outputs)
        print("bn_outputs: ", tf.reshape(bn_outputs, [1, -1]).get_shape())
        print("bn_outputs ", bn_outputs.get_shape())

        return bn_outputs

