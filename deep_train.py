#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf
import deep_model as detector
from deep_features import DeepFeatureExtractor as Dfea
from configs import FEATURE_CONFIG

import faulthandler
from sklearn.metrics import classification_report

faulthandler.enable()

# TODO: move to configurations
MODEL_DIR="./deep_models/"
LOG_DIR="./logs/"
OUTPUT_DIR="./deep_output/"
MAX_LEN=1024

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_data', r'train.csv', 'Training data file')
tf.app.flags.DEFINE_string(
    'test_data', r'test.csv', 'Test data file')
tf.app.flags.DEFINE_string(
    'valid_data', r'validate.csv', 'Validation data file')
tf.app.flags.DEFINE_string('log_dir', LOG_DIR, 'The log dir')
tf.app.flags.DEFINE_string('model_dir', MODEL_DIR, 'Models dir')
tf.app.flags.DEFINE_string(
    # 'model', "BLSTM", 'Model type: LSTM/BLSTM/CNNBLSTM')
    'model', "CNN_SEP", 'Model type: BLSTM/CNN/CNN_FEA/ATTN/CNN_ATTN')
# tf.app.flags.DEFINE_string('restore_model', "./models/23_01_45-2020-04-14/-6380", 'Path of the model to restored')
tf.app.flags.DEFINE_string('restore_model', None, 'Path of the model to restored')
# tf.app.flags.DEFINE_integer("emb_dim", 1, "embedding size")
tf.app.flags.DEFINE_string("output_dir", OUTPUT_DIR, "Output dir")
# tf.app.flags.DEFINE_boolean('only_test', False, 'Only do the test')
tf.app.flags.DEFINE_float("lr", 0.0001, "learning rate")
tf.app.flags.DEFINE_float("dropout", 0.2, "Dropout rate of input layer")
tf.app.flags.DEFINE_boolean(
    'eval_test', True, 'Whether evaluate the test data.')
tf.app.flags.DEFINE_integer("max_len", MAX_LEN,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("hidden_dim", 128, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 200, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 200, "training steps")
tf.app.flags.DEFINE_integer("display_step", 10, "number of test display step")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization weight")
tf.app.flags.DEFINE_boolean(
    'log', False, 'Whether to record the TensorBoard log.')
tf.app.flags.DEFINE_integer("nb_classes", 2, "Label sizes")
tf.app.flags.DEFINE_integer("nb_segs", 4, "input segments for signals")

def main(_):
    np.random.seed(1337)
    random.seed(1337)

    print("#" * 67)
    print("# Loading data from:")
    print("#" * 67)
    print("Train:", FLAGS.train_data)
    print("Valid:", FLAGS.valid_data)
    print("Test: ", FLAGS.test_data)

    feature_config = FEATURE_CONFIG['feature_v2.2.5.1']
    train_ins_fpath = feature_config['train_ins']
    validate_ins_fpath = feature_config['validate_ins']
    test_ins_fpath = feature_config['test_ins']
    feature_list = feature_config['feature_list']

    split_train_signals, \
    split_validate_signals, \
    split_test_signals, \
    manual_train_fea, \
    manual_validate_fea, \
    manual_test_fea, \
    manual_train_label, \
    manual_validate_label, \
    manual_test_label, \
    seq_len,\
    n_split_channel = \
        Dfea.batch_features(train_ins_fpath, validate_ins_fpath, test_ins_fpath, feature_list, True, FLAGS.nb_segs)

    print("#" * 67)
    print("Training arguments")
    print("#" * 67)
    print("L2 regular:    %f" % FLAGS.l2_reg)
    print("nb_classes:    %d" % FLAGS.nb_classes)
    print("Batch size:    %d" % FLAGS.batch_size)
    print("Hidden layer:  %d" % FLAGS.hidden_dim)
    print("Train epochs:  %d" % FLAGS.train_steps)
    print("Learning rate: %f" % FLAGS.lr)
    print("Feature number: %d" % (len(feature_list)))
    print("Number of split channel: %d" % n_split_channel)

    print("#" * 67)
    print("Training process start.")
    print("#" * 67)

    if FLAGS.model == 'BLSTM':
        model_type = detector.Bi_LSTM_DETECTOR
    elif FLAGS.model == 'CNN':
        model_type = detector.CNN_DETECTOR
    elif FLAGS.model == 'CNN_FEA':
        model_type = detector.CNN_FEA_DETECTOR
    elif FLAGS.model == 'ATTN':
        model_type = detector.MultiheadAttentionDetector
    elif FLAGS.model =="CNN_ATTN":
        model_type = detector.CNN_ATTN_DETECTOR
    elif FLAGS.model == "CNN_SEP":
        model_type = detector.SEPERABLE_CNN_DETECTOR
    else:
        raise TypeError("Unknow model type % " % FLAGS.model)

    model = model_type(
        FLAGS.hidden_dim,
        FLAGS.nb_classes, FLAGS.dropout, FLAGS.batch_size,
        seq_len[0][0], FLAGS.l2_reg, len(feature_list), n_split_channel)

    pred_test, test_loss, test_acc = model.run(
        split_train_signals, manual_train_label, seq_len[0],
        split_validate_signals, manual_validate_label, seq_len[1],
        split_test_signals, manual_test_label, seq_len[2],
        manual_train_fea, manual_validate_fea, manual_test_fea,
        FLAGS)

    # prep.write_prediction_evaluation_records('./test_evaluation.csv', test_labels, pred_test)
    # pResult = [round(value) for value in pred_test]
    print(classification_report(np.argmax(manual_test_label, 1), np.argmax(pred_test, 1)))

if __name__ == "__main__":
    tf.app.run()
