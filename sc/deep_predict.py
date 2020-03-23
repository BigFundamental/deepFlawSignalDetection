import numpy as np
import tensorflow as tf
import pandas as pd

import os
from sc.deep_feature_extractor import DeepFeatureExtractor
from sc.model import ModelVersionFeatureConfig as mc
from sc.signal_manager import SignalMgr

# print(sys.path)

class DeepPredictor(object):

    def __init__(self, model_meta_path, model_ckpt_path):
        self.igraph = tf.Graph()
        self.sess = tf.Session(graph=self.igraph)
        with self.igraph.as_default():
            # new_saver = tf.train.import_meta_graph(model_meta_path)
            # new_saver = tf.train.import_meta_graph('./models/deep_model/-8729.meta')
            new_saver = tf.train.import_meta_graph(model_meta_path)
            new_saver.restore(self.sess, tf.train.latest_checkpoint(model_ckpt_path))
            # new_saver.restore(self.sess, tf.train.latest_checkpoint('./models/deep_model/'))
            self.X_INPUT_PLACEHOLDER = self.igraph.get_tensor_by_name('inputs/X_placeholder:0')
            self.X_FEA_INPUT_PLACEHOLDER = self.igraph.get_tensor_by_name('inputs/X_FEA_placeholder:0')
            self.X_LEN_PLACEHOLDER = self.igraph.get_tensor_by_name('inputs/X_len_placeholder:0')
            self.pred = tf.get_collection('pred')
        return

    def predict(self, raw_signals, features, n_channels):
        # self.graph = tf.get_default_graph()
        deep_signals, signal_len = DeepFeatureExtractor.features(raw_signals[0:1024], features, n_channels)
        feature_names = mc['gbdt_smooth_signal']['features']
        feature_vec = get_features_vec(features, feature_names)
        pred_score = self.sess.run([self.pred], feed_dict={self.X_INPUT_PLACEHOLDER: deep_signals,
                                                    self.X_FEA_INPUT_PLACEHOLDER: np.array([feature_vec]),
                                                    self.X_LEN_PLACEHOLDER: np.array([signal_len])})
        return pred_score


def get_features_vec(features, feature_names):
    sorted_features = sorted(feature_names)

    fea_vec = np.zeros(len(feature_names))
    i = 0
    for feature_name in sorted_features:
        fea_vec[i] = features[feature_name]
        i += 1
    return fea_vec

    return feature_set
