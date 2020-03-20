import numpy as np
import tensorflow as tf
import pandas as pd


from sc.deep_feature_extractor import DeepFeatureExtractor
from sc.model import ModelVersionFeatureConfig as mc
from sc.signal_manager import SignalMgr

# print(sys.path)

class DeepPredictor(object):

    def __init__(self, model_meta_path, model_ckpt_path):
        # print('model_meta_path', model_meta_path)
        # print('model_ckpt_path', model_ckpt_path)
        self.igraph = tf.Graph()
        self.sess = tf.Session(graph=self.igraph)
        with self.igraph.as_default():
            # new_saver = tf.train.import_meta_graph(model_meta_path)
            new_saver = tf.train.import_meta_graph('./models/deep_model/-8729.meta')
            # new_saver.restore(self.sess, tf.train.latest_checkpoint(model_ckpt_path))
            new_saver.restore(self.sess, tf.train.latest_checkpoint('./models/deep_model/'))
            self.X_INPUT_PLACEHOLDER = self.igraph.get_tensor_by_name('inputs/X_placeholder:0')
            self.X_FEA_INPUT_PLACEHOLDER = self.igraph.get_tensor_by_name('inputs/X_FEA_placeholder:0')
            self.X_LEN_PLACEHOLDER = self.igraph.get_tensor_by_name('inputs/X_len_placeholder:0')
            self.pred = tf.get_collection('pred')
        return

    def predict(self, raw_signals, features, n_channels):

        # self.graph = tf.get_default_graph()
        deep_signals, signal_len = DeepFeatureExtractor.features(raw_signals, features, n_channels)
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

def batch_eval(fpath):
    sigMgr = SignalMgr()
    predictor = DeepPredictor('./models/15_10_17-2020-03-16/-8729.meta', './models/15_10_17-2020-03-16/')
    eval_df = pd.read_csv(fpath, names=['date', 'channel', 'path', 'type'], skiprows=1)
    for path in eval_df['path']:
        # print(path)
        dt, raw_signals = sigMgr.parse_signals_from_file(path, 1)
        features = sigMgr.get_features(path, request_param={'skip_row': [1], 'model_path': ['train']})
        score = predictor.predict(raw_signals, features, 4)
        # print(np.argmax(score))

if __name__ == '__main__':
    # predictor = DeepPredictor('./models/15_10_17-2020-03-16/-8729.meta', './models/15_10_17-2020-03-16/')
    # # read_raw_signals
    # file_path = '/Users/changkong/workspace/signal_classification/projects/signal_classification/data/特殊次品样本/20191231统计数据-1/异常-243/20191224_164621759/Channel_2.csv'
    # sigMgr = SignalMgr()
    batch_eval("./data/ng_eval.csv")
    #
    # dt, raw_signals = sigMgr.parse_signals_from_file(file_path, 1)
    # features = sigMgr.get_features(file_path, request_param={'skip_row': [1], 'model_path': ['train']})
    #
    # score = predictor.predict(raw_signals, features, 4)
    # print(np.argmax(score))

#     # print("abc")
#     with tf.Session() as sess:
#         new_saver = tf.train.import_meta_graph('./models/15_10_17-2020-03-16/-8729.meta')
#         new_saver.restore(sess, tf.train.latest_checkpoint('./models/15_10_17-2020-03-16/'))
#         graph = tf.get_default_graph()
#         X_INPUT = graph.get_tensor_by_name('inputs/X_placeholder:0')
#         X_FEA_INPUT = graph.get_tensor_by_name('inputs/X_FEA_placeholder:0')
#         pred = tf.get_collection('pred')
#         print(pred)
#         # X_FEA_INPUT = graph.get_tensor_by_name('inputs/')
#
#         # for node in sess.graph_def.node:
#         #     print(node.name)
#
#     with tf.Session() as sess:
