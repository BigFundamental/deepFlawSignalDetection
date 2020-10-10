#! -*- encoding: utf-8
import os
import collections
import pandas as pd
import hashlib
import numpy as np
from sc.signal_manager import SignalMgr
from configs import FEATURE_CONFIG
import ast


class Features(object):
    """
    Generating Features For Future Use
    """
    DEFAULT_CACHE_DIR = "fea_cache"
    DEFAULT_META_SEP = ":"
    DEFAULT_FILEPATH_NAME = "filepath"
    DEFAULT_SIGMGR_PARAM = {'skip_row': [1], 'model_path': ['train']}
    DEFAULT_LABEL_NAME = "label"
    DEFAULT_KIND_NAME = "kind"

    def __init__(self,
                 input_fpath):
        self.instance_fpath = input_fpath
        self.feature_df = None
        self.labels = None
        self.meta_info = Features.read_meta(input_fpath)
        pass

    @staticmethod
    def read_meta(instance_fpath):
        """
        Read Meta Info Under Instance Directory
        return: meta dict params
        """
        meta_info = dict()
        meta_fpath = os.path.dirname(instance_fpath) + os.sep + 'meta'
        for line in open(meta_fpath, 'r'):
            key, val = line.strip().split(':')
            meta_info[key] = val

        return meta_info

    @staticmethod
    def abs_instance_root(instance_fpath):
        """

        :param instance_fpath: return absolute instance root path
        :return:
        """
        meta = Features.read_meta(instance_fpath)
        abs_ins_dir = os.path.dirname(instance_fpath)
        return abs_ins_dir + os.sep + meta['root']

    @property
    def label(self):
        """

        :return: Get Label fields for Features
        """
        ins_df = pd.read_csv(self.instance_fpath)
        return ins_df[self.DEFAULT_LABEL_NAME]

    @property
    def insw(self, weight_field='kind'):
        """
        :return: Instance weights for training & testing dataframes
        """
        ins_df = pd.read_csv(self.instance_fpath)
        weight_map = {
            -2: 5.0,
            -1: 1.0,
            2: 10.0,
            3: 10.0,
            4: 10.0,
            5: 10.0,
            6: 10.0,
            7: 10.0,
            10: 10.0,
            11: 10.0,
            12: 10.0
        }
        return ins_df[self.DEFAULT_KIND_NAME].map(lambda x: weight_map[x])
        # print(ins_df.groupby([weight_field]).count())
        # return ins_df['label']

    @property
    def kind(self):
        """

        :return: Get Label Kind For results
        """
        ins_df = pd.read_csv(self.instance_fpath)
        return ins_df[self.DEFAULT_KIND_NAME]

    def feature_list_normalize_(self, feature_names):
        """

        :param feature_names: normalize orders of given features names
        :return: normalized ordered names of features
        """
        sorted_features = sorted(feature_names)
        return sorted_features

    @staticmethod
    def cache_key(raw_keys):
        """
        generate cache key for cache
        :return: string for file keys
        """
        raw_str = ','.join(raw_keys)
        return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

    def is_feature_cache_exists(self, feature_names):
        """
        Check whether feature cache exists
        :return: True if exists, otherwise False
        """
        # 使用样本文件 + 特征名作为构建cache名的唯一目标
        cache_key = Features.cache_key([self.instance_fpath, ','.join(feature_names)])
        if os.path.exists(Features.DEFAULT_CACHE_DIR + os.sep + cache_key):
            return True
        return False

    def generate_from_cache(self, feature_names):
        cache_key = Features.cache_key([self.instance_fpath, ','.join(feature_names)])
        cache_fpath = Features.DEFAULT_CACHE_DIR + os.sep + cache_key

        if not os.path.exists(cache_fpath):
            raise Exception("%s not exists" % cache_fpath)

        return pd.read_csv(cache_fpath)

    def generate(self,
                 feature_list,
                 enable_cache=True):
        """

        :param

        force: If True given, feature module will generate features in every run.
                      Otherwise Feature module will skip generate features and use local feature cache instead
                      for time-saving consideration
        :return:
        """
        normalized_feature_names = self.feature_list_normalize_(feature_list)
        print("normalized_feature_names: ", normalized_feature_names)

        if enable_cache:
            if self.is_feature_cache_exists(normalized_feature_names):
                print("features already exists: %s for instance: %s" %
                      (Features.cache_key([self.instance_fpath, ','.join(normalized_feature_names)]),
                       self.instance_fpath))
                self.feature_df = self.generate_from_cache(normalized_feature_names)
                return self.feature_df

        # 读取样本index数据索引
        ins_df = pd.read_csv(self.instance_fpath)

        # 重新生成特征集合
        sig_manager = SignalMgr()
        feature_set = collections.OrderedDict()
        for name in normalized_feature_names:
            feature_set[name] = list()

        paths = ins_df[Features.DEFAULT_FILEPATH_NAME]
        abs_ins_dir = os.path.dirname(self.instance_fpath)
        for relpath in paths:
            # 将相对路径转换成绝对路径
            abs_case_path = abs_ins_dir + os.sep + self.meta_info['root'] + os.sep + relpath
            features = sig_manager.get_features(abs_case_path, request_param=Features.DEFAULT_SIGMGR_PARAM)
            for name in normalized_feature_names:
                feature_set[name].append(features[name])

        self.feature_df = pd.DataFrame(feature_set)

        cache_fpath = Features.DEFAULT_CACHE_DIR + os.sep + \
                          Features.cache_key([self.instance_fpath, ','.join(normalized_feature_names)])
        self.to_csv(cache_fpath)
        print("features: %s, for instances: %s" % (cache_fpath, self.instance_fpath))

        assert np.sum(self.feature_df.isna().sum()) == 0
        return self.feature_df

    def to_csv(self, output_fpath):
        """

        :param output_fpath:  output path
        :return:
        """
        dir_path = os.path.dirname(output_fpath)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        self.feature_df.to_csv(output_fpath, index=False)

def feature_statistics(train_fea_obj):
    """
    :param features:
    :return: normal & mean
    """
    # get paired_edge_height number mean & std
    # get edge_height_num
    norm_features = train_fea_obj.feature_df[train_fea_obj.label == 0]
    flatten_edge_height = []
    edge_nums = []
    for elem in norm_features['paired_edge_height'].tolist():
        edge_nums.append(len(ast.literal_eval(elem)))
        for a, b in ast.literal_eval(elem):
            # print(a, b)
            flatten_edge_height.append(a)
            flatten_edge_height.append(b)

    print("edge_height mean: %f  std: %f" % (np.mean(flatten_edge_height), np.std(flatten_edge_height)))
    print("edge_num mean: %f  std: %f" % (np.mean(edge_nums), np.std(edge_nums)))

    interviene_length = []
    for elem in norm_features['unit_interviene_length'].tolist():
        interviene_length.extend(ast.literal_eval(elem))

    # edge_height mean: 1.793800  std: 0.178329
    # edge_num mean: 31.795307  std: 4.474540
    # interviene_length mean: 6.621549  std: 2.183348
    print("interviene_length mean: %f  std: %f" % (np.mean(interviene_length), np.std(interviene_length)))


if __name__ == '__main__':
    # print(Features.cache_key(['/Volumn/key1/key2/trbo65c8v9uiolkjhgfxz¸ikfdszXfghifxzghjklhgpoco0Cop-=0[9pin.csv', 'abc']))
    feature_config = FEATURE_CONFIG['feature_statistic']
    train_ins_fpath = feature_config['train_ins']
    train_fea_obj = Features("/Volumes/workspace/projects/signal_detection/instance/train_ins.csv")
    train_fea_obj.generate(feature_config['feature_list'], enable_cache=True)
    feature_statistics(train_fea_obj)
    # print(train_fea_obj.feature_df.head())

    # print(train_fea_obj.feature_df['paired_edge_height'].tolist)
    # print(np.sum(train_fea_obj.feature_df.isna().sum()))
    # print(train_fea_obj.insw)
    # train_fea_obj.generate(feature_config['feature_list'])