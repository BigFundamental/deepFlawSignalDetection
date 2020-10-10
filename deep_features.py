#!-*- encoding: utf-8 -*-
"""
author: zhou lin
date: 2017-01-13
brief: signal pattern extractors
"""

import numpy as np
import pandas as pd
import os, copy
from sc.filter import Filter
from sc.signal_manager import SignalMgr
from sc.model import ModelVersionFeatureConfig as mc
from features import Features
from configs import FEATURE_CONFIG

class DeepFeatureExtractor(object):
    """
    Deep Signal Feature Extractors
    """

    def __init__(self):
        pass

    @staticmethod
    def features(raw_signal, norm_feas, n_channel=4):
        print(mc['skew_valley_diff_enhance']['norm_mean'], mc['skew_valley_diff_enhance']['norm_std'])
        norm_signal = np.array([DeepFeatureExtractor.get_norm_signals(raw_signal, mc['skew_valley_diff_enhance']['norm_mean'], mc['skew_valley_diff_enhance']['norm_std'])])
        # print("norm_signal", norm_signal)
        # print("norm_signal shape", norm_signal.shape)
        # print("norm_signal shape:", norm_signal.shape)
        medfilter_signal = np.array([DeepFeatureExtractor.get_medfilter_signals(raw_signal)])
        # print("medfilter_signal", medfilter_signal)
        # print("medfitler_signal size", medfilter_signal.shape)
        # print("medfilter_signal shape:", medfilter_signal.shape)
        res_signal = np.array(DeepFeatureExtractor.get_res_signals(raw_signal, medfilter_signal))
        # print("res_signal", res_signal)
        # print("res_signal size", res_signal.shape)
        # print("res signal shape:", res_signal.shape)
        bottom_shapes = np.array([DeepFeatureExtractor.get_bottom_shape_signals(raw_signal, norm_feas)])
        # print("bottom signal:", bottom_shapes)
        # print("bottom signal shape:", bottom_shapes.shape)
        signal_len = len(raw_signal)
        return DeepFeatureExtractor.stacked_channel_signals_((norm_signal, medfilter_signal, res_signal, bottom_shapes),
                                                             signal_len, n_channel)

    @staticmethod
    def batch_features(train_index_fpath, validate_index_fpath, test_index_fpath, feature_list, channel_first_stack=False, n_channel=4):
        """

        :param train_fpath: 训练地址，主要为训练的索引路径
        :param validate_fpath: 验证地址，主要为验证集合的索引路径
        :param test_fpath: 验证地址，主要为测试集合的索引路径
        :param feature_list: 特征集合地址
        :param n_channel:
        :return:
        """
        # 读取读取训练数据集合（通过前后两项进行输入）
        print("start reading signals from raw index files")
        train_signals, train_labels, train_lens, max_train_lens = DeepFeatureExtractor.read_batch_signals(train_index_fpath)
        validate_signals, validate_labels, validate_lens, max_validate_lens = DeepFeatureExtractor.read_batch_signals(validate_index_fpath)
        test_signals, test_labels, test_lens, max_test_lens = DeepFeatureExtractor.read_batch_signals(test_index_fpath)

        # extracting traditional
        manual_train_obj = Features(train_index_fpath)
        manual_train_fea = manual_train_obj.generate(feature_list).values
        manual_train_label = manual_train_obj.label.values

        manual_validate_obj = Features(validate_index_fpath)
        manual_validate_fea = manual_validate_obj.generate(feature_list).values
        manual_validate_label = manual_validate_obj.label.values

        manual_test_obj = Features(test_index_fpath)
        manual_test_fea = manual_test_obj.generate(feature_list).values
        manual_test_label = manual_test_obj.label.values

        # calculate mean, std
        signal_mean = np.mean(train_signals)
        signal_std = np.std(train_signals)
        print("signal mean:%f, signal std:%f" % (signal_mean, signal_std))

        # bi-directional signals
        train_signals = np.concatenate([train_signals, DeepFeatureExtractor.batch_mirror(train_signals)], axis=0)
        # fast fake manual features for another directions
        manual_train_fea = np.concatenate((manual_train_fea, copy.deepcopy(manual_train_fea)), axis=0)
        # fake labels
        manual_train_label = np.concatenate((manual_train_label, copy.deepcopy(manual_train_label)), axis=0)
        # fake lengths
        train_lens = np.concatenate((train_lens, copy.deepcopy(train_lens)), axis=0)

        # normalize channel
        norm_train_singals, norm_validate_signals, norm_test_signals = DeepFeatureExtractor.batch_normalize_signals(
            [train_signals, validate_signals, test_signals], signal_mean, signal_std)

        # medfilter channel
        med_train_signals, med_validate_signals, med_test_signals = DeepFeatureExtractor.batch_medfilter_signals(
            [train_signals, validate_signals, test_signals])

        # res channel
        res_train_signals, res_validate_signals, res_test_signals = DeepFeatureExtractor.batch_res_signals(
            [train_signals, validate_signals, test_signals], [med_train_signals, med_validate_signals, med_test_signals])

        # # bottom shape channel
        # bottom_train_signals, bottom_validate_signals, bottom_test_signals = DeepFeatureExtractor.batch_bottom_signals(
        #     [train_signals, validate_signals, test_signals], [norm_train_singals, norm_validate_signals, norm_test_signals])

        # multi-channel informations in order to capture cyclic informations
        train_stacked_signals, n_split_channel = DeepFeatureExtractor.stacked_channel_signals_(
            (norm_train_singals, med_train_signals, res_train_signals), train_lens[0], channel_first_stack, n_channel)
        validate_stacked_signals, _ = DeepFeatureExtractor.stacked_channel_signals_(
            (norm_validate_signals, med_validate_signals, res_validate_signals), validate_lens[0], channel_first_stack, n_channel)
        test_stacked_signals, _ = DeepFeatureExtractor.stacked_channel_signals_(
            (norm_test_signals, med_test_signals, res_test_signals), test_lens[0], channel_first_stack, n_channel)


        def one_hot(label):
            return (np.arange(2) == label[:, None]).astype(np.float32)

        if channel_first_stack:
            return train_stacked_signals, validate_stacked_signals, test_stacked_signals, \
               manual_train_fea, manual_validate_fea, manual_test_fea, \
               one_hot(manual_train_label), one_hot(manual_validate_label), one_hot(manual_test_label),\
               [train_lens, validate_lens, test_lens], n_split_channel
        return train_stacked_signals, validate_stacked_signals, test_stacked_signals, \
               manual_train_fea, manual_validate_fea, manual_test_fea, \
               one_hot(manual_train_label), one_hot(manual_validate_label), one_hot(manual_test_label),\
               [train_lens // n_channel, validate_lens // n_channel, test_lens // n_channel], n_split_channel

    @staticmethod
    def batch_medfilter_signals(signals):
        med_signals = [np.array(DeepFeatureExtractor.get_medfilter_signals(sig)) for sig in signals]
        return med_signals

    @staticmethod
    def batch_res_signals(signals, med_signals):
        res_sig = []
        for i, sig in enumerate(signals):
            res_sig.append(np.array(DeepFeatureExtractor.get_res_signals(sig, med_signals[i])))
        return res_sig

    @staticmethod
    def batch_normalize_signals(signals, mean, std):
        norm_signals = [np.array(DeepFeatureExtractor.normalize_signals(sig, mean, std)) for sig in signals]
        return norm_signals

    @staticmethod
    def normalize_signals(raw_signals, mean, std):
        return (np.array(raw_signals) - mean) / std

    @staticmethod
    def stacked_channel_signals_(signal_vec, signal_len, channel_seperable=False, n_fold=4):
        if channel_seperable == False:
            stacked_signals = np.stack(signal_vec, axis=-1)
            print("stacked size: ", stacked_signals.shape)
            # stacked_signals = np.concatenate(signal_vec, axis=-1)
            # print("stack signal shape", stacked_signals.shape)
            x, y, z = stacked_signals.shape[0], stacked_signals.shape[1] // n_fold, stacked_signals.shape[2] * n_fold
            return np.reshape(stacked_signals, (x, y, z)), z
            # signals.shape[0], signals.shape[1] // split_num, signals.shape[2] * split_num
        else:
            return np.stack(signal_vec, axis=1), len(signal_vec)

    @staticmethod
    def get_norm_signals(raw_signal, mu, delta):
        # mu = np.mean(raw_signal)
        # delta = np.std(raw_signal)
        if delta == 0.0:
            delta = 1.0
        return (raw_signal - mu) / delta

    @staticmethod
    def get_medfilter_signals(raw_signal):
        return Filter.medfilter(raw_signal, SignalMgr.signalParams['PEAK_WINDOW_SIZE'])

    @staticmethod
    def get_res_signals(raw_signal, medfilter_signal):
        peak_candidates = raw_signal - medfilter_signal
        return peak_candidates

    @staticmethod
    def get_bottom_shape_signals(raw_signal, feas):
        bottom_shapes = [0.0] * len(raw_signal)
        for up, down in feas['paired_edges']:
            bottom_shapes[up[0]] = raw_signal[up[0]]
            bottom_shapes[down[1]] = raw_signal[down[1]]
        return bottom_shapes


    @staticmethod
    def biread_batch_features(input_dir, input_fname, skip_rows=1):
        feas = DeepFeatureExtractor.read_batch_features(input_dir, input_fname, skip_rows)
        feas.extend(copy.deepcopy(feas))
        return feas

    @staticmethod
    def read_batch_features(input_dir, input_fname, skip_rows=1):
        line_no = 0
        feas = []
        with open(os.path.join(input_dir, input_fname), 'r') as fr:
            for line in fr:
                if line_no < skip_rows:
                    line_no += 1
                    continue
                elems = list(map(np.float32, line.rstrip().split(',')))
                feas.append(elems)
        return feas

    @staticmethod
    def read_signal_signals(fpath, skip_row=1):
        """read signal values"""
        signals = []
        with open(fpath, 'r') as fr:
            signals = list(map(str.rstrip, fr.readlines()))[skip_row:]
        return list(map(float, signals))

    @staticmethod
    def batch_mirror(batch_signals):
        mirror_signals = []
        for signal in batch_signals:
            mirror_signals.append(signal[::-1])
        return mirror_signals

    @staticmethod
    def read_signal_signals(fpath, skip_row=1):
        """read signal values"""
        signals = []
        with open(fpath, 'r') as fr:
            signals = list(map(str.rstrip, fr.readlines()))[skip_row:]
        return list(map(float, signals))


    @staticmethod
    def read_batch_signals(index_fpath, length=1024, with_label=True):
        """
        :param input_dir: signal-index dir
        :param input_fname: signal-index fpath
        :return signals, labels, lengths, max_length
        """
        ins_df = pd.read_csv(index_fpath)
        signal_paths = ins_df[Features.DEFAULT_FILEPATH_NAME].to_list()
        labels = ins_df[Features.DEFAULT_LABEL_NAME].to_list()
        abs_ins_dir = Features.abs_instance_root(index_fpath)

        signals = []
        lengths = []
        max_length = 0
        for relpath in signal_paths:
            abs_case_path = abs_ins_dir + os.sep + relpath
            signal = DeepFeatureExtractor.read_signal_signals(abs_case_path)[0:length]
            if len(signal) < length:
                print("signal [%s] length less than [%s]" % (abs_case_path, length))
                continue
            signals.append(signal)
            lengths.append(len(signal))
            max_length = max(max_length, len(signal))
        # return np.array(signals), np.array(labels), np.array(lengths), max_length
        return np.array(signals), np.array(labels), np.array(lengths), max_length


    @staticmethod
    def biread_batch_signals(index_fpath, length=1024, with_label=True):
        signals, labels, lengths, max_length = DeepFeatureExtractor.read_batch_signals(index_fpath, length, with_label)
        # doing mirror operatioin
        mirror_signals = DeepFeatureExtractor.batch_mirror(signals)
        signals.extend(mirror_signals)
        labels.extend(copy.deepcopy(labels))
        lengths.extend(copy.deepcopy(lengths))
        return signals, labels, lengths, max_length

if __name__ == "__main__":
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
        seq_len, \
    split_channel =\
        DeepFeatureExtractor.batch_features(train_ins_fpath, validate_ins_fpath, test_ins_fpath, feature_list, 4)

    print(manual_train_label.shape)
    print(manual_train_label)
