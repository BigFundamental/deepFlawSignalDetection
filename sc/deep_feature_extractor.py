#!-*- encoding: utf-8 -*-
"""
author: zhou lin
date: 2017-01-13
brief: signal pattern extractors
"""

import numpy as np
from sc.filter import Filter
from sc.signal_manager import SignalMgr


class DeepFeatureExtractor(object):
    """
    Deep Signal Feature Extractors
    """

    def __init__(self):
        pass

    @staticmethod
    def features(raw_signal, norm_feas, n_channel=4):
        norm_signal = np.array([DeepFeatureExtractor.get_norm_signals(raw_signal)])
        # print("norm_signal shape:", norm_signal.shape)
        medfilter_signal = np.array([DeepFeatureExtractor.get_medfilter_signals(raw_signal)])
        # print("medfilter_signal shape:", medfilter_signal.shape)
        res_signal = np.array(DeepFeatureExtractor.get_res_signals(raw_signal, medfilter_signal))
        # print("res signal shape:", res_signal.shape)
        bottom_shapes = np.array([DeepFeatureExtractor.get_bottom_shape_signals(raw_signal, norm_feas)])
        # print("bottom signal shape:", bottom_shapes.shape)
        signal_len = len(raw_signal)
        return DeepFeatureExtractor.stacked_channel_signals_((norm_signal, medfilter_signal, res_signal, bottom_shapes),
                                                             signal_len, n_channel)

    @staticmethod
    def stacked_channel_signals_(signal_vec, signal_len, n_fold=4):
        stacked_signals = np.stack(signal_vec, axis=-1)
        # print("stack signal shape", stacked_signals.shape)
        x, y, z = stacked_signals.shape[0], stacked_signals.shape[1] // n_fold, stacked_signals.shape[2] * n_fold
        # signals.shape[0], signals.shape[1] // split_num, signals.shape[2] * split_num
        return np.reshape(stacked_signals, (x, y, z)), z

    @staticmethod
    def get_norm_signals(raw_signal):
        mu = np.mean(raw_signal)
        delta = np.std(raw_signal)
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

