#! -*- encoding: utf8 -*-

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('server')

GLOBAL_MODEL_VERSION = "skew_enhance"

ModelVersionFeatureConfig = {
        "gbdt_smooth_signal": {"features": ['peaks_num', 'down_peaks_num', 'up_edges_num', 'down_edges_num', 'peak_edge_ratio', 'down_peak_edge_ratio','edge_diff_10', 'edge_diff_20', 'width_diff_10', 'negative_peak_num', 'max_down_peak_point', 'inter_diff_mean', 'inter_diff_delta','skewness_mean', 'skewness_delta', 'cyclic_intense_nopeak', 'cyclic_intense_downpeak'],
                               "path": "gbdt_model",
                               "norm_mean": 0.5060711372944259,
                               "norm_std": 0.17538730539850703
                               },
        "skew_enhance": {"features": ['peaks_num', 'down_peaks_num', 'up_edges_num', 'down_edges_num', 'peak_edge_ratio', 'down_peak_edge_ratio','edge_diff_10', 'edge_diff_20', 'width_diff_10', 'negative_peak_num', 'max_down_peak_point', 'inter_diff_mean', 'inter_diff_delta','skewness_mean', 'skewness_delta', 'cyclic_intense_nopeak', 'cyclic_intense_downpeak', 'skew_level_1', 'skew_level_2', 'skew_level_3', 'skew_level_4'],
                         "path": "gbdt_model",
                         "norm_mean": 0.5060711372944259,
                         "norm_std": 0.17538730539850703
                         },
        "skew_valley_diff_enhance": {"features": ['peaks_num', 'down_peaks_num', 'up_edges_num', 'down_edges_num', 'peak_edge_ratio', 'down_peak_edge_ratio','edge_diff_10', 'edge_diff_20', 'width_diff_10', 'negative_peak_num', 'max_down_peak_point', 'inter_diff_mean', 'inter_diff_delta','skewness_mean', 'skewness_delta', 'cyclic_intense_nopeak', 'cyclic_intense_downpeak', 'skew_level_1', 'skew_level_2', 'skew_level_3', 'skew_level_4', 'valley_height_diff'],
                                     "path": "xgboost_model", "norm_mean": 0.5060711372944259, "norm_std": 0.17538730539850703
                         },
        "neck_valley_diff_enhance": {"features": ['peaks_num', 'down_peaks_num', 'up_edges_num', 'down_edges_num', 'peak_edge_ratio', 'down_peak_edge_ratio','edge_diff_10', 'edge_diff_20', 'width_diff_10', 'negative_peak_num', 'max_down_peak_point', 'inter_diff_mean', 'inter_diff_delta','skewness_mean', 'skewness_delta', 'cyclic_intense_nopeak', 'cyclic_intense_downpeak', 'skew_level_1', 'skew_level_2', 'skew_level_3', 'skew_level_4', 'valley_height_diff', 'neck_height_diff'],
                                     "path": "xgboost_model", "norm_mean": 0.5060711372944259, "norm_std": 0.17538730539850703},
        "skew_valley_diff_with_edge_height": {"features": ['paired_edge_num', 'paired_edge_avg_height', 'peaks_num', 'down_peaks_num', 'up_edges_num', 'down_edges_num', 'peak_edge_ratio', 'down_peak_edge_ratio','edge_diff_10', 'edge_diff_20', 'width_diff_10', 'negative_peak_num', 'max_down_peak_point', 'inter_diff_mean', 'inter_diff_delta','skewness_mean', 'skewness_delta', 'cyclic_intense_nopeak', 'cyclic_intense_downpeak', 'skew_level_1', 'skew_level_2', 'skew_level_3', 'skew_level_4', 'valley_height_diff'],
                                     "path": "xgboost_model", "norm_mean": 0.5060711372944259, "norm_std": 0.17538730539850703},
        "glue_enhance": {"features": ['paired_edge_num', 'paired_edge_avg_height', 'peaks_num', 'down_peaks_num', 'up_edges_num', 'down_edges_num', 'peak_edge_ratio', 'down_peak_edge_ratio','edge_diff_10', 'edge_diff_20', 'width_diff_10', 'negative_peak_num', 'max_down_peak_point', 'inter_diff_mean', 'inter_diff_delta','skewness_mean', 'skewness_delta', 'cyclic_intense_nopeak', 'cyclic_intense_downpeak', 'skew_level_1', 'skew_level_2', 'skew_level_3', 'skew_level_4', 'valley_height_diff',
                                      'level1_down_peaks_num', 'level2_down_peaks_num', 'cyclic_intense_level1_downpeak', 'cyclic_intense_level2_downpeak', 'neck_height_diff'],
                                     "path": "xgboost_model", "norm_mean": 0.5060711372944259, "norm_std": 0.17538730539850703}
}
