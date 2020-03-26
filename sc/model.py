#! -*- encoding: utf8 -*-

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('server')

ModelVersionFeatureConfig = {
        "gbdt_smooth_signal": {"features": ['peaks_num', 'down_peaks_num', 'up_edges_num', 'down_edges_num', 'peak_edge_ratio', 'down_peak_edge_ratio','edge_diff_10', 'edge_diff_20', 'width_diff_10', 'negative_peak_num', 'max_down_peak_point', 'inter_diff_mean', 'inter_diff_delta','skewness_mean', 'skewness_delta', 'cyclic_intense_nopeak', 'cyclic_intense_downpeak'], "path": "gbdt_model"},
        "skew_enhance": {"features": ['peaks_num', 'down_peaks_num', 'up_edges_num', 'down_edges_num', 'peak_edge_ratio', 'down_peak_edge_ratio','edge_diff_10', 'edge_diff_20', 'width_diff_10', 'negative_peak_num', 'max_down_peak_point', 'inter_diff_mean', 'inter_diff_delta','skewness_mean', 'skewness_delta', 'cyclic_intense_nopeak', 'cyclic_intense_downpeak', 'skew_level_1', 'skew_level_2', 'skew_level_3', 'skew_level_4'], "path": "gbdt_model"}
}
