#!-*- encoding: utf-8 -*-
import os, sys

"""
定义样本数据的筛选细节，务必保证样本数据能够去除掉随机性的影响，能够复现
1. 最基本的方式就是定义训练、测试、验证集合各自的比例
2. 根据某个字段进行筛选样本，一般是使用label来进行筛选
3. random_seeds就是用来进行随机样本的筛选判断
4. sample_by一般用于指定的字段
"""

# TODO: 后续sample有可能会出现根据样本进行筛选的情况，按照数量的方式去选例如正负样本偏差实在太大，需要过采样或者欠采样，或者reweight
# 样本均衡或者不均衡需要判断的问题
strategy_v2020809 = {
    'train_ratio': 0.75,
    'test_ratio': 0.15,
    'validate_ratio': 0.1,
    'random_seeds': 816,
    'monitor_tags': ["label", "kind"]
}

STRATEGY_CONFIG = {
    'v20200809': strategy_v2020809
}

