#! -*- encodings: utf-8
from sc.model import ModelVersionFeatureConfig as mc
############  FEATURE CONFIGURATIONS   #############

feature_v224 = {
    "feature_list": mc['skew_valley_diff_enhance']['features'],
    "train_ins": "/Volumes/workspace/projects/signal_detection/instance/train_ins.csv",
    "validate_ins": "/Volumes/workspace/projects/signal_detection/instance/validate_ins.csv",
    "test_ins": "/Volumes/workspace/projects/signal_detection/instance/test_ins.csv"
}

feature_v225 = {
    "feature_list": mc['skew_valley_diff_with_edge_height']['features'],
    "train_ins": "/Volumes/workspace/projects/signal_detection/instance/train_ins.csv",
    "validate_ins": "/Volumes/workspace/projects/signal_detection/instance/validate_ins.csv",
    "test_ins": "/Volumes/workspace/projects/signal_detection/instance/test_ins.csv"
}

feature_v2251 = {
    "feature_list": mc['neck_valley_diff_enhance']['features'],
    "train_ins": "/Volumes/workspace/projects/signal_detection/instance/train_ins.csv",
    "validate_ins": "/Volumes/workspace/projects/signal_detection/instance/validate_ins.csv",
    "test_ins": "/Volumes/workspace/projects/signal_detection/instance/test_ins.csv"
}

feature_vstat = {
    "feature_list":["unit_interviene_length", "paired_edge_height", "paired_edges"],
    "train_ins": "/Volumes/workspace/projects/signal_detection/instance/train_ins.csv",
    "validate_ins": "/Volumes/workspace/projects/signal_detection/instance/validate_ins.csv",
    "test_ins": "/Volumes/workspace/projects/signal_detection/instance/test_ins.csv"
}

FEATURE_CONFIG = {
    "feature_v2.2.4": feature_v224,
    "feature_v2.2.5": feature_v225,
    "feature_v2.2.5.1": feature_v2251,
    "feature_statistic": feature_vstat
}