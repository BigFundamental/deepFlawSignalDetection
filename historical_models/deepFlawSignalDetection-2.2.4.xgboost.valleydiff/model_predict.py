#!-*- encoding: utf-8 -*-
import os, sys
import pandas as pd
from sc.signal_manager import SignalMgr
from sklearn.metrics import classification_report

EVAL_INSTANCE_DIR="/Volumes/workspace/projects/signal_detection/instance"
EVAL_INSTANCE_FPATH=EVAL_INSTANCE_DIR + os.sep + 'test_ins.csv'
META_INSTANCE_FNAME="meta"

def get_abs_instance_fpath(meta_fpath):
    dirname = os.path.dirname(meta_fpath)
    with open(meta_fpath, 'r') as fr:
        key, relroot = fr.readline().strip().split(':')
        if key == 'root':
            return relroot

if __name__ == '__main__':
    params = SignalMgr.signalParams
    params['mode'] = 'train'
    signal_mgr = SignalMgr()

    print(params)

    abs_instance_root = get_abs_instance_fpath(EVAL_INSTANCE_DIR + os.sep + META_INSTANCE_FNAME)
    abs_instance_root = EVAL_INSTANCE_DIR + os.sep + abs_instance_root
    print("instance root: ", abs_instance_root)

    eval_ins_df = pd.read_csv(EVAL_INSTANCE_FPATH)
    case_pathes = eval_ins_df['filepath'].to_list()
    predict_ret = []
    for path in case_pathes:
        try:
            abs_path = abs_instance_root + os.sep + path
            pred_ret = signal_mgr.process(abs_path, params)
            predict_ret.append(pred_ret['waveResult'])
        except:
            print(path)
            break
    eval_ins_df['pred'] = predict_ret
    print(predict_ret)
    print(classification_report(eval_ins_df['label'], eval_ins_df['pred']))

