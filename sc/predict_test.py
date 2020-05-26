from sc.signal_manager import SignalMgr
import warnings
from sc.data_reader import DataReader
import numpy as np

warnings.filterwarnings('ignore')
# pathes = DataReader.search_files('/Users/changkong/Desktop/2.2.4软件误判数据', pattern='.csv')
pathes = DataReader.search_files('/Volumes/workspace/projects/signal_classification/data/特殊次品样本/20200428高低脚102/', pattern='.csv')
# Predict All Cases

def benchmark_run(feature_name):
    path_ng = '/Users/changkong/Desktop/2.2.4软件误判数据/'
    path_uplow = '/Volumes/workspace/projects/signal_classification/data/特殊次品样本/20200428高低脚102/'

    ret1 = benchmark_test(feature_name, path_ng)
    ret2 = benchmark_test(feature_name, path_uplow)
    return dict({'ng_eval': ret1, 'uplow': ret2})

def benchmark_test(feature_name, benchmark_path, enable_detail=False):
    signal_mgr = SignalMgr()
    pathes = DataReader.search_files(benchmark_path, pattern='.csv')
    request_param={'skip_row':[1], 'mode': ['train'], 'model_version':feature_name}
    pred_result = []
    for path in pathes:
        pred_ret = signal_mgr.process(path, request_param)
        pred_result.append(pred_ret['waveResult'])
        if enable_detail:
            print(path, pred_ret['waveResult'], pred_ret['waveScore'])
    #summery output
    # print(pred_result)
    negtive_num = sum(pred_result)
    total_num = len(pred_result)
    neg_ratio = negtive_num * 1.0 / total_num
    return dict({'neg_num': negtive_num, 'total_num': total_num, 'neg_ratio': neg_ratio})


if __name__ == '__main__':
    ret = benchmark_test('skew_valley_diff_enhance', '/Users/changkong/Desktop/2.2.4软件误判数据', True)
    print(ret)