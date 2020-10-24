import os
import xgboost
from sklearn.metrics import classification_report
from sc.deep_predict import DeepPredictor
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from instance import config as insconf
import warnings
import sc.predict_test
from configs import FEATURE_CONFIG
from features import Features
from sc.signal_manager import SignalMgr
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


# def xgboost_predict(train_x, train_y, test_x, test_y):
#     # model = joblib.load("./models/xgboost_model/model.pkl")
#     # train_x, train_y, test_x, test_y, kind = train_model_data(feature_name)
#
#     # print("learning_rate=", i)
#     # model = xgboost.XGBClassifier(colsample_bytree=0.58, subsample=0.56, learning_rate=1.0, max_depth=6)
#     model = xgboost.XGBClassifier(scale_pos_weight=1.1)
#     # model = xgboost.XGBClassifier(learning_rate=i)
#     model.fit(train_x.values, train_y.values, verbose=True)
#
#     # joblib.dump(model, './model.xgboost')
#     # joblib.dump(model, '/Users/changkong/project/deepSignalFlawDetect/models/xgboost_model/model.pkl')
#     # newModel = joblib.load('./model.xgboost')
#     pResult = model.predict(test_x.values)
#     print(classification_report(test_y, pResult))
#     # ret = sc.predict_test.benchmark_run(feature_name)
#     # print(ret)
#     # myplt = plot_learning_curve(model, 'training monitoring', train_x, train_y)
#     # myplt.show()
#     return pResult

def xgboost_multi_predict(train_x, train_y, test_x, test_y, weight):
    dtrain = xgboost.DMatrix(train_x.values, label=train_y.values, weight=weight)
    dtest = xgboost.DMatrix(test_x.values, label=test_y.values)
    param = {
        'booster': 'gbtree',
        'objective': 'multi:softprob',
        'num_class': 13,
        'colsample_bytree': 1.0,
        'subsample': 1.0,
        'max_depth': 5,
        'learning_rate': 0.15
    }
    param['eval_metric'] = ['error']
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    xgbModel = xgboost.train(param, dtrain, 300, evallist, early_stopping_rounds=30)
    # xgbModel = joblib.load("./model.xgboost")
    preds = xgbModel.predict(dtest, ntree_limit=xgbModel.best_ntree_limit)
    print(preds)
    # for i in range(31, 51):
    #     print("======= Param: ", i / 100.0)
    #     pResult = [1 if value >= i / 100.0 else 0 for value in preds]
    #     print(pResult)
    #     print(classification_report(test_y, pResult))

    # joblib.dump(xgbModel, './model.xgboost')
    # xgboost.plot_importance(xgbModel, importance_type='gain')
    # plt.show()
    return pResult


# Dmatrix Version
def xgboost_predict(train_x, train_y, test_x, test_y, weight):
    dtrain = xgboost.DMatrix(train_x.values, label=train_y.values, weight=weight)
    dtest = xgboost.DMatrix(test_x.values, label=test_y.values)
    param = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'colsample_bytree': 1.0,
        'subsample': 1.0,
        'max_depth': 5,
        'learning_rate': 0.15
    }
    param['eval_metric'] = ['error']
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    xgbModel = xgboost.train(param, dtrain, 300, evallist, early_stopping_rounds=30)
    # xgbModel = joblib.load("./model.xgboost")
    preds = xgbModel.predict(dtest, ntree_limit=xgbModel.best_ntree_limit)

    for i in range(31, 51):
        print("======= Param: ", i / 100.0)
        pResult = [1 if value >= i / 100.0 else 0 for value in preds]
        print(pResult)
        print(classification_report(test_y, pResult))

    joblib.dump(xgbModel, './model.xgboost')
    # xgboost.plot_importance(xgbModel, importance_type='gain')
    # plt.show()
    return pResult


def deep_predict(test_index_df, abs_root, path_name='filepath'):
    signal_mgr = SignalMgr()
    # deep_predictor
    case_pathes = test_index_df[path_name].tolist()
    deep_predictor = DeepPredictor(os.path.join('models', 'deep_model', '-8729.meta'),
                                   os.path.join('models', 'deep_model'))
    result_df = test_index_df[['kind', 'label']]
    pred = list()
    for path in case_pathes:
        abs_path = abs_root + os.sep + path
        dt, raw_signals = signal_mgr.parse_signals_from_file(abs_path, 1)
        deep_features = signal_mgr.get_features(abs_path,
                                                request_param={'skip_row': [1], 'model_path': ['train']})
        deep_score = deep_predictor.predict(raw_signals, deep_features, 4)
        deep_wave_result = np.argmax(deep_score)
        pred.append(deep_wave_result)
    result_df['pred'] = pred
    print(classification_report(result_df['label'], pred))
    return result_df

def feature_prepare(feature_version, enable_cache=True):
    """
    generate train/test/validate features
    """
    feature_config = FEATURE_CONFIG[feature_version]
    print(feature_config['feature_list'])
    feasets = list()
    feasets.append(Features(feature_config['train_ins']))
    feasets.append(Features(feature_config['test_ins']))
    feasets.append(Features(feature_config['validate_ins']))

    fea_df = list()
    for fea in feasets:
        fea_df.append(fea.generate(feature_config['feature_list']))
        fea_df.append(fea.label)
        fea_df.append(fea.kind)
    return fea_df

def instance_weight(kind):
    weight_map = {
        # -2: 5.0,
        # -1: 1.0,
        # 2: 10.0,
        # 3: 10.0,
        # 4: 10.0,
        # 5: 10.0,
        -2: 10,
        11: 10,
        6: 5.0,
        7: 25.0
        # 10: 10.0,
        # 11: 10.0,
        # 12: 10.0
    }
    equal_mapping = lambda x: 1.0
    weight_mapping = lambda x: weight_map[x] if x in weight_map else 1.0
    return kind.map(weight_mapping)

def signle_xgboost_predict():
    train_fea_df_x, train_fea_df_y, tkind, test_fea_df_x, test_fea_df_y, test_k, validate_fea_df_x, validate_fea_df_y, vk = feature_prepare(
        "feature_v2.2.5")
    weight = instance_weight(tkind)
    test_predict_label = xgboost_predict(train_fea_df_x, train_fea_df_y, test_fea_df_x, test_fea_df_y, weight)
    err_predict_index = (test_predict_label != test_fea_df_y)
    err_kinds_df = test_k[err_predict_index]
    print(err_kinds_df.value_counts())

def multi_xgboost_predict():
    train_fea_df_x, train_fea_df_y, tkind, test_fea_df_x, test_fea_df_y, test_k, validate_fea_df_x, validate_fea_df_y, vk = feature_prepare(
        "feature_v2.2.5")



if __name__ == "__main__":
    # train_fea_df_x, train_fea_df_y, tkind, test_fea_df_x, test_fea_df_y, test_k, validate_fea_df_x, validate_fea_df_y, vk = feature_prepare("feature_v2.2.5")
    train_fea_df_x, train_fea_df_y, tkind, test_fea_df_x, test_fea_df_y, test_k, validate_fea_df_x, validate_fea_df_y, vk = feature_prepare("feature_v3.0.0")
    filter_index = (tkind == -2)
    print(filter_index)
    train_fea_df_x = train_fea_df_x[~filter_index]
    train_fea_df_y = train_fea_df_y[~filter_index]
    tkind = tkind[~filter_index]

    weight = instance_weight(tkind)
    test_predict_label = xgboost_predict(train_fea_df_x, train_fea_df_y, test_fea_df_x, test_fea_df_y, weight)
    # # validate_predict_label = xgboost_predict(train_fea_df_x, train_fea_df_y, validate_fea_df_x, validate_fea_df_y, weight)
    # # train_predict_label = xgboost_predict(train_fea_df_x, train_fea_df_y, train_fea_df_x, train_fea_df_y)

    err_predict_index = (test_predict_label != test_fea_df_y)
    # print(test_k.value_counts())
    # print(insconf.INVERSE_KIND_DEF)
    for k in test_k.tolist():
        if k not in insconf.INVERSE_KIND_DEF:
            print(k, " is not found")

    kind_names = test_k.map(lambda x: insconf.INVERSE_KIND_DEF[x])
    err_dist_df = pd.DataFrame({'kind': kind_names, 'err_predict': err_predict_index.astype(int)})
    gerr_dist_df = err_dist_df.groupby('kind').agg({'err_predict': ['count', 'sum']})
    # flatten columns
    gerr_dist_df.columns = ['_'.join(col).strip() for col in gerr_dist_df.columns.values]
    # print(gerr_dist_df['err_predict']['sum'] * 100.0 / gerr_dist_df['err_predict']['count'])
    gerr_dist_df = gerr_dist_df.reset_index()
    gerr_dist_df['err_rate'] = gerr_dist_df['err_predict_sum'] * 100.0 / gerr_dist_df['err_predict_count']
    print(gerr_dist_df)
    # print(err_dist_df)
    # test_k['kind'] = kind_names
    # err_kinds_df = test_k[err_predict_index]
    # print(err_kinds_df)
    # print(err_kinds_df.value_counts())

    # err_predict_index = (validate_predict_label != validate_fea_df_y)
    # err_kinds_df = vk[err_predict_index]
    # print(err_kinds_df.value_counts())

    # === deep predictions ====
    # feature_config = FEATURE_CONFIG["feature_v2.2.5"]
    # absroot = Features.abs_instance_root(feature_config['test_ins'])
    # test_index_df = pd.read_csv(feature_config['test_ins'])
    # print(test_index_df.head())
    # pResult = deep_predict(test_index_df, abs_root=absroot)
    # print(pResult)
    # err_predict_index = (pResult['label'] != pResult['pred'])
    # vk = pResult['kind']
    # err_kinds_df = vk[err_predict_index].value_counts()
    # print(err_kinds_df)




