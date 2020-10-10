#!-*- encoding: utf-8 -*-
import os, sys


# 训练数据索引数据均使用相对路径进行表示，保证进行各种目录拷贝之后仍能继续运行
# PROJECT_ROOT = os.path.abspath(os.getcwd())
# RELATIVE_PROJECT_ROOT = os.path.relpath(PROJECT_ROOT, SCRIPT_DIR)
# print("rel root:", REL_ROOT)
# 当前文件所在的位置锚点
SCRIPT_DIR = os.path.dirname(__file__)
# 实际的工程启动位置
PROJECT_ROOT = "."
# 从当前文件所在的目录结构反推应该启动的工程位置
EXPECT_PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# if os.path.abspath(PROJECT_ROOT) != EXPECT_PROJECT_ROOT:
#     raise Exception("命令启动根目录与预期不符，预期:" + EXPECT_PROJECT_ROOT + " 实际:" + os.path.abspath(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT + os.sep + 'data'
# Get Configs Via Version Key
"""
按照高低波的要求重新做的一版本新的错误分类数据
使用三厂的实际线上测试数据进行分类，需要进行数据分析
"""
KIND_DEF = {
    u"二等波": -2,
    u"一等波": -1,
    u"轻微不良": 1,
    u"异常": 2,           #特指无规律的异常波形
    u"无头": 3,
    u"长短": 4,
    u"拉虚": 5,
    u"斜角": 6,
    u"高低": 7,           #默认下高低脚
    u"转速不合格": 8,
    u"其它": 9,
    u"胶水": 10,
    u"NG一等波":11,       #误判为次品的一等波
    u"上高低": 12
}

INVERSE_KIND_DEF = dict()
for key, val in KIND_DEF.items():
    INVERSE_KIND_DEF[val] = key


# benchmark版本，基本覆盖所有错误样本，可作为benchmark方案进行index的创建
config_v20200718 = {
    'directory': DATA_ROOT,
    "vdir": "v20200718",
    'desc': """1. v20200718 是加入底层硬件波形平滑过滤之后重新整理的数据集合，
               2. 按照常见的错误分类进行采集数据的分组，并添加了大量的上高低和下高低数据
               作为benchmark重新进行波形训练，用于后续版本的进一步迭代
            """,
    'tag_info': [
        {"dname": u"二等波", "kind": KIND_DEF[u"二等波"], "label": 0},
        {"dname": u"一等波", "kind": KIND_DEF[u"一等波"], "label": 0},
        {"dname": u"胶水", "kind": KIND_DEF[u"胶水"], "label": 1},
        {"dname": u"拉虚", "kind": KIND_DEF[u"拉虚"], "label": 1},
        {"dname": u"上高低脚", "kind": KIND_DEF[u"上高低"], "label":1},
        {"dname": u"无头", "kind": KIND_DEF[u"无头"], "label":1},
        {"dname": u"下高低脚", "kind": KIND_DEF[u"高低"], "label":1},
        {"dname": u"斜角", "kind": KIND_DEF[u"斜角"], "label":1},
        {"dname": u"异常", "kind": KIND_DEF[u"异常"], "label":1},
        {"dname": u"长短", "kind": KIND_DEF[u"长短"], "label":1},
        {"dname": u"NG一等波", "kind": KIND_DEF[u"NG一等波"], "label":1}
    ]
}

INDEX_CONFIG = {
    'v20200718': config_v20200718
}


if __name__ == '__main__':
    print(INDEX_CONFIG)
    print(SCRIPT_DIR)

