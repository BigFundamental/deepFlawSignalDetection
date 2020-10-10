import os, sys
import getopt
# from dataset import data_reader, index_maker
from dataset.index_maker import IndexMaker
from dataset.data_maker import DataMaker
from config import INDEX_CONFIG
from strategy import STRATEGY_CONFIG

def print_help():
    print("""Usage: 
    python %s: --make --output_dir [output directory] --config [config_name]""" % (__file__))


def run_make_index(params):
    """
    Create Dataframe for all data path
    :param params: params to output from current values
    :return: dataframe structure
    """
    # check config names:
    config_name = params['config_name'] if 'config_name' in params else None
    if config_name is None:
        raise Exception("config name is missing")
    if config_name not in INDEX_CONFIG:
        raise Exception("Index Config %s not exists" % (config_name))

    cfg = INDEX_CONFIG[config_name]
    imaker = IndexMaker()

    # Adding Index Information into index maker
    # Change data directory by versionId
    root_dir = cfg['directory'] + os.sep + cfg['vdir']
    for sub_ins in cfg['tag_info']:
        subdir_name = sub_ins['dname']
        subcase_dir = root_dir + os.sep + subdir_name
        kind = sub_ins['kind']
        label = sub_ins['label']
        imaker.addDirectory(dir=subcase_dir, label=label, kind=kind)

    root_data, case_info = imaker.create_dataset_index()
    # print(case_info.head())
    imaker.describe()
    # print(imaker.dir_meta.keys())
    imaker.dump_index(output_dir=params['output'])
    return

def run_make_instance(params):
    """
    Load data from index and seperate it into train / test / validate [optional] three parts
    params:
    strategy_name: configurations for train/test/validation dataset splitting strategy

    :return:
    """
    strategy_name = params['strategy_name'] if 'strategy_name' in params else None
    idx_dir = params['input'] if 'input' in params else None
    output_dir = params['output'] if 'output' in params else None

    if strategy_name is None:
        raise Exception("strategy name is missing")
    if strategy_name not in STRATEGY_CONFIG:
        raise Exception("strategy config %s is not existed" % strategy_name)

    # load existing dirs
    imaker = IndexMaker()
    _, df = imaker.load_index(idx_dir).create_dataset_index()

    # data divide strategies
    strategy = STRATEGY_CONFIG[strategy_name]
    dmaker = DataMaker()
    dmaker.run(df, strategy)
    print(dmaker.describe())
    dmaker.to_csv(output_dir)

def main():
    """
    build data process command entries
    :return:
    """
    class ParamExtractor(object):
        @staticmethod
        def make_index_parameters(ops, args):
            output_path_param = None
            config_name = None
            for op, val in ops:
                if op in ['-o', '--output']:
                    output_path_param = val
                if op in ['-c', '--config']:
                    config_name = val
            return {'output': output_path_param,
                    'config_name': config_name}

        @staticmethod
        def make_instance_parameters(ops, args):
            input_index_fpath = None
            output_path_param = None
            strategy_name = None
            for op, val in ops:
                if op in ['-o', '--output']:
                    output_path_param = val
                if op in ['-s', '--strategy_name']:
                    strategy_name = val
                if op in ['-i', '--input']:
                    input_index_fpath = val

            return {'output': output_path_param,
                    'strategy_name': strategy_name,
                    'input': input_index_fpath}

    try:
        opts, args = getopt.getopt(sys.argv[1:], shortopts='mo:c:ds:i:', longopts=['make', 'output=', 'config=', 'mdata',
                                                                                 'strategy_name=', 'input='])
        if len(opts) == 0:
            raise getopt.GetoptError("No parameter is specified")

        for key, value in opts:
            if key in ['-m', '--make']:
                print("Start Making Indexes... input params:", opts)
                params = ParamExtractor.make_index_parameters(opts, args)
                run_make_index(params)
                return
            if key in ['-d', '--mdata']:
                print("Start Making Training Instances... input params:", opts)
                params = ParamExtractor.make_instance_parameters(opts, args)
                run_make_instance(params)
                return

    except getopt.GetoptError as e:
        print("GetOptError: ", e)
        print_help()

    return

if __name__ == '__main__':
    main()

