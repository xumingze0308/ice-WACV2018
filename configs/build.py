import os.path as osp
import json

__all__ = ['build_data_info']

def build_data_info(args):
    with open(args.data_info, 'r') as f:
        data_info = json.load(f)[args.split]
    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    return args

