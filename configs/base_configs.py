import argparse

__all__ = ['parse_base_args']

def parse_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/CReSIS', type=str)
    parser.add_argument('--data_info', default='data/ice.json', type=str)
    parser.add_argument('--checkpoint', default='model_zoo/c3d.pth', type=str)
    parser.add_argument('--split', default='split1', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--test_interval', default=1, type=int)
    return parser

