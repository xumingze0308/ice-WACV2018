from __future__ import print_function
import os
import os.path as osp
import sys

import torch
import torch.nn as nn
import numpy as np

import _init_paths
import utils as utl
from configs import parse_c3d_args as parse_args
from models import build_model

def main(args):
    this_dir = osp.join(osp.dirname(__file__), '.')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = utl.build_data_loader(args, 'extract')

    model = build_model(args).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.train(False)

    with torch.set_grad_enabled(False):
        for batch_idx, (data, air_target, bed_target, save_path) in enumerate(data_loader):
            print('{:3.3f}%'.format(100.0*batch_idx/len(data_loader)))
            batch_size = data.shape[0]
            data = data.to(device)
            air_feature, bed_feature = model.features(data)
            air_feature = air_feature.to('cpu').numpy()
            bed_feature = bed_feature.to('cpu').numpy()
            for bs in range(batch_size):
                if not osp.isdir(osp.dirname(save_path[bs])):
                    os.makedirs(osp.dirname(save_path[bs]))
                np.save(
                    save_path[bs],
                    np.concatenate((air_feature[bs], bed_feature[bs]), axis=0)
                )

if __name__ == '__main__':
    main(parse_args())
