import os.path as osp

import torch
import torch.utils.data as data
import numpy as np
import numpy.linalg as LA

__all__ = [
    'RNNDataLayer',
]

class RNNDataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.sessions = getattr(args, phase+'_session_set')
        self.features = args.features

        self.inputs = []
        for session_name in self.sessions:
            session_path = osp.join(self.data_root, 'target', session_name+'.txt')
            session_data = open(session_path, 'r').read().splitlines()
            self.inputs.extend(session_data)

    def rnn_loader(self, path, number):
        data_path = osp.join(self.data_root, 'slices_npy_64x64', path)
        data = np.load(osp.join(data_path, number.zfill(5)+'.npy'))
        data = (data - 0.5) / 0.5
        norm = LA.norm(data, axis=0)
        data /= norm[None, :]
        init_path = osp.join(self.data_root, self.features, path)
        init = np.load(osp.join(init_path, number.zfill(5)+'.npy'))
        return data, init

    def __getitem__(self, index):
        path, number, air_target, bed_target = self.inputs[index].split()
        data, init = self.rnn_loader(path, number)
        data = torch.from_numpy(data)
        init = torch.from_numpy(init)
        air_target = np.array(air_target.split(','), dtype=np.float32)
        air_target = torch.from_numpy(air_target)
        bed_target = np.array(bed_target.split(','), dtype=np.float32)
        bed_target = torch.from_numpy(bed_target)
        return data, init, air_target, bed_target

    def __len__(self):
        return len(self.inputs)
