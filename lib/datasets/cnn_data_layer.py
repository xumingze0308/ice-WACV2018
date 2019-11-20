import os.path as osp

import torch
import torch.utils.data as data
import numpy as np

__all__ = ['CNNDataLayer']

def c3d_loader(path, number):
    data = []
    for index in range(number - 2, number + 3):
        index = min(max(index, 1), 3332)
        data.append(np.load(osp.join(path, str(index).zfill(5)+'.npy')))
    data = np.array(data, dtype=np.float32)
    data = (data - 0.5) / 0.5
    data = data[np.newaxis, ...]
    return data

# TODO: Add c2d loader
_DATA_LOADERS = {
    'C3D': c3d_loader,
}

class CNNDataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.phase = phase
        if self.phase == 'extract':
            self.sessions = args.train_session_set + args.test_session_set
        else:
            self.sessions = getattr(args, phase+'_session_set')
        self.loader = _DATA_LOADERS[args.model]

        self.inputs = []
        for session_name in self.sessions:
            session_path = osp.join(self.data_root, 'target', session_name+'.txt')
            session_data = open(session_path, 'r').read().splitlines()
            self.inputs.extend(session_data)

    def __getitem__(self, index):
        data_path, number, air_target, bed_target = self.inputs[index].split()
        data = self.loader(osp.join(self.data_root, 'slices_npy_64x64', data_path), int(number))
        data = torch.from_numpy(data)
        air_target = np.array(air_target.split(','), dtype=np.float32)
        air_target = torch.from_numpy(air_target)
        bed_target = np.array(bed_target.split(','), dtype=np.float32)
        bed_target = torch.from_numpy(bed_target)

        if self.phase == 'extract':
            save_path = osp.join(self.data_root, 'c3d_features', data_path, number.zfill(5)+'.npy')
            return data, air_target, bed_target, save_path
        else:
            return data, air_target, bed_target

    def __len__(self):
        return len(self.inputs)
