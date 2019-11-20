import torch.nn as nn
import torch.utils.data as data

from datasets import build_dataset


__all__ = [
    'build_data_loader',
    'weights_init',
    'count_parameters',
]

def build_data_loader(args, phase='train'):
    data_loaders = data.DataLoader(
        build_dataset(args, phase),
        batch_size=args.batch_size,
        shuffle=phase=='train',
        num_workers=args.num_workers,
    )
    return data_loaders

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1.0, std=0.001)
        nn.init.constant_(m.bias.data, 0.001)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
