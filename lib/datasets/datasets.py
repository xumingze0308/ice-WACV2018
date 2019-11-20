from .cnn_data_layer import CNNDataLayer
from .rnn_data_layer import RNNDataLayer

__all__ = ['build_dataset']

# TODO: Add c2d
_DATA_LAYER = {
    'C3D': CNNDataLayer,
    'RNN': RNNDataLayer,
}

def build_dataset(args, phase):
    data_layer = _DATA_LAYER[args.model]
    return data_layer(args, phase)

