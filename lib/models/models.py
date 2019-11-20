from .cnn import C3D
from .rnn import RNN

__all__ = ['build_model']

# TODO: Add c2d
_META_ARCHITECTURE = {
    'C3D': C3D,
    'RNN': RNN,
}

def build_model(args):
    meta_arch = _META_ARCHITECTURE[args.model]
    return meta_arch()

