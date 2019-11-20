from configs import parse_base_args, build_data_info

__all__ = ['parse_rnn_args']

def parse_rnn_args():
    parser = parse_base_args()
    parser.add_argument('--model', default='RNN', type=str)
    parser.add_argument('--features', default='c3d_features', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-03, type=float)
    return build_data_info(parser.parse_args())

