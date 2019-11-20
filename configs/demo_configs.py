from configs import parse_base_args, build_data_info

__all__ = ['parse_demo_args']

def parse_demo_args():
    parser = parse_base_args()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--c3d_pth', default='model_zoo/c3d.pth', type=str)
    parser.add_argument('--rnn_pth', default='model_zoo/rnn.pth', type=str)
    return build_data_info(parser.parse_args())

