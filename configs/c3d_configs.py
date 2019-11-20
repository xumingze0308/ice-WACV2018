from configs import parse_base_args, build_data_info

__all__ = ['parse_c3d_args']

def parse_c3d_args():
    parser = parse_base_args()
    parser.add_argument('--model', default='C3D', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-04, type=float)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    return build_data_info(parser.parse_args())

