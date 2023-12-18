import argparse
import itertools

import torch
from torch.distributed import rpc

import parameter_server


# parser
## three parts: ['RPC_setting', 'high_level_control_options', 'low_level_control_options']
## each 'control_options' have four parts: ['device', 'dataset_and_model', 'training', 'output_file']
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
    pass
parser = argparse.ArgumentParser(
    description='Progressive Dual Batch Size Deep Learning for Distributed Parameter Server Systems',
    epilog=(
        'The parser only supports high-level control options. '
        'If the user wants to adjust low-level control options, modify the code. '
        'Required settings [--rank, --world-size, --server-addr, --dataset, --dir-path] '
        'or [-r, -w, -a, -d, -p], '
        'optional settings [--amp, --xla, --depth, --server-port, --no-cycle, --no-temp, --no-save].'
    ),
    formatter_class=CustomFormatter,
)
## RPC setting
parser.add_argument(
    '--rank', '-r',
    type=int,
    help='global ranking of the process, pass in 0 for master, and pass in others for workers',
)
parser.add_argument(
    '--world-size', '-w',
    type=int,
    help='total number of servers participating in the process',
)
parser.add_argument(
    '--server-addr', '--addr', '-a',
    dest='addr',
    type=str,
    help='the master address of the parameter server',
)
parser.add_argument(
    '--server-port', '--port',
    dest='port',
    default='48763',
    type=str,
    help='the port that the master listens to, the default is "48763"',
)
## high-level control options
parser.add_argument(
    '--dataset', '--data', '-d',
    type=str,
    help='dataset to train, currently supports ["cifar10", "cifar100", "imagenet"]',
)
parser.add_argument(
    '--dir-path', '--path', '-p',
    type=str,
    help='path to the dataset directory',
)
parser.add_argument(
    '--depth',
    type=int,
    default=18,
    help='resnet depth, currently supports [18, 34]',
)
parser.add_argument(
    '--mixed-precision', '--amp',
    dest='amp',
    action='store_true',
    help='train with mixed precision (amp)',
)
parser.add_argument(
    '--jit-compile', '--xla',
    dest='xla',
    action='store_true',
    help='train with jit compile (xla)',
)
parser.add_argument(
    '--comments', '-c',
    type=str,
    help='add additional comments on filename',
)
parser.add_argument(
    '--no-cycle',
    dest='cycle',
    action='store_false',
    help='do not use all image resolutions with different learning rates',
)
parser.add_argument(
    '--no-temp',
    dest='temp',
    action='store_false',
    help='do not save the temporary state during training, including "_model" and ".npy"',
)
parser.add_argument(
    '--no-save',
    dest='save',
    action='store_false',
    help='do not save the training results, including "_model" and ".npy"',
)

# running server && worker
def run_server(args):
    ps_rref = rpc.RRef(parameter_server.Server(args))
    future_list = []
    for i in range(1, args.world_size):
        future_list.append(
            rpc.rpc_async(
                f'worker_{i}',
                run_worker,
                args=(ps_rref, i, args),
            )
        )
    torch.futures.wait_all(future_list)
    if args.save:
        ps_rref.rpc_sync().save_history()
    print('Complete, End Program')

def run_worker(ps_rref, rank, args):
    worker = parameter_server.Worker(ps_rref, rank, args)
    worker.train()
    print('Training Complete')


# main
def main():
    # get args
    args = parser.parse_args()
    print(args)

    if args.rank == None:
        raise ValueError('"rank" argument is required')
    if args.world_size == None:
        raise ValueError('"world_size" argument is required')
    if args.addr == None:
        raise ValueError('"master_addr" argument is required')
    dataset_list = ['cifar10', 'cifar100', 'imagenet']
    if args.dataset not in dataset_list:
        raise ValueError(f'Invalid dataset "{args.dataset}", it should be in {dataset_list}.')
    
    '''
    # extend and rebuild args 1
    batch_size_ls = [1000, 500] if 'cifar' in args.dataset else [510, 360, 170]
    resolution_ls = [24, 32] if 'cifar' in args.dataset else [160, 224, 288]
    dropout_rate_ls = [0.1, 0.2] if 'cifar' in args.dataset else [0.1, 0.2, 0.3]
    parser.add_argument(
        '--batch-size-iter',
        default=itertools.cycle(batch_size_ls)
    )
    parser.add_argument(
        '--resoultion-iter',
        default=itertools.cycle(resolution_ls)
    )
    parser.add_argument(
        '--dropout-rate-iter',
        default=itertools.cycle(dropout_rate_ls)
    )
    parser.add_argument(
        '--milestones',
        default=list(int(args.epochs * i / args.step) for i in range(1, args.step))
    )
    args = parser.parse_args()
    
    # extend and rebuild args 2
    parser.add_argument(
        '--modify-freq',
        default=int((args.milestones[0] if args.cycle else args.epochs) / len(resolution_ls))
    )
    parser.add_argument(
        '--outfile',
        default=(
            f'{args.dataset}_resnet{args.depth}_{args.epochs}'
            f'{"_amp" if args.amp else ""}'
            f'{"_xla" if args.xla else ""}'
            f'{"" if args.cycle else "_nocycle"}'
        )
    )
    args = parser.parse_args()
    print(args)
    '''

"""
    # RPC
    ########os.environ['MASTER_ADDR'] = args.addr
    ########os.environ['MASTER_PORT'] = args.port
    backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=f'tcp://{args.addr}:{args.port}',
    )
    if args.rank == 0:  # server
        print(f'Server {args.rank} initializing RPC')
        rpc.init_rpc(
            name=f'server_{args.rank}',
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=backend_options,
        )
        run_server(args)
    else:               # worker
        print(f'Worker {args.rank} initializing RPC')
        rpc.init_rpc(
            name=f'worker_{args.rank}',
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=backend_options,
        )
    rpc.shutdown()
"""

if __name__ == '__main__':
    # args:
    # [rank, word_size, addr, port,
    #  dataset, dir_path, amp, xla, comments,
    #  depth, cycle, temp, save]
    main()
