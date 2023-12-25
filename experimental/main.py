import argparse
import os

import tensorflow as tf
from tensorflow import keras
import torch
from torch.distributed import rpc

import parameter_server


# parser
## three parts: ['RPC_setting', 'high_level_control_options', 'low_level_control_options']
## 'low_level_control_options' is located in 'parameter_server.py'
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
    pass
parser = argparse.ArgumentParser(
    description='Progressive Dual Batch Size Deep Learning for Distributed Parameter Server Systems',
    epilog=(
        'The parser only supports high-level control options. '
        'If the user wants to adjust low-level control options, modify the code. '
        'Required settings [--rank, --world-size, --num-small, --server-addr, --dataset, --dir-path] '
        'or [-r, -w, -s, -a, -d, -p], '
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
    '--num-small', '--small', '-s',
    dest='small',
    default=0,
    type=int,
    help='number of small-batch workers in the process, the default is "0"',
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
    '--device-index',
    type=int,
    default=0,
    help='the index of the GPU used to run the program, "0" or "-1" is a good choice',
)
parser.add_argument(
    '--depth',
    type=int,
    default=18,
    help='resnet depth, currently supports [18, 34]',
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
                args=(args, ps_rref, i, True if i <= args.small else False),
            )
        )
    torch.futures.wait_all(future_list)
    if args.save:
        ps_rref.rpc_sync().save_history()
    print('Complete, End Program')

def run_worker(args, ps_rref, rank, is_small_batch):
    worker = parameter_server.Worker(args, ps_rref, rank, is_small_batch)
    worker.train()
    print(f'Worker {rank} Training Complete')


# main
def main():
    # parse args
    args = parser.parse_args()
    if args.rank == None:
        raise ValueError('"rank" argument is required')
    if args.world_size == None:
        raise ValueError('"world_size" argument is required')
    if args.addr == None:
        raise ValueError('"master_addr" argument is required')
    print('----')
    print(args)
    print('----')

    # amp, xla
    if args.xla:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
        tf.config.optimizer.set_jit('autoclustering')
        print(f'Optimizer set_jit: "{tf.config.optimizer.get_jit()}"')
    if args.amp:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print(f'Policy: {policy.name}')
        print(f'Compute dtype: {policy.compute_dtype}')
        print(f'Variable dtype: {policy.variable_dtype}')
    print('----')
    print(f'MIXED_PRECISION: {args.amp}')
    print(f'JIT_COMPILE: {args.xla}')
    print('----')

    # GPU initialization
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[args.device_index], 'GPU')
    for device in tf.config.get_visible_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    print('----')
    print(f'The Number of Available Physical Devices: {len(physical_devices)}')
    print(f'Using Devices: {tf.config.get_visible_devices("GPU")}')
    print('----')

    """
    # RPC
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
    # [rank, world_size, small, addr, port,
    #  dataset, dir_path, amp, xla, comments,
    #  device_index, depth, cycle, temp, save]
    main()
