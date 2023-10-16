import argparse
import itertools
import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
from torch.distributed import rpc

import parameter_server


# parser
parser = argparse.ArgumentParser(description='Progressive Dual Batch Size Deep Learning for Distributed Parameter Server Systems')
## RPC setting
parser.add_argument(
    '--rank', '-r',
    metavar='NUM',
    type=int,
    help='global ranking of the process, pass in 0 for master, and pass in others for workers',
)
parser.add_argument(
    '--world-size', '-w',
    metavar='NUM',
    type=int,
    help='total number of servers participating in the process',
)
parser.add_argument(
    '--server-addr', '-a',
    metavar='ADDRESS',
    dest='addr',
    type=int,
    help='the master address of the parameter server',
)
parser.add_argument(
    '--server-port', '-p',
    metavar='PORT',
    dest='port',
    default=48763,
    type=int,
    help='the port that the master listens to, the default is "48763"',
)
## high-level control options
### device
parser.add_argument(
    '--multi-gpu',
    action='store_true',
    help='training using multiple GPUs, not yet completed',
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
### dataset and model
parser.add_argument(
    '--dataset',
    metavar='DATASET',
    type=str,
    help='dataset to train, currently supports ["cifar10", "cifar100", "imagenet"]',
)
parser.add_argument(
    '--dir-path', '--path',
    metavar='PATH/TO/THE/DIR',
    type=str,
    help='path to the dataset directory',
)
### training
parser.add_argument(
    '--cycle',
    action='store_true',
    help='use all image resolutions with different learning rates',
)
### output files
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
## low-level control options
### device
parser.add_argument(
    '--device-index',
    default=0,
)
parser.add_argument(
    '--depth',
    default=18,
)
parser.add_argument(
    '--learning-rate',
    default=1e-1,
)
parser.add_argument(
    '--momentum',
    default=0.9,
)
parser.add_argument(
    '--weight-decay',
    default=1e-4,
    help='it is useless in TF2.6, modify it directly in "tf_data_model.py"',
)
parser.add_argument(
    '--epochs',
    default=90,
)
parser.add_argument(
    '--step',
    default=3,
)
parser.add_argument(
    '--gamma',
    default=0.1,
)
######## build args -> parser add_arg -> re-build args

def run_worker(ps_rref, rank, args):
    worker = parameter_server.Worker(ps_rref, rank, args)
    worker.train()
    print('Training Complete')


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


def main():
    # get args
    ## check the file type is '.py' or '.ipynb'
    ### parse args of '.ipynb' from here
    ### ex. ['--dataset=imagenet', '--path=./dataset', '--cycle', '--amp', '--xla']
    args = (
        parser.parse_args(['--dataset=cifar100', '--cycle', '--amp', '--xla'])
        if len(sys.argv) > 2 and sys.argv[1] == '-f' and '.json' in sys.argv[2]
        else parser.parse_args()
    )
    print(args)

    if args.rank == None:
        raise ValueError('"rank" argument is required')
    if args.world_size == None:
        raise ValueError('"world_size" argument is required')
    if args.addr == None:
        raise ValueError('"master_addr" argument is required')
    
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

if __name__ == '__main__':
    main()
