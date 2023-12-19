import itertools
import threading
import time

from torch.distributed import rpc

import tf_data_model

# args:
# [rank, world_size, small, addr, port,
#  dataset, dir_path, amp, xla, comments,
#  depth, cycle, temp, save]

# Server
class Server(object):
    def __init__(self, args):
        # global setting
        self.args = args
        self.mission_complete = False
        self.start_time = time.perf_counter()
        # training parameters
        self.parameter_lock = threading.Lock()
        self.epochs = 90
        self.steps = 3
        self.mini_epochs = self.epochs * args.word_size
        if args.dataset == 'cifar10' or args.dataset == 'cifar100': 
            self.large_batch_size_ls = [1000, 500]
            self.small_batch_size_ls = []
            self.resolution_ls = [24, 32]
            self.dropout_rate_ls = [0.1, 0.2]
        elif args.dataset == 'imagenet':
            self.large_batch_size_ls = [510, 360, 170]
            self.small_batch_size_ls = []
            self.resolution_ls = [160, 224, 288]
            self.dropout_rate_ls = [0.1, 0.2, 0.3]
        else:
            raise ValueError(f'Invalid dataset "{args.dataset}".')
        self.large_batch_size_iter = itertools.cycle(self.large_batch_size_ls)
        self.small_batch_size_iter = itertools.cycle(self.small_batch_size_ls)
        self.resolution_iter = itertools.cycle(self.resolution_ls)
        self.dropout_rate_iter = itertools.cycle(self.dropout_rate_ls)
        self.milestones = list(self.mini_epochs // self.steps * i for i in range(1, self.steps))
        self.modify_freq = (self.milestones[0] if args.cycle else self.mini_epochs) // len(self.resolution_ls)
        self.parameter = {
            ## global commit ID
            'global_commit_ID': 0,
            ## low-level control options
            'device_index': 0,
            'learning_rate': 1e-1,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'gamma': 0.1,
            ## adaptive options
            'large_batch_size': self.large_batch_size_ls[0],
            'small_batch_size': self.small_batch_size_ls[0],
            'resolution': self.resolution_ls[0],
            'dropout_rate': self.dropout_rate_ls[0],
        }
        # global model
        self.model_lock = threading.Lock()
        self.global_model = tf_data_model.modify_resnet(
            dataset=args.dataset,
            depth=args.depth,
            dropout_rate=self.dropout_rate_ls[0],
            resolution=self.resolution_ls[0],
        )
        # record
        self.history_lock = threading.Lock()
        self.history = {
            # ID
            'worker_ID': [],
            'global_comit_ID': [],
            'local_commit_ID': [],
            'step_ID': [],
            'stage_ID': [],
            # train
            'train_loss': [],
            'train_acc': [],
            'train_time': [],   # count from model.fit start
            # val
            'val_loss': [],
            'val_acc': [],
            'commit_time': [],  # count from program start
        }
        self.outfile = (
            f'{args.dataset}_resnet{args.depth}_{self.epochs}'
            f'_W{args.word_size}S{args.small}'
            f'{"_amp" if args.amp else ""}'
            f'{"_xla" if args.xla else ""}'
            f'{"" if args.cycle else "_noCycle"}'
            f'{"_" + args.comments if args.comments else ""}'
        )
        self.tempfile = f'temp_{self.outfile}'
    
    def get_mission_complete(ps_rref):
        self = ps_rref.local_value()
        pass

    def get_parameter():
        pass

    def push_and_pull_model():
        pass

    def set_history():
        pass

# Worker
class Worker(object):
    def __init__(self, args, ps_rref, rank, is_small_batch):
        # others
        self.args = args
        self.ps_rref = ps_rref
        self.rank = rank
        self.is_small_batch = is_small_batch
        self.epoch_index = 0
        self.parameter = None#rpc.rpc_sync(self.ps_rref.owner(), Server.get_parameter)
        # data
        self.dataloader = None
        '''tf_data_model.load_data(
            resolution=self.parameter['resolution'],
            batch_size=self.parameter['small_batch_size'] if is_small_batch else self.parameter['large_batch_size'],
            dataset=args.dataset,
        )'''
        # local model
        self.model = None

    def train(self):
        while not rpc.rpc_sync(self.ps_rref.owner(), Server.get_mission_complete):
            self.parameter = rpc.rpc_sync(self.ps_rref.owner(), Server.get_parameter)
        pass