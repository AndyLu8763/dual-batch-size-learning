import itertools
import math
import threading
import time

from tensorflow import keras
from torch.distributed import rpc

import tf_data_model

# args:
# [rank, world_size, small, addr, port,
#  dataset, dir_path, amp, xla, comments,
#  device_index, depth, cycle, temp, save]

# Server
class Server(object):
    def __init__(self, args):
        # global setting
        self.args = args
        self.start_time = time.perf_counter()
        self.mission_complete = False
        # training parameters
        self.parameter_lock = threading.Lock()
        self.epochs = 90
        self.steps = 3
        self.mini_epochs = self.epochs * args.world_size
        self.global_commit_ID = -1
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
        self.milestones = list(self.mini_epochs // self.steps * i for i in range(1, self.steps + 1))
        self.modify_freq = (self.milestones[0] if args.cycle else self.mini_epochs) // len(self.resolution_ls)
        if self.modify_freq == 0:
            raise ValueError('"modify_freq" is "0"')
        self.parameter = {
            # other options
            'global_step_ID': 0,
            'global_stage_ID': 0,
            # low-level control options
            'learning_rate': 1e-1,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'gamma': 0.1,
            # adaptive options
            'large_batch_size': next(self.large_batch_size_iter),
            'small_batch_size': next(self.small_batch_size_iter),
            'resolution': next(self.resolution_iter),
            'dropout_rate': next(self.dropout_rate_iter),
        }
        # global model
        self.global_model_lock = threading.Lock()
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
            'global_commit_ID': [],  # count by server
            'local_commit_ID': [],  # count by worker
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
            f'_W{args.world_size}S{args.small}'
            f'{"_amp" if args.amp else ""}'
            f'{"_xla" if args.xla else ""}'
            f'{"" if args.cycle else "_noCycle"}'
            f'{"_" + args.comments if args.comments else ""}'
        )
        self.tempfile = f'temp_{self.outfile}'

    #### check which function should I update global_commit
    #### all function should thinking again
    #### count 'self.global_commit_ID' to worker when push_and_pull_model
    #### 'self.global_commit_ID' mean different things

    def get_parameter(ps_rref):
        self = ps_rref.local_value()
        with self.parameter_lock:
            return self.parameter

    def get_global_model_weights(ps_rref):
        self = ps_rref.local_value()
        with self.global_model_lock:
            return self.global_model.get_weights()

    def push_and_pull_model(ps_rref, worker_weights, rank, is_small_batch, local_commit_ID):
        self = ps_rref.local_value()
        with self.parameter_lock and self.global_model_lock:
            self.global_commit_ID += 1
            if self.global_commit_ID == self.mini_epochs:
                self.mission_complete = True
            if not self.mission_complete:
                #### update parameter
                pass
            return self.global_model.get_weights(), self.global_commit_ID, self.mission_complete

    def update_history(ps_rref, record):
        self = ps_rref.local_value()
        with self.history_lock:
            record['commit_time'] = time.perf_counter() - self.start_time
            #### update record to history
            pass

    def save_tempfile():
        #### use "rpc_async" to call, by worker??
        #### how to decide save tempfile??
        pass

    def save_outfile():
        #### use "rpc_sync" to call, by main.py
        pass

# Worker
class Worker(object):
    def __init__(self, args, ps_rref, rank, is_small_batch):
        # settings
        self.args = args
        self.ps_rref = ps_rref
        self.rank = rank
        self.is_small_batch = is_small_batch
        self.local_commit_ID = 0
        self.step_ID = None
        self.stage_ID = None
        # parameters, data, and model
        self.parameter = None
        self.dataloader = None
        self.model = None
        self.verbose = 'auto'
        # callback
        class TimeCallback(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.history = []
            def on_epoch_begin(self, epoch, logs=None):
                self.time_epoch_begin = time.perf_counter()
            def on_epoch_end(self, epoch, logs=None):
                self.history.append(time.perf_counter() - self.time_epoch_begin)
        self.time_callback = TimeCallback()

    def train(self):
        # get mission_complete
        while True:
            # get parameter
            self.parameter = rpc.rpc_sync(self.ps_rref.owner(), Server.get_parameter, args=(self.ps_rref))
            # check if parameter is modified
            if self.step_ID != self.parameter['global_step_ID'] or self.stage_ID != self.parameter['global_stage_ID']:
                self.step_ID = self.parameter['global_step_ID']
                self.stage_ID = self.parameter['global_stage_ID']
                # get data
                self.dataloader = tf_data_model.load_data(
                    resolution=self.parameter['resolution'],
                    batch_size=self.parameter['small_batch_size'] if self.is_small_batch else self.parameter['large_batch_size'],
                    dataset=self.args.dataset,
                    dir_path=self.args.dir_path,
                )
                # get model
                self.model = tf_data_model.modify_resnet(
                    dataset=self.args.dataset,
                    depth=self.args.depth,
                    dropout_rate=self.parameter['dropout_rate'],
                    resolution=self.parameter['resolution'],
                    old_model=self.model if self.local_commit_ID != 0 else None,
                )
                # set model weights first time
                if self.local_commit_ID == 0:
                    self.model.set_weights(
                        rpc.rpc_sync(self.ps_rref.owner(), Server.get_global_model_weights, args=(self.ps_rref))
                    )
            # compile model
            self.model.compile(
                optimizer=keras.optimizers.experimental.SGD(
                    learning_rate=self.parameter['learning_rate'],
                    momentum=self.parameter['momentum'],
                    weight_decay=self.parameter['weight_decay'],
                ),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'],
            )
            # train
            train_logs = self.model.fit(
                self.dataloader['train'].take(math.ceil()),
                #### tf.data.Dataset.take(# of batch = allocated data / total data)
                verbose=self.verbose,
                callbacks=[self.time_callback],
            )
            train_logs.history['t'] = self.time_callback.history
            # push and pull model
            global_model_weights, global_commit_ID, mission_complete = rpc.rpc_sync(
                self.ps_rref.owner(),
                Server.push_and_pull_model,
                args=(
                    self.ps_rref,
                    self.model.get_weights(),
                    self.args.rank,
                    self.is_small_batch,
                    self.local_commit_ID,
                ),
            )
            self.model.set_weights(global_model_weights)
            # val
            val_logs = self.model.evaluate(
                self.dataloader['val'],
                verbose=self.verbose,
                return_dict=True,
            )
            # record
            record = {
                # ID
                'worker_ID': self.args.rank,
                'global_commit_ID': global_commit_ID,  # count by server
                'local_commit_ID': self.local_commit_ID,  # count by worker
                'step_ID': self.step_ID,
                'stage_ID': self.stage_ID,
                # train
                'train_loss': train_logs.history['loss'],
                'train_acc': train_logs.history['accuracy'],
                'train_time': train_logs.history['t'],   # count from model.fit start
                # val
                'val_loss': val_logs['loss'],
                'val_acc': val_logs['accuracy'],
                'commit_time': None,  # count from program start
            }
            rpc.rpc_sync(self.ps_rref.owner(), Server.update_history, args=(self.ps_rref, record))
            self.local_commit_ID += 1
            # check mission_complete
            if mission_complete:
                return
