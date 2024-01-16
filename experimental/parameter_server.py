import itertools
import os
import shutil
import threading
import time

import numpy as np
from tensorflow import keras
from torch.distributed import rpc

import tf_data_model

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
        self.mini_epochs = self.epochs * (args.world_size - 1)
        self.global_commit_ID = -1
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            self.total_data_amount = 50000
            self.intercept_ls = [0.00669587700416735, 0.010769692695705546] if args.xla else [0.01748355855302361, 0.018174712366753637]
            self.coef_ls = [0.000325070055313371, 0.00045254396224656614] if args.xla else [0.00036181042716526004, 0.0005055768285404]
            self.large_batch_size_ls = [2560, 1460] if args.xla else [600, 570]
            self.resolution_ls = [24, 32]
            self.dropout_rate_ls = [0.1, 0.2]
        elif args.dataset == 'imagenet':
            self.total_data_amount = 1281167
            if args.amp:
                self.intercept_ls = [0.005357090139827214, 0.015084276167305788, 0.013815540081899336] if args.xla else [0.01286918090689787, 0.01654844854797155, 0.013245930411050433]
                self.coef_ls = [0.001584763317361175, 0.002446198152606714, 0.003958850089352455] if args.xla else [0.0016273078905029127, 0.0025862208882727975, 0.004203924445708112]
                self.large_batch_size_ls = [1280, 620, 300] if args.xla else [340, 160, 140]
            else:
                raise ValueError('The ImageNet training process only supports "--amp"')
            self.resolution_ls = [160, 224, 288]
            self.dropout_rate_ls = [0.1, 0.2, 0.3]
        else:
            raise ValueError(f'Invalid dataset "{args.dataset}".')
        self.large_data_amount, self.small_data_amount, self.small_batch_size_ls = self.get_large_small_dataAmount_batchSize()
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
            'large_data_amount': self.large_data_amount,
            'small_data_amount': self.small_data_amount,
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
            'global_commit_ID': [], # count by server
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
    
    def get_large_small_dataAmount_batchSize(self):
        # numbers of GPUs
        num_total = self.args.world_size - 1
        num_small = self.args.small
        num_large = num_total - num_small
        # calculation
        large_data_amount = round(self.args.time_ratio * self.total_data_amount / num_total) if num_small else round(self.total_data_amount / num_total)
        small_data_amount = round((self.total_data_amount - large_data_amount * num_large) / num_small) if num_small else 0
        small_batch_size_ls = []
        for large_batch_size, intercept, coef in zip(self.large_batch_size_ls, self.intercept_ls, self.coef_ls):
            time_origin = (coef + intercept / large_batch_size) * self.total_data_amount / num_total
            time_new = self.args.time_ratio * time_origin
            small_batch_size_ls.append(round(intercept / (time_new / small_data_amount - coef)) if num_small else 0)
        return large_data_amount, small_data_amount, small_batch_size_ls

    def get_parameter(ps_rref):
        self = ps_rref.local_value()
        with self.parameter_lock:
            return self.parameter

    def get_global_model_weights(ps_rref):
        self = ps_rref.local_value()
        with self.global_model_lock:
            return self.global_model.get_weights()

    #### milestones is [10, 20, ..., 90]
    #### record commit_ID from '0' to '(89 * # of GPUs)'
    #### thus the last commit is '(90 * # of GPUs - 1)'
    def push_and_pull_model(ps_rref, worker_weights, rank, is_small_batch, parameter):
        self = ps_rref.local_value()
        with self.parameter_lock and self.global_model_lock:
            self.global_commit_ID += 1
            if self.global_commit_ID == self.mini_epochs:
                self.mission_complete = True
            if not self.mission_complete:
                #### update parameter and global model
                server_weights = self.global_model.get_weights()
                update_factor = parameter['small_data_amount'] / parameter['large_data_amount'] if is_small_batch else 1
                for i in range(len(server_weights)):
                    server_weights[i] = (
                        (2 - update_factor) * server_weights[i] + update_factor * worker_weights[i]
                    ) / 2
                self.model.set_weights(server_weights)
                ####
                print(f'Update Global Model by Worker {rank} at Global Mini-Epoch {self.global_commit_ID}')
            return self.global_model.get_weights(), self.global_commit_ID, self.mission_complete

    def update_history(ps_rref, record):
        self = ps_rref.local_value()
        with self.global_model_lock and self.history_lock:
            # update history
            record['commit_time'] = time.perf_counter() - self.start_time
            for key, value in record.items():
                self.history[key].append(value)
            print(f'worker {record["worker_ID"]} commit at time {record["commit_time"]}')
            # save tempfile
            if self.args.temp and record['global_commit_ID'] + 1 in self.milestones:
                self.save_tempfile()

    def save_tempfile(self):
        # lock is held by update_history()
        keras.models.save_model(self.global_model, f'{self.tempfile}_model')
        np.save(f'{self.tempfile}.npy', self.history)
        print(f'Save The Temporary Files at Global Mini-Epoch {self.global_commit_ID}')

    def save_outfile(self):
        # No lock is needed because training is complete
        if self.args.save:
            keras.models.save_model(self.global_model, f'{self.outfile}_model')
            np.save(f'{self.outfile}.npy', self.history)
            print(f'Save Model: {self.outfile}_model')
            print(f'Save Logs: {self.outfile}.npy')
        if self.args.temp:
            shutil.rmtree(f'{self.tempfile}_model', ignore_errors=True)
            if os.path.isfile(f'{self.tempfile}.npy'):
                os.remove(f'{self.tempfile}.npy')
            print('Clean The Temporary Files')

# Worker
class Worker(object):
    def __init__(self, ps_rref, args, rank, is_small_batch):
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
            # set model weights
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
                self.dataloader['train'].take(round(
                    self.parameter['small_data_amount'] / self.parameter['small_batch_size'] if self.is_small_batch
                    else self.parameter['large_data_amount'] / self.parameter['large_batch_size']
                )),
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
                    self.parameter,
                ),
            )
            self.model.set_weights(global_model_weights)
            # val
            val_logs = self.model.evaluate(
                self.dataloader['val'],
                verbose=self.verbose,
                return_dict=True,
            )
            # check mission_complete, if True, return
            if mission_complete:
                return
            # record
            record = {
                # ID
                'worker_ID': self.args.rank,
                'global_commit_ID': global_commit_ID,       # count by server
                'local_commit_ID': self.local_commit_ID,    # count by worker
                'step_ID': self.step_ID,
                'stage_ID': self.stage_ID,
                # train
                'train_loss': train_logs.history['loss'][0],
                'train_acc': train_logs.history['accuracy'][0],
                'train_time': train_logs.history['t'][0],   # count from model.fit start
                # val
                'val_loss': val_logs['loss'],
                'val_acc': val_logs['accuracy'],
                'commit_time': None,    # count from program start
            }
            rpc.rpc_sync(self.ps_rref.owner(), Server.update_history, args=(self.ps_rref, record))
            print(f'Worker {self.rank} Local Mini-Epoch {self.local_commit_ID} Complete')
            self.local_commit_ID += 1
