#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tf_data_model


# In[ ]:


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
    pass

parser = argparse.ArgumentParser(
    description='Record Training Time by Using Tensorflow',
    epilog=(
        'Required settings [--resolution, --dataset, --path] or [-r, -d, -p], '
        'optional for loop arguments [--start, --stop, --step], '
        'optional settings [--amp, --xla, --depth, --take, --comments]'
    ),
    formatter_class=CustomFormatter,
)

# parser arguments
## GPU
parser.add_argument(
    '--device-index',
    type=int,
    default=0,
    help='the index of the GPU used to run the program, "0" or "-1" is a good choice',
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
## dataset and model
parser.add_argument(
    '--resolution', '-r',
    type=int,
    help='image resolution, currently support [24, 32] for "cifar" && [160, 224, 288] for "imagenet"',
)
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
## for loop
parser.add_argument(
    '--start',
    type=int,
    default=100,
    help='"start" value for range() in the for loop',
)
parser.add_argument(
    '--stop',
    type=int,
    default=101,
    help='"stop" value for range() in the for loop',
)
parser.add_argument(
    '--step',
    type=int,
    default=10,
    help='"step" value for range() in the for loop',
)
## others
parser.add_argument(
    '--take', '-t',
    type=int,
    help='creates a "Dataset" with at most "count" elements from this dataset, could be [int, None]',
)
parser.add_argument(
    '--comments', '-c',
    type=str,
    help='add additional comments on filename',
)
parser.add_argument(
    '--no-temp',
    dest='temp',
    action='store_false',
    help='do not save the temporary record during running program',
)
parser.add_argument(
    '--no-save',
    dest='save',
    action='store_false',
    help='do not save the record',
)

# check the file type is '.py' or '.ipynb'
## parse args of '.ipynb' from here
## ex. ['--dataset=imagenet', '--path=./dataset', '--amp', '--xla']
ipynb_args = [
    '-r=32', '-d=cifar100', '-p=/ssd',
    '--start=100', '--stop=501', '--step=100',
    '-t=10', '--amp', '--xla', '--no-temp', '--no-save',
]
args = (
    parser.parse_args(ipynb_args)
    if len(sys.argv) > 2 and sys.argv[1] == '-f' and '.json' in sys.argv[2]
    else parser.parse_args()
)
outfile = (
    f'time_{args.dataset}_resnet{args.depth}_r{args.resolution}'
    f'_{args.start}_{args.stop}_{args.step}'
    f'{"_amp" if args.amp else ""}'
    f'{"_xla" if args.xla else ""}'
)
tempfile = f'temp_{outfile}'
print('----')
print(outfile)
print(args)
print('----')


# In[ ]:


# mixed_precision and jit_compile
### for unknown reasons, '--tf_xla_cpu_global_jit' only supports the first GPU
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


# In[ ]:


# GPU initialization, data perallel not complete yet
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[args.device_index], 'GPU')
for device in tf.config.get_visible_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
print('----')
print(f'The Number of Available Physical Devices: {len(physical_devices)}')
print(f'Using Devices: {tf.config.get_visible_devices("GPU")}')
print('----')


# In[ ]:


class TimeCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.history = []
    def on_epoch_begin(self, epoch, logs=None):
        self.time_epoch_begin = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        self.history.append(time.perf_counter() - self.time_epoch_begin)

time_callback = TimeCallback()


# In[ ]:


logs = {
    'batch_size': [],
    'total_train_time': [],
    'avg_train_time': [],
    'take': args.take,
    'data': args.dataset,
    'depth': args.depth,
    'resolution': args.resolution,
    'amp': args.amp,
    'xla': args.xla,
    'sss': (args.start, args.stop, args.step),
    'comments': args.comments,
}


# In[ ]:


for batch_size in range(args.start, args.stop, args.step):
    print(f'Batch Size = {batch_size}')
    # data && model
    dataloader = tf_data_model.load_data(
        resolution=args.resolution,
        batch_size=batch_size,
        dataset=args.dataset,
        dir_path=args.dir_path,
    )
    model = tf_data_model.modify_resnet(
        dataset=args.dataset,
        depth=args.depth,
        dropout_rate=0,
        resolution=args.resolution,
        old_model=None,
    )
    # compile
    model.compile(
        optimizer=keras.optimizers.experimental.SGD(
            learning_rate=1e-1,
            momentum=0.9,
            weight_decay=None if tf_data_model.OLD_VERSION else 1e-4,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    # warmup
    print('Warmup:', end=' ')
    model.fit(
        dataloader['train'].take(1),
        verbose=2,
    )
    # train
    print('Train:', end=' ')
    temp_logs = model.fit(
        dataloader['train'].take(args.take) if args.take else dataloader['train'],
        verbose=2,
        callbacks=[time_callback],
    )
    # record
    logs['batch_size'].append(batch_size)
    logs['total_train_time'].append(time_callback.history[0])
    if args.take: # for data with limited batches
        logs['avg_train_time'].append(time_callback.history[0] / args.take)
    elif args.dataset == 'imagenet': # for total 'imagenet' dataset
        logs['avg_train_time'].append(time_callback.history[0] / np.ceil(1281167 / batch_size))
    else: # for total 'cifar' dataset
        logs['avg_train_time'].append(time_callback.history[0] / np.ceil(50000 / batch_size))
    # temp file
    if args.temp:
        np.save(f'{tempfile}.npy', logs)


# In[ ]:


# save file
if args.save:
    np.save(f'{outfile}.npy', logs)
    print(f'Save Logs: {outfile}.npy')

# remove temporary file after training
if args.temp:
    if os.path.isfile(f'{tempfile}.npy'):
        os.remove(f'{tempfile}.npy')
    print('Clean The Temporary File')

# load file
if False:
    f = np.load(f'{outfile}.npy', allow_pickle=True).item()


# In[ ]:




