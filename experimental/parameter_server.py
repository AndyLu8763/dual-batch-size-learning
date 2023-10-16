import argparse
import itertools
import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tf_data_model as tfdm


class Server(object):
    def __init__(self, args):
        self.args = args

    def count_epoch(self):
        pass

    def get_model_state(self):
        pass

    def update_model(self):
        pass

    def record_history(self):
        pass

    def save_history(self):
        pass


class Worker(object):
    def __init__(self, ps_rref, rank, args):
        self.ps_rref = ps_rref
        self.rank = rank
        self.args = args
        self.dataloader = tfdm.load_data(
            
        )

    def print_results(self):
        pass
    
    def train(self):
        pass
