import threading
import time

# args
# world_size, 

# Server
class Server(object):
    def __init__(self, args):
        # setting
        self.mission_complete = False
        self.start_time = time.perf_counter()
        self.total_mini_epoch = None # total_epoch * world_size
        # model
        self.model_lock = threading.Lock()
        # epoch counter
        # record
        self.history_lock = threading.Lock()
        self.history = {
            # ID
            'worker_ID': [],
            'commit_ID': [],
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
    
    def check_mission_complete():
        pass

    def get_parameter():
        pass

    def push_and_pull_model():
        pass

    def send_history():
        pass

# Worker
class Worker(object):
    def __init__(self, ps_rref, rank):
        self.ps_rref = ps_rref
        self.rank = rank

    def train(self):
        while True:
            pass
