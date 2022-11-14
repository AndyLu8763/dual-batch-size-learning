# Efficient Dual Batch Size Deep Learning
<!--
K. -W. Lu, P. Liu, D. -Y. Hong and J. -J. Wu, "Efficient Dual Batch Size Deep Learning for Distributed Parameter Server Systems," 2022 IEEE 46th Annual Computers, Software, and Applications Conference (COMPSAC), 2022, pp. 630-639, doi: [10.1109/COMPSAC54236.2022.00110](https://doi.org/10.1109/COMPSAC54236.2022.00110).
-->

## Environment (Recommendation)
- python 3.10
- cuda 11.7
- cudnn 8.5
- pytorch 1.13
- torchvision 0.14
- tensorflow 2.10

## Dataset and Model
- The CIFAR-100 dataset
  - A. Krizhevsky, G. Hinton, et al. [Learning multiple layers of features from tiny images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf). Citeseer, 2009.
- The ResNet-18 model
  - K. He, X. Zhang, S. Ren, and J. Sun. [Deep residual learning for image recognition](https://doi.org/10.48550/arXiv.1512.03385). In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

## Installation
1. Upgrade CUDA at https://developer.nvidia.com/cuda-downloads.
2. Create a new virtual environment. Recommend using `conda` to control it.
  ```
  conda create -n dbsl
  ```
3. Activate the virtual environment.
  ```
  conda activate dbsl
  ```
4. Install conda packages.
  ```
  conda install -c pytorch -c conda-forge -c nvidia python=3.10 matplotlib notebook scikit-learn pytorch torchvision tensorflow
  ```
  You could also appoint the package's version, e.g., `python=3.10`.

## Create Folders
`mkdir DBSL_npy DBSL_model`

### 80, 120, 140 - 5 GPUs
- 0 small 1000
  Epoch 126 BS 1000 LR 0.004000000000000001
  train_loss: 0.3621405065059662, train_acc: 0.9092381000518799
  test_loss: 1.4774372577667236, test_acc: 0.6129000186920166
- 1 small
  Epoch 155 BS 1000 LR 0.004000000000000001
  train_loss: 0.7857744693756104, train_acc: 0.7660952210426331
  test_loss: 1.2635349035263062, test_acc: 0.6449999809265137
- 2 small 165
  Epoch 121 BS 1000 LR 0.004000000000000001
  train_loss: 0.4634036421775818, train_acc: 0.8686666488647461
  test_loss: 1.2191914319992065, test_acc: 0.669700026512146
- 3 small 237
  Epoch 119 BS 1000 LR 0.004000000000000001
  train_loss: 0.1065344363451004, train_acc: 0.9777143001556396
  test_loss: 1.260744571685791, test_acc: 0.6890000104904175
- 4small 297
  Epoch 122 BS 1000 LR 0.004000000000000001
  train_loss: 0.07687840610742569, train_acc: 0.9860000014305115
  test_loss: 1.3130261898040771, test_acc: 0.6870999932289124
- 5 small 349
  Epoch 126 BS 349 LR 0.004000000000000001
  train_loss: 0.046178147196769714, train_acc: 0.9934999942779541
  test_loss: 1.3479220867156982, test_acc: 0.6884999871253967


### 80, 120, 140 - 6 GPUs
- 0 small, BS = 1000
  Epoch 126 BS 1000 LR 0.004000000000000001
  train_loss: 0.5886362195014954, train_acc: 0.8390856981277466
  test_loss: 1.4859565496444702, test_acc: 0.605400025844574
- 1small, BS = 62
  Epoch 120 BS 1000 LR 0.004000000000000001
  train_loss: 1.0873005390167236, train_acc: 0.681942880153656
  test_loss: 1.425419807434082, test_acc: 0.6019999980926514
- 2 small, BS = 138
  Epoch 123 BS 138 LR 0.004000000000000001
  train_loss: 0.5868402719497681, train_acc: 0.8245333433151245
  test_loss: 1.1917192935943604, test_acc: 0.6675999760627747
- 3 small, BS = 203
  Epoch 124 BS 1000 LR 0.004000000000000001
  train_loss: 0.19602468609809875, train_acc: 0.9521142840385437
  test_loss: 1.2392641305923462, test_acc: 0.6819999814033508
- 4 small, BS = 258
  Epoch 125 BS 258 LR 0.004000000000000001
  train_loss: 0.15854904055595398, train_acc: 0.9652923345565796
  test_loss: 1.2823264598846436, test_acc: 0.6730999946594238
- 5 small, BS = 307
  Epoch 127 BS 307 LR 0.004000000000000001
  train_loss: 0.10307334363460541, train_acc: 0.9820606112480164
  test_loss: 1.3178762197494507, test_acc: 0.6808000206947327
- 6 small, BS = 349
  Epoch 127 BS 349 LR 0.004000000000000001
  train_loss: 0.07780713587999344, train_acc: 0.9883595108985901
  test_loss: 1.3448033332824707, test_acc: 0.6784999966621399

<!--
## Errr....
python DBSL_6worker.py -a='140.109.23.232' -w=7 -r= &

---

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> ~/.bashrc
echo 'conda activate dbsl' >> ~/.bashrc
scp -r dual-batch-size-learning/experimental/ r08944044@csl.iis.sinica.edu.tw:~
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python experimental/DBSL1080.py -a='140.109.23.144' -w=5 -r= &
python experimental/DBSL3090.py -a='140.109.23.230' -w=5 -r= &

---

2022-11-09 16:53:51.830616: W tensorflow/core/common_runtime/bfc_allocator.cc:290] Allocator (GPU_0_bfc) ran out of memory trying to allocate 108.00MiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.

---

Traceback (most recent call last):
  File "/local/r08944044/experimental/DBSL1080.py", line 311, in <module>
    run_program(args.rank, args.world_size, args.master_addr, args.master_port)
  File "/local/r08944044/experimental/DBSL1080.py", line 279, in run_program
    run_parameter_server(world_size)
  File "/local/r08944044/experimental/DBSL1080.py", line 254, in run_parameter_server
    torch.futures.wait_all(future_list)
  File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/torch/futures/__init__.py", line 313, in wait_all
    return [fut.wait() for fut in torch._C._collect_all(cast(List[torch._C.Future], futures)).wait()]
TypeError: TypeError: ResourceExhaustedError.__init__() missing 2 required positional arguments: 'op' and 'message'

At:
  /local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/torch/distributed/rpc/internal.py(220): _handle_exception

---

2022-11-09 16:53:58.305846: W tensorflow/core/common_runtime/bfc_allocator.cc:491] ****************************************************************************************************
2022-11-09 16:53:58.305871: W tensorflow/core/framework/op_kernel.cc:1780] OP_REQUIRES failed at conv_grad_input_ops.cc:327 : RESOURCE_EXHAUSTED: OOM when allocating tensor with shape[500,512,4,4] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
On WorkerInfo(id=3, name=worker_3):
ResourceExhaustedError()
Traceback (most recent call last):
  File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/torch/distributed/rpc/internal.py", line 206, in _run_function
    result = python_udf.func(*python_udf.args, **python_udf.kwargs)
  File "/local/r08944044/experimental/DBSL1080.py", line 244, in run_worker
    worker.train()
  File "/local/r08944044/experimental/DBSL1080.py", line 203, in train
    train_logs = self.model.fit(
  File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/keras/utils/traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/tensorflow/python/eager/execute.py", line 54, in quick_execute
    tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
tensorflow.python.framework.errors_impl.ResourceExhaustedError: Graph execution error:

Detected at node 'gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropInput' defined at (most recent call last):
    File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/torch/distributed/rpc/internal.py", line 206, in _run_function
      result = python_udf.func(*python_udf.args, **python_udf.kwargs)
    File "/local/r08944044/experimental/DBSL1080.py", line 244, in run_worker
      worker.train()
    File "/local/r08944044/experimental/DBSL1080.py", line 203, in train
      train_logs = self.model.fit(
    File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/keras/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/keras/engine/training.py", line 1564, in fit
      tmp_logs = self.train_function(iterator)
    File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/keras/engine/training.py", line 1160, in train_function
      return step_function(self, iterator)
    File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/keras/engine/training.py", line 1146, in step_function
      outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/keras/engine/training.py", line 1135, in run_step
      outputs = model.train_step(data)
    File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/keras/engine/training.py", line 997, in train_step
      self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/keras/optimizers/optimizer_v2/optimizer_v2.py", line 576, in minimize
      grads_and_vars = self._compute_gradients(
    File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/keras/optimizers/optimizer_v2/optimizer_v2.py", line 634, in _compute_gradients
      grads_and_vars = self._get_gradients(
    File "/local/r08944044/miniconda3/envs/dbsl/lib/python3.10/site-packages/keras/optimizers/optimizer_v2/optimizer_v2.py", line 510, in _get_gradients
      grads = tape.gradient(loss, var_list, grad_loss)
Node: 'gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropInput'
OOM when allocating tensor with shape[500,512,4,4] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
         [[{{node gradient_tape/model/conv2d_19/Conv2D/Conv2DBackpropInput}}]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info. This isn't available when running in Eager mode.
 [Op:__inference_train_function_398882]

[W tensorpipe_agent.cpp:726] RPC agent for worker_3 encountered error when reading incoming request from server_0: EOF: end of file (this error originated at tensorpipe/transport/uv/connection_impl.cc:132)

---
 
2022-11-10 15:36:51.109853: W tensorflow/core/common_runtime/bfc_allocator.cc:360] Garbage collection: deallocate free memory regions (i.e., allocations) so that we can re-allocate a larger region to avoid OOM due to memory fragmentation. If you see this message frequently, you are running near the threshold of the available device memory and re-allocation may incur great performance overhead. You may try smaller batch sizes to observe the performance impact. Set TF_ENABLE_GPU_GARBAGE_COLLECTION=false if you'd like to disable this feature.
-->

<!--
## DBSL
Run `DBSL.py` by:
```
python DBSL.py -a='$(serverIP)' -w=$(wordSize) -r=$(rank)
```
- You should check ufw first
  - need the permission to access any `port` of the devices
  - `ufw allow from $(deviceIP)`
  - maybe you also need to modify `/etc/hosts` and comment `127.0.0.1 localhost`
  - suck PyTorch RPC zzz...
- addres: Server IP
- world: numbers of machines on parameter server
- rank: 1~(w-1) if worker, 0 if server
- hyperparameters in code:
    - a, b: device information, get from linear regression
    - num_GPU, num_small
    - base_BS, base_LR
    - extra_time_ratio
    - rounds, threshold, gamma

## Plot Figure
Please use `Makefile` under the directory `plot`.
1. gnuplot: `make gnuplot`
2. pyplot: `make pyplot`
3. both: `make`
4. clean: `make clean`
-->