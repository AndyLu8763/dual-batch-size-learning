# Efficient Dual Batch Size Deep Learning
<!--
K. -W. Lu, P. Liu, D. -Y. Hong and J. -J. Wu, "Efficient Dual Batch Size Deep Learning for Distributed Parameter Server Systems," 2022 IEEE 46th Annual Computers, Software, and Applications Conference (COMPSAC), 2022, pp. 630-639, doi: [10.1109/COMPSAC54236.2022.00110](https://doi.org/10.1109/COMPSAC54236.2022.00110).
-->

## Environment (Recommendation)
- python 3.9
- cudatoolkit 11.3
- pytorch 1.10
- torchvision 0.11
- tensorflow 2.6

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
  conda install -c pytorch -c nvidia -c conda-forge python matplotlib notebook scikit-learn pytorch torchvision tensorflow
  ```
  You could also appoint the package's version, e.g., `python=3.11`.

## Create Folders
`mkdir DBSL_npy DBSL_model`
If you do not do this, the results can not be saved.

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