# Experimental Project of the Dual Batch Size Learning

## Environment (Recommendation)
- python 3.10
- cudatoolkit 11.6
- pytorch 1.12
- torchvision 0.13
- tensorflow 2.10

## Dataset and Model
- The CIFAR-100 dataset
  - A. Krizhevsky, G. Hinton, et al. [Learning multiple layers of features from tiny images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf). Citeseer, 2009.
- The ResNet-18 model
  - K. He, X. Zhang, S. Ren, and J. Sun. [Deep residual learning for image recognition](https://doi.org/10.48550/arXiv.1512.03385). In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

## Installation
1. First, create a new virtual environment. Recommend using `conda` to control it.
  ```
  conda create -n $(env_name)
  ```
2. Second, install the necessary packages.
  ```
  conda install -n $(env_name) -c pytorch -c conda-forge jupyter matplotlib scikit-learn pytorch torchvision tensorflow cudatoolkit
  ```
  You could also appoint the package's version, e.g., `cudatoolkit=11.6`.
3. Third, activate the virtual environment.
  ```
  conda activate $(env_name)
  ```

## Create Folders
`mkdir tf_npy tf_model`

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