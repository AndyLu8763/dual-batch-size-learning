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
1. Create a new virtual environment. Recommend using `conda` to control it.
  ```
  conda create -n dbsl
  ```
2. Activate the virtual environment.
  ```
  conda activate dbsl
  ```
3. Add commands to `.bashrc`
  ```
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > ~/.bashrc
  echo 'conda activate dbsl' > ~/.bashrc
  ```
4. Add commands to the virtual environment.
  ```
  mkdir -p $CONDA_PREFIX/etc/conda/activate.d
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
  ```
5. Install conda packages.
  ```
  conda install -n dbsl -c pytorch -c conda-forge -c nvidia cudatoolkit cudnn matplotlib notebook scikit-learn pytorch torchvision
  ```
  You could also appoint the package's version, e.g., `python=3.10`.
6. Install other packages by `pip` in the conda virtual environment.
  ```
  pip install tensorflow
  ```
  Since that TensorFlow official support doesn't offer installation by `conda`, using `conda` instead of `pip` might occur unexpected errors.

## Create Folders
`mkdir DBSL_npy DBSL_model`
scp -r dual-batch-size-learning/experimental/ r08944044@csl.iis.sinica.edu.tw:~
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python experimental/DBSL1080.py -a='140.109.23.144' -w=5 -r= &
python experimental/DBSL3090.py -a='140.109.23.230' -w=5 -r= &
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