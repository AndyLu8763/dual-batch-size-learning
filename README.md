# Dual Batch Size Learning

## Environment
- cudatoolkit 11.3
- pytorch 1.10.1
- tensorflow 2.6.2
- torchvision 0.11.2

## Install Command
Please use conda and create a new environment by:
```
conda create -n $(env_name)
```
1. install one by one by one:
```
conda install cudatoolkit=11.3 -c nvidia
conda install pytorch=1.10.1 torchvision=0.11.2 -c pytorch
conda install matplotlib scikit-learn tensorflow=2.6.2 -c conda-forge
```
2. setting channel and install once
    1. vim ~/.condarc
    ```
    channels:
      - pytorch
      - nvidia
      - conda-forge
      - defaults
    ```
    2. install all packages
    ```
    conda install matplotlib scikit-learn cudatoolkit=11.3 pytorch=1.10.1 torchvision=0.11.2 tensorflow=2.6.2
    ```

## DBSL
Run `DBSL.py` by:
```
python DBSL.py -a='$(serverIP)' -w=$(wordSize) -r=$(rank)
```
- You should check ufw first
  - need the permission to access any `port` of the devices
  - `ufw allow from $(deviceIP)`
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
