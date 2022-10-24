# Experimental of the Dual Batch Size Learning

## Environment
- python 3.10
- cudatoolkit 11.6
- pytorch 1.12.1
- torchvision 0.13.1
- tensorflow 2.10.0

## Install Command
Please use conda and create a new environment by:
```
conda create -n $(env_name)
```
1. install once
```
conda install -c pytorch -c conda-forge jupyter matplotlib scikit-learn pytorch torchvision tensorflow cudatoolkit=11.6
```
2. setting channel than install
    1. vim `~/.condarc`
    ```
    channels:
      - pytorch
      - conda-forge
      - defaults
    ```
    2. install all packages
    ```
    conda install jupyter matplotlib scikit-learn pytorch torchvision tensorflow cudatoolkit=11.6
    ```

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