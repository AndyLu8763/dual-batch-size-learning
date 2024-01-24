# Experimental of Efficient Dual Batch Size Deep Learning for Distributed Parameter Server Systems
<!--
K. -W. Lu, P. Liu, D. -Y. Hong and J. -J. Wu, "Efficient Dual Batch Size Deep Learning for Distributed Parameter Server Systems," 2022 IEEE 46th Annual Computers, Software, and Applications Conference (COMPSAC), 2022, pp. 630-639, doi: [10.1109/COMPSAC54236.2022.00110](https://doi.org/10.1109/COMPSAC54236.2022.00110).
-->

## Dataset and Model
- The CIFAR dataset
    - A. Krizhevsky, G. Hinton, et al., [“Learning multiple layers of features from tiny images,”](https://www.cs.toronto.edu/~kriz/) Citeseer, 2009.
- The ResNet model
    - K. He, X. Zhang, S. Ren, and J. Sun, [“Deep residual learning for image recognition,”](https://doi.org/10.1109/CVPR.2016.90) in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

## Create Folders
`mkdir DBSL_npy DBSL_model`
If you do not do this, the results can not be saved.

## Environment
Build at 2024/01/07
- python 3.11
- tensorflow 2.13
- pytorch 2.1
- pytorch-cuda 12.1

## Installation
1. Create conda environment:
    ```
    conda create -n ${ENV}
    ```
2. Activate conda environment:
    ```
    conda activate ${ENV}
    ```
3. Install packages:
    - For just running programs:
        ```
        conda update --all -c pytorch -c nvidia python=3.11 tensorflow=2.13 pytorch pytorch-cuda=12.1
        ```
    - For running programs, analysing data, and others:
        ```
        conda update --all -c pytorch -c nvidia python=3.11 tensorflow=2.13 pytorch pytorch-cuda=12.1 matplotlib scikit-learn keras-cv torchvision ${JUPYTER}
        ```
        - ${JUPYTER} could be `jupyterlab` or `nbclassic`
    - Do not use `conda update --all` anymore, as package management issues can cause errors.
    - Others:
        - tensorflow 2.15 can not work, still trying now, `pip install -U tensorflow[and-cuda]==2.15.* torch`

## Jupyter Remote Setting
- packages: `jupyterlab`, `notebook`, `nbclassic`
1. python
    - from jupyter_server.auth import passwd
    - passwd()
        [argon2:xxx]
2. jupyter
    - for `jupyterlab`:
        - `jupyter lab`
        - the setting is same as `notebook`
        - the optional themes are `jupyterlab_legos_ui` and `jupyterlab_darkside_ui`
    - for `notebook`:
        - `jupyter notebook`
        - jupyter server --generate-config
        - vim ~/.jupyter/jupyter_server_config.py
            - c.ServerApp.ip = '*'
            - c.ServerApp.open_browser = False
            - c.ServerApp.password = u'argon2:xxxx'
            - c.ServerApp.port = 8763
    - for `nbclassic`:
        - `jupyter nbclassic`
        - jupyter notebook --generate-config
        - the optional theme is `jupyterthemes`
            - `jt -t oceans16`
        - vim ~/.jupyter/jupyter_notebook_config.py
            - c.ExtensionApp.open_browser = False
            - c.ServerApp.ip = '*'
            - c.ServerApp.password = u'argon2:xxx'
            - c.ServerApp.port = 8763

## Others
- conda create -n tf213 -c pytorch -c nvidia python=3.11 tensorflow=2.13 pytorch pytorch-cuda=12.1
- suck pip, thank you conda
- For tensorflow old version, try to use `imgaug` instead of `keras-cv` for doing image augmentation.
- Dataset
    - CIFAR
        - objects classes: 10 / 100
        - training images: 50000
        - validation images: 10000
    - ImageNet
        - objects classes: 1000
        - training images: 1281167
        - validation images: 50000
- Maximum Batch Size for GTX-1080
    - CIFAR
        - resolution_ls = [24, 32]
        - batch_size_ls = [600, 570], for `--no-amp --no-xla`
        - batch_size_ls = [2560, 1460], for `--no-amp --xla`
    - ImageNet
        - resolution_ls = [160, 224, 288]
        - batch_size_ls = [340, 160, 140], for `--amp --no-xla`
        - batch_size_ls = [1280, 620, 300], for `--amp --xla`
- Test Maximum Batch Size
    - `python record_batchSize_trainTime.py -r=${RES} -d=${DATA} -p=/ssd --start=${TEST_BS} --stop=5001 --step=10000 -t=10 ${--amp --xla --no-save}`
- Record Training Time
    - `--start=20 --step=20` with `--xla`, else `--start=10 --step=10`
    - `python record_batchSize_trainTime.py -r=${RESOLUTION} -d=${DATASET} -p=/ssd --start= --stop= --step= -t=10 ${--amp --xla}`
    - ex. `python record_batchSize_trainTime.py -r=32 -d=cifar100 -p=/ssd --start=10 --stop=571 --step=10 -t=10`
- Training
    - `python main.py -r= -w= -s= -a= -d= -p=/ssd -t=1.05 ${--amp --xla}`
    - ex.
        - for server, `python main.py -r=0 -w=2 -s=0 -a=192.168.0.1 -d=cifar100 -p=/ssd -t=1.05`
        - for worker, `python main.py -r=1 -w=2 -s=0 -a=192.168.0.1 -d=cifar100 -p=/ssd -t=1.05`

## ~~Too Old (Depricated)~~
### Environment
- python 3.9
- tensorflow 2.6.* (2.6.5)
- torch 1.10.* (1.10.2)
### Installation
1. Create conda environment:
    ```
    conda create -n DBSDL python=3.9
    ```
2. Activate conda environment:
    ```
    conda activate DBSDL
    ```
3. Install pip packages:
    - For just running programs:
        ```
        pip install -U tensorflow==2.6.* torch==1.10.*
        ```
    - For running programs, analyzing data, and others:
        - for `jupyterlab`
            ```
            pip install -U tensorflow==2.6.* torch==1.10.* jupyterlab matplotlib scikit-learn imgaug torchvision
            ```
        - for `nbclassic`
            ```
            pip install -U tensorflow==2.6.* torch==1.10.* nbclassic matplotlib scikit-learn imgaug torchvision
            ```
