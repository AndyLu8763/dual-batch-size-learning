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
4. others:
    - quick start:
        ```
        conda create -n ${ENV} -c pytorch -c nvidia python=3.11 tensorflow=2.13 pytorch pytorch-cuda=12.1
        ```

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
        - batch_size_ls = [600, 560]
        - batch_size_ls = [430, 580], for `--xla`
    - ImageNet
        - resolution_ls = [160, 224, 288]
        - batch_size_ls = [340, 170, 100], for `--amp`
        - batch_size_ls = [550, 160, 100], for `--amp --xla`
- Test Maximum Batch Size
    - `python record_batchSize_trainTime.py -r=${RES} -d=${DATA} -p=/ssd --start=${TEST_BS} --stop=5001 --step=10000 -t=10 ${--amp --xla --no-save}`
- Record Training Time
    - `--start=5 --step=5 --take=50`
    - `python record_batchSize_trainTime.py -r=${RESOLUTION} -d=${DATASET} -p=/ssd --start= --stop= --step= -t=50 ${--amp --xla}`
    - ex. `python record_batchSize_trainTime.py -r=32 -d=cifar100 -p=/ssd --start=5 --stop=561 --step=10 -t=50`
- Training
    - `python main.py -r= -w= -s= -a= -d= -p=/ssd -t=1.05 ${--amp --xla}`
    - ex.
        - for server, `python main.py -r=0 -w=2 -s=0 -a=$(IP_ADDRESS) -d=cifar100 -p=/ssd -t=1.05`
        - for worker, `python main.py -r=1 -w=2 -s=0 -a=$(IP_ADDRESS) -d=cifar100 -p=/ssd -t=1.05`
- Experiments
    - GTX-1080 for cifar100
        - `python exp1080/main.py -r= -w=5 -s= -a=140.109.23.106 -d=cifar100 -t=1.05`
        - file = `main.py` / `main_conf.py`
        - t = `1.05` / `1.1`
    - RTX-3090 for imagent with `--amp`
        - `python exp3090/main_3090.py -r= -w=5 -s= -a=140.109.23.231 -d=imagenet -p=/data -t=1.05 --amp`
        - file = `main_3090.py` / `main_conf_3090.py`
        - t = `1.05` / `1.1`

<!--
- GTX-1080 setting
    - gpu05 as server, gpu01/02/03/04 as workers
    - cifar100
        - `python main.py -r= -w=5 -s= -a=140.109.23.106 -d=cifar100 -t=1.05`
    - imagenet
        - `python main.py -r= -w=5 -s= -a=140.109.23.106 -d=imagenet -p=/ssd -t=1.05 --amp --xla`
- RTX-3090 imagenet setting
    - gpu10 as server, gpu06/07/08/14 as workers
    - ImageNet
        - test max BS
            - `vim ~/parameter_server_conf.py`
            - `cp ~/parameter_server_conf.py .`
            - `python main_conf.py -r= -w=2 -s=0 -a=140.109.23.231 -d=imagenet -p=/data -t=1.05 --amp`
            - resolution_ls = [160, 224, 288]
            - batch_size_ls = [2330, 1110, 740], for `--amp`
            - batch_size_ls = [2800, 1400, 900], for `--amp --xla` ~~[660, 990, 560] with no diff ERROR but mem not full~~
            - for `--amp --xla`
                - intercept_ls = [5.683134939754586e-05, 0.01073572758256125, 0.010680175780982126]
                - coef_ls = [0.00025895413494090355, 0.00045862692849402506, 0.0007576280737663753]
        - record BS train time
            - `--amp`
            ```
            python record_batchSize_trainTime.py -r=160 -d=imagenet -p=/data --start=10 --stop=2331 --step=10 -t=50 --amp ;\
            python record_batchSize_trainTime.py -r=224 -d=imagenet -p=/data --start=10 --stop=1111 --step=10 -t=50 --amp ;\
            python record_batchSize_trainTime.py -r=288 -d=imagenet -p=/data --start=10 --stop=741 --step=10 -t=50 --amp
            ```
            - `--amp --xla`
            ```
            python record_batchSize_trainTime.py -r=160 -d=imagenet -p=/data --start=10 --stop=2801 --step=10 -t=50 --amp --xla ;\
            python record_batchSize_trainTime.py -r=224 -d=imagenet -p=/data --start=10 --stop=1401 --step=10 -t=50 --amp --xla ;\
            python record_batchSize_trainTime.py -r=288 -d=imagenet -p=/data --start=10 --stop=901 --step=10 -t=50 --amp --xla
            ```
        - training
            - `python test3090/main_3090.py -r= -w=5 -s= -a=140.109.23.231 -d=imagenet -p=/data -t=1.05 --amp`
            - ~~`python test3090/main_3090.py -r= -w=5 -s= -a=140.109.23.231 -d=imagenet -p=/data -t=1.05 --amp --xla`~~
- Comments
    - `python record_batchSize_trainTime.py -r=160 -d=imagenet -p=/ssd --start=5 --stop=6 --step=5 -t=50 --amp --xla`
    - very slow...
    - 2024-01-27 00:27:58.708671: E tensorflow/compiler/xla/service/slow_operation_alarm.cc:133] The operation took 54m38.019217519s
-->
