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
Build at 2023/11/16
- python 3.11
- tensorflow 2.13.*
- torch
- (Optional) keras-cv
- (Optional) torchvision

## Installation
1. Create conda environment:
    ```
    conda create -n DBSDL python=3.11
    ```
2. Activate conda environment:
    ```
    conda activate DBSDL
    ```
3. Install pip packages:
    - For just running programs:
        ```
        pip install -U tensorflow==2.13.* torch
        ```
    - For running programs, analyzing data, and others:
        - for `jupyterlab`
            ```
            pip install -U tensorflow==2.13.* torch jupyterlab matplotlib scikit-learn keras-cv torchvision
            ```
        - for `nbclassic`
            ```
            pip install -U tensorflow==2.13.* torch nbclassic matplotlib scikit-learn keras-cv torchvision
            ```
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
- EZ quick start with `pip install -U tensorflow[and-cuda] torch torchvision`
- For tensorflow old version, try to use `imgaug` instead of `keras-cv` for doing image augmentation.

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
