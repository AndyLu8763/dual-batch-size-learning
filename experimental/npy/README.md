# logs
- load files: `np.load('PATH/TO/THE/NPY/FILE', allow_pickle=True).item()`
- type: dict
- dict\_keys: ['batch\_size', 'resolution', 'dropout\_rate', 't', 'loss', 'accuracy', 'val\_loss', 'val\_accuracy', 'lr']
## cifar100
- training setting:
    - batch\_size: [1000, 500]
    - resolution: [24, 32]
    - dropout\_rate: [0.1, 0.2]
    - mixed\_precision True
    - jit\_compile True
-file name: {dataset}\_{model}\_{cycle}\_{epochs}\_{same?}.npy
    - ex. cifar100\_resnet18\_iter\_90\_same.npy
## imagenet
- training setting
    - batch\_size: [510, 360, 170]
    - resolution: [160, 224, 288]
    - dropout\_rate: [0.1, 0.2, 0.3]
- folder: bs{nx}\_{mp?}\_{xla?}
    - ex. bs4x\_mp
- file name: {dataset}\_{model}\_{cycle}\_{epochs}\_{shm?}.npy
    - ex. imagenet\_resnet18\_cycle\_90\_shm.npy
