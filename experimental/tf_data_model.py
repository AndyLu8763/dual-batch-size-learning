from typing import Optional

import tensorflow as tf
from tensorflow import keras


# "True" for TF 2.6, "False" for TF 2.13
# Support weight_dacay via keras.model instead of keras.optimizers
TEST = True
weight_decay = 1e-4


def load_cifar(resolution: int, batch_size: int, dataset: str):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    var = [0.089401, 0.050176, 0.050625] # tf.math.square(std)
    
    resolution_list = [16, 24, 32]
    dataset_list = ['cifar10', 'cifar100']
    
    if resolution not in resolution_list:
        raise ValueError(f'Invalid resolution "{resolution}", it should be in {resolution_list}.')
    if dataset not in dataset_list:
        raise ValueError(f'Invalid resolution "{dataset}", it should be in {dataset_list}.')
    
    def preprocessing_map(image):
        transform = keras.Sequential([
            keras.layers.Resizing(resolution, resolution),
            keras.layers.Rescaling(1/255),
            keras.layers.Normalization(mean=mean, variance=var)
        ])
        return transform(image)
    
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    
    dataloader = {
        'train': (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .map(
                lambda x, y: (preprocessing_map(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            .cache()
            .shuffle(buffer_size=len(y_train))
            .batch(batch_size=batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        ),
        'val': (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .map(
                lambda x, y: (preprocessing_map(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(batch_size=batch_size)
            .cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
    }
    
    return dataloader


def load_imagenet(resolution: int, batch_size: int, dir_path: str):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    var = [0.089401, 0.050176, 0.050625] # tf.math.square(std)
    
    resolution_list = [160, 224, 288]
    
    if resolution not in resolution_list:
        raise ValueError(f'Invalid resolution "{resolution}", it should be in {resolution_list}.')
    
    '''
    # keras.utils.image_dataset_from_directory() can not allow simple augmentation pipeline
    # simple augmentation pipeline == keras.layers.RandomXXX()
    # move simple augmentation pipeline to build_model
    simple_aug = keras.Sequential([
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(factor=0.02),
        keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    ])
    '''
    
    def preprocessing_map(image):
        transform = keras.Sequential([
            keras.layers.Rescaling(1/255),
            keras.layers.Normalization(mean=mean, variance=var)
        ])
        return transform(image)
    
    dataloader = {
        'train': (
            keras.utils.image_dataset_from_directory(
                directory=f'{dir_path}/imagenet/train',
                label_mode='int', # for keras.losses.SparseCategoricalCrossentropy()
                batch_size=batch_size,
                image_size=(resolution, resolution),
                shuffle=True
            )
            .map(
                lambda x, y: (preprocessing_map(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            # tf.data.cache() is a bomb, causing excessive memory usage when training imagenet
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        ),
        'val': (
            keras.utils.image_dataset_from_directory(
                directory=f'{dir_path}/imagenet/val',
                label_mode='int', # for keras.losses.SparseCategoricalCrossentropy()
                batch_size=batch_size,
                image_size=(resolution, resolution),
                shuffle=False
            )
            .map(
                lambda x, y: (preprocessing_map(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            # tf.data.cache() is a bomb, causing excessive memory usage when training imagenet
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
    }
    
    return dataloader


def build_resnet(
    dataset: str,
    depth: int,
    dropout_rate: float,
    resolution: int,
    TEST_KERNEL_REGULARIZERS=keras.regularizers.L2(weight_decay) if TEST else None
) -> keras.Model:
    dataset_list = ['cifar10', 'cifar100', 'imagenet']
    depth_list = [18, 34]
    cifar_resolution_list = [16, 24, 32]
    imagenet_resolution_list = [160, 224, 288]
    
    if dataset not in dataset_list:
        raise ValueError(f'Invalid dataset "{dataset}", it should be in {dataset_list}.')
    if depth not in depth_list:
        raise ValueError(f'Invalid depth "{depth}", it should be in {depth_list}.')
    if 'cifar' in dataset:
        if resolution not in cifar_resolution_list:
            raise ValueError(f'Invalid resolution "{resolution}", it should be in {cifar_resolution_list}.')
    if 'imagenet' in dataset:
        if resolution not in imagenet_resolution_list:
            raise ValueError(f'Invalid resolution "{resolution}", it should be in {imagenet_resolution_list}.')
    
    if dataset == 'cifar10':
        classes = 10
    elif dataset == 'cifar100':
        classes = 100
    elif dataset == 'imagenet':
        classes = 1000
    
    if depth == 18:
        stack_list = [2, 2, 2, 2]
    elif depth == 34:
        stack_list = [3, 4, 6, 3]
    
    def basic_block(x: keras.Input, filters: int, conv_shortcut: bool = False):
        if conv_shortcut:
            shortcut = keras.layers.Conv2D(
                filters, 1, strides=2, use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=TEST_KERNEL_REGULARIZERS
            )(x)
            shortcut = keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(shortcut)
            x = keras.layers.Conv2D(
                filters, 3, strides=2, padding='same', use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=TEST_KERNEL_REGULARIZERS
            )(x)
        else:
            shortcut = x
            x = keras.layers.Conv2D(
                filters, 3, padding='same', use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=TEST_KERNEL_REGULARIZERS
            )(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(
            filters, 3, padding='same', use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=TEST_KERNEL_REGULARIZERS
        )(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(x)
        x = keras.layers.Add()([shortcut, x])
        x = keras.layers.Activation('relu')(x)
        return x
    
    def basic_stack(x: keras.Input, filters: int, stack: int, conv_shortcut: bool = False):
        for i in range(stack):
            if i == 0 and conv_shortcut == True:
                filters *= 2
                x = basic_block(x, filters, conv_shortcut)
            else:
                x = basic_block(x, filters)
        return x, filters
    
    inputs = keras.Input(shape=(resolution, resolution, 3))
    filters = 64
    ## simple augmentation pipeline
    simple_aug = keras.Sequential([
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(factor=0.02),
        keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    ])
    x = simple_aug(inputs)
    ## stem
    if 'cifar' in dataset:
        x = keras.layers.Conv2D(
            filters, 3, padding='same', use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=TEST_KERNEL_REGULARIZERS
        )(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(x)
        x = keras.layers.Activation('relu')(x)
    elif dataset == 'imagenet':
        x = keras.layers.Conv2D(
            filters, 7, strides=2, padding='same', use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=TEST_KERNEL_REGULARIZERS
        )(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    ## trunk
    for i, stack in enumerate(stack_list):
        if i == 0:
            x, filters = basic_stack(x, filters, stack)
        else:
            x, filters = basic_stack(x, filters, stack, True)
    ## classifier
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(
        classes, activation='softmax',
        kernel_regularizer=TEST_KERNEL_REGULARIZERS
    )(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)


def load_data(
    resolution: int,
    batch_size: int,
    dataset: str,
    dir_path: Optional[str] = None
):
    dataset_list = ['cifar10', 'cifar100', 'imagenet']
    
    if 'cifar' in dataset:
        return load_cifar(resolution=resolution, batch_size=batch_size, dataset=dataset)
    elif dataset == 'imagenet':
        if dir_path == None:
            raise ValueError(f'Invalid directory path "{dir_path}".')
        return load_imagenet(resolution=resolution, batch_size=batch_size, dir_path=dir_path)
    else:
        raise ValueError(f'Invalid dataset "{dataset}", it should be in {dataset_list}.')


def modify_resnet(
    dataset: str,
    depth: int,
    dropout_rate: float,
    resolution: int,
    old_model: Optional[keras.Model] = None
) -> keras.Model:
    keras.backend.clear_session()
    new_model = build_resnet(
        dataset=dataset,
        depth=depth,
        dropout_rate=dropout_rate,
        resolution=resolution,
        TEST_KERNEL_REGULARIZERS=keras.regularizers.L2(weight_decay) if TEST else None
    )
    if old_model:
        new_model.set_weights(old_model.get_weights())
    
    return new_model
