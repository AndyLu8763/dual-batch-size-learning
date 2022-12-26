import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_cifar(
    CIFAR_NAME: str = 'cifar100',
    batch_size: int = 500,
    seed: int = None,
    num_parallel_calls: int = tf.data.AUTOTUNE):
    def map_func(image, label, CIFAR_NAME, TRAIN_FLAG):
        if CIFAR_NAME == 'cifar10':
            mean = [0.49137256, 0.48235294, 0.44666666]
            variance = [0.06103806, 0.05930657, 0.06841814]
        else:
            mean = [0.50705886, 0.48666665, 0.4407843 ]
            variance = [0.07153001, 0.06577717, 0.0762193 ]
        transform = keras.Sequential([
            layers.Rescaling(1/255),
            layers.Normalization(mean=mean, variance=variance)])
        if TRAIN_FLAG:
            transform = keras.Sequential([
                transform,
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant'),
                layers.RandomFlip('horizontal')])
        return transform(image), label
    
    def dataloader(
        image, label, CIFAR_NAME, TRAIN_FLAG, batch_size, seed, num_parallel_calls):
        return (
            tf.data.Dataset.from_tensor_slices((image, label))
            .shuffle(buffer_size=len(label), seed=seed)
            .map(
                lambda x, y: (map_func(x, y, CIFAR_NAME, TRAIN_FLAG)),
                num_parallel_calls=num_parallel_calls)
            .batch(batch_size=batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE))
    
    if CIFAR_NAME == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    elif CIFAR_NAME == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    else:
        print('Wrong inputs, the argument "CIFAR_NAME" should be "cifar10" or "cifar100".')
        return None, None
    train_dataset = dataloader(
        x_train, y_train, CIFAR_NAME, True, batch_size, seed, num_parallel_calls)
    test_dataset = dataloader(
        x_test, y_test, CIFAR_NAME, False, batch_size, seed, num_parallel_calls)
    print(f'load_data: {CIFAR_NAME}')
    return train_dataset, test_dataset


def make_resnet18(
    CIFAR_NAME: str = 'cifar100',
    num_classes: int = 100,
    inputs: keras.Input = keras.Input(shape=(32, 32, 3))) -> keras.Model:
    def basicblock(inputs: keras.Input, filters: int, bottleneck: bool):
        if bottleneck:
            identity = layers.Conv2D(
                filters, 1, strides=2, padding='valid', kernel_initializer='he_normal'
            )(inputs)
            identity = layers.BatchNormalization()(identity)
            x = layers.Conv2D(
                filters, 3, strides=2, padding='same', kernel_initializer='he_normal'
            )(inputs)
        else:
            identity = inputs
            x = layers.Conv2D(
                filters, 3, strides=1, padding='same', kernel_initializer='he_normal'
            )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, strides=1, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, identity])
        x = layers.Activation('relu')(x)
        return x
    
    x = layers.Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = basicblock(x, 64, False)
    x = basicblock(x, 64, False)
    x = basicblock(x, 128, True)
    x = basicblock(x, 128, False)
    x = basicblock(x, 256, True)
    x = basicblock(x, 256, False)
    x = basicblock(x, 512, True)
    x = basicblock(x, 512, False)
    x = layers.GlobalAveragePooling2D()(x)
    if CIFAR_NAME == 'cifar10':
        num_classes = 10
    elif CIFAR_NAME == 'cifar100':
        num_classes = 100
    print(f'resnet18 num_classes = {num_classes}')
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)
