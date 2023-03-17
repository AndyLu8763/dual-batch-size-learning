import tensorflow as tf
from tensorflow import keras


def load_cifar100(
    batch_size: int,
    validation_batch_size: int = batch_size,
    seed: int = None,
    num_parallel_calls: int = tf.data.AUTOTUNE):
    
    def map_func(image, label, TRAIN_FLAG):
        mean = [0.50705886, 0.48666665, 0.4407843 ]
        variance = [0.07153001, 0.06577717, 0.0762193 ]
        transform = keras.Sequential([
            keras.layers.Rescaling(1/255),
            keras.layers.Normalization(mean=mean, variance=variance)
        ])
        if TRAIN_FLAG:
            transform = keras.Sequential([
                transform,
                keras.layers.RandomTranslation(
                    height_factor=0.1,
                    width_factor=0.1,
                    fill_mode='constant'
                ),
                keras.layers.RandomFlip('horizontal')
            ])
        return transform(image), label
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    dataloader = {
        'train': (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                  .cache()
                  .shuffle(buffer_size=len(x_train), seed=seed)
                  .map(
                      lambda x, y: (map_func(x, y, TRAIN_FLAG=True)),
                      num_parallel_calls=num_parallel_calls)
                  .batch(batch_size=batch_size)
                  .prefetch(buffer_size=tf.data.AUTOTUNE)),
        'test': (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                 .cache()
                 .map(
                     lambda x, y: (map_func(x, y, TRAIN_FLAG=False)),
                     num_parallel_calls=num_parallel_calls)
                 .batch(batch_size=validation_batch_size)
                 .prefetch(buffer_size=tf.data.AUTOTUNE))
    }
    return dataloader


def make_resnet18(
    inputs: keras.Input = keras.Input(shape=(32, 32, 3)),
    num_classes: int = 100) -> keras.Model:
    
    def basicblock(inputs: keras.Input, filters: int, bottleneck: bool):
        if bottleneck:
            identity = keras.layers.Conv2D(
                filters, 1, strides=2, padding='valid', kernel_initializer='he_normal'
            )(inputs)
            identity = keras.layers.BatchNormalization()(identity)
            x = keras.layers.Conv2D(
                filters, 3, strides=2, padding='same', kernel_initializer='he_normal'
            )(inputs)
        else:
            identity = inputs
            x = keras.layers.Conv2D(
                filters, 3, strides=1, padding='same', kernel_initializer='he_normal'
            )(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(
            filters, 3, strides=1, padding='same', kernel_initializer='he_normal'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, identity])
        x = keras.layers.Activation('relu')(x)
        return x
    
    x = keras.layers.Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = basicblock(x, 64, False)
    x = basicblock(x, 64, False)
    x = basicblock(x, 128, True)
    x = basicblock(x, 128, False)
    x = basicblock(x, 256, True)
    x = basicblock(x, 256, False)
    x = basicblock(x, 512, True)
    x = basicblock(x, 512, False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)
