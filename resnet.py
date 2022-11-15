from tensorflow import keras
import tensorflow as tf


# define identity and conv block of resnet50

def identity_block(X: tf.Tensor, f: int, filters: list, training: bool = True, initializer: tf.keras.initializers = keras.initializers.random_uniform):
    '''
    Implementation of the identity block of ResNet50

    param X: input tensor
    param f: shape of the middle CONV's window for the main path
    param filters: number of filters in the CONV layers of the main path
    param training: specifying whether is training mode or not
    param initializer: set up the initial weights of a layer

    :return output of the identity block
    '''
    F1, F2, F3 = filters

    X_shortcut = X

    X = keras.layers.Conv2D(filters=F1, kernel_size=1, strides=(
        1, 1), padding='valid', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3)(X, training=training)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv2D(filters=F2, kernel_size=f, strides=(
        1, 1), padding='same', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3)(X, training=training)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv2D(filters=F3, kernel_size=1, strides=(
        1, 1), padding='valid', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3)(X, training=training)

    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X


def convolutional_block(X: tf.Tensor, f: int, filters: list, s: int = 2, training: bool = True, initializer: tf.keras.initializers = keras.initializers.glorot_uniform):
    '''
    Implementation of the convolutional block of ResNet50

    param X: input tensor
    param f: shape of the middle CONV's window for the main path
    param s: stride 
    param filters: number of filters in the CONV layers of the main path
    param training: specifying whether is training mode or not
    param initializer: set up the initial weights of a layer

    :return output of the convolutional block 
    '''
    F1, F2, F3 = filters

    X_shortcut = X

    X = keras.layers.Conv2D(filters=F1, kernel_size=1, strides=(
        s, s), padding='valid', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3)(X, training=training)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv2D(filters=F2, kernel_size=f, strides=(
        1, 1), padding='same', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3)(X, training=training)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv2D(filters=F3, kernel_size=1, strides=(
        1, 1), padding='valid', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3)(X, training=training)

    X_shortcut = keras.layers.Conv2D(filters=F3, kernel_size=1, strides=(
        s, s), padding='valid', kernel_initializer=initializer)(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(
        axis=3)(X_shortcut, training=training)

    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X
