from tensorflow import keras
import tensorflow as tf
from resnet import *


class MutexBlock(keras.layers.Layer):
    '''
    mutex attention block 
    '''

    def __init__(self,) -> None:
        super(MutexBlock, self).__init__()

    def call(self, f_rei, f_rem):

        h, w, c = f_rei.shape[1], f_rei.shape[2], f_rei.shape[3]
        batch = f_rei.shape[0]
        dist = tf.square(tf.subtract(f_rei, f_rem))

        dist = tf.reshape(dist, shape=[batch, -1, c])

        attn_map = tf.nn.softmax(dist, axis=0)

        attn_map = tf.reshape(attn_map, shape=[batch, h, w, c])

        f_am = tf.math.multiply(attn_map, f_rem)

        return f_am


class FusionBlock(keras.layers.Layer):
    '''
    Fusion attention block 
    param fc_dim: number of neurons in 2 dense layers
    '''

    def __init__(self, fc_dim: int) -> None:

        super(FusionBlock, self).__init__()
        self.fc_dim = fc_dim
        self.dense_1 = keras.layers.Dense(self.fc_dim)
        self.bn_1 = keras.layers.BatchNormalization()
        self.dense_2 = keras.layers.Dense(self.fc_dim)
        self.bn_2 = keras.layers.BatchNormalization()

    def call(self, f_am, f_rem, training):

        f_mix = tf.math.add(f_am, f_rem)

        global_avg = keras.layers.GlobalAveragePooling2D()(f_mix)
        global_max = keras.layers.GlobalMaxPooling2D()(f_mix)

        mix_feature = tf.math.add(global_avg, global_max)

        x = self.dense_1(mix_feature)
        x = self.bn_1(x, training)
        x = tf.nn.relu(x)

        x = self.dense_2(x)
        x = self.bn_2(x, training)
        z = tf.nn.relu(x)

        self.m_dense = keras.layers.Dense(f_am.shape[-1])
        self.n_dense = keras.layers.Dense(f_rem.shape[-1])

        m = self.m_dense(z)
        n = self.n_dense(z)

        concat = tf.concat([m, n], axis=0)

        atten_map = tf.nn.softmax(concat, axis=0)

        f_am = tf.math.multiply(f_am, atten_map[0, ...])
        f_rem = tf.math.multiply(f_rem, atten_map[1, ...])

        f_fm = tf.math.add(f_am, f_rem)
        return f_fm


class MANet(keras.models.Model):
    '''
    Mutex attention network
    param mutex_attention: mutex attention block class
    param fusion_attention: fusion attention block class
    '''

    def __init__(self, mutex_attention: MutexBlock, fusion_attention: FusionBlock, input_size: tuple = (224, 224, 3)) -> None:

        super(MANet, self).__init__()
        self.input_size = input_size
        # define mutex and fusion block for each res-layer block

        self.mutex_attention_0 = mutex_attention()
        self.fusion_attention_0 = fusion_attention(128)

        self.mutex_attention_1 = mutex_attention()
        self.fusion_attention_1 = fusion_attention(128)

        self.mutex_attention_2 = mutex_attention()
        self.fusion_attention_2 = fusion_attention(128)

        self.mutex_attention_3 = mutex_attention()
        self.fusion_attention_3 = fusion_attention(128)

        self.mutex_attention_4 = mutex_attention()
        self.fusion_attention_4 = fusion_attention(128)

        # define Res-layer blocks ....
        self.layer_0 = keras.Sequential([
            keras.layers.Input(shape=input_size),
            keras.layers.Conv2D(64, (7, 7), strides=(
                2, 2), kernel_initializer=keras.initializers.glorot_uniform()),
            keras.layers.BatchNormalization(axis=3),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
        ])

        input_layer_1 = self.layer_0.output
        X = convolutional_block(
            input_layer_1, f=3, filters=[64, 64, 256], s=1),
        X = identity_block(X[0], 3, [64, 64, 256]),
        X = identity_block(X[0], 3, [64, 64, 256]),
        self.layer_1 = keras.models.Model(input_layer_1, X[0])

        input_layer_2 = self.layer_1.output
        X = convolutional_block(
            input_layer_2, f=3, filters=[128, 128, 512], s=2)
        X = identity_block(X, 3,  [128, 128, 512])
        X = identity_block(X, 3,  [128, 128, 512])
        X = identity_block(X, 3,  [128, 128, 512])
        self.layer_2 = keras.models.Model(input_layer_2, X)

        input_layer_3 = self.layer_2.output
        X = convolutional_block(input_layer_3, f=3, filters=[
                                256, 256, 1024], s=2)
        X = identity_block(X, 3, [256, 256, 1024])
        X = identity_block(X, 3, [256, 256, 1024])
        X = identity_block(X, 3, [256, 256, 1024])
        X = identity_block(X, 3, [256, 256, 1024])
        X = identity_block(X, 3, [256, 256, 1024])
        self.layer_3 = keras.models.Model(input_layer_3, X)

        input_layer_4 = self.layer_3.output
        X = convolutional_block(input_layer_4, f=3, filters=[
                                512, 512, 2048], s=2)
        X = identity_block(X, 3, [512, 512, 2048])
        X = identity_block(X, 3, [512, 512, 2048])
        self.layer_4 = keras.models.Model(input_layer_4, X)

        self.dense = keras.layers.Dense(128, 'relu')

        self.global_avg_pooling = keras.layers.GlobalAveragePooling2D()
        # classifier layers
        self.last_dense_1 = keras.layers.Dense(2, 'softmax')
        self.last_dense_2 = keras.layers.Dense(2, 'softmax')

    def call(self, inputs):
        # unpack input
        ct_input, mutex_input = inputs

        ####### layer 0 ##############
        f_rei = self.layer_0(ct_input)
        f_rem = self.layer_0(mutex_input)
        f_am = self.mutex_attention_0(f_rei, f_rem)
        f_om = self.fusion_attention_0(f_am, f_rem)

        ####### layer 1 #######

        f_rei = self.layer_1(f_rei)
        f_rem = self.layer_1(f_om)
        f_am = self.mutex_attention_1(f_rei, f_rem)
        f_om = self.fusion_attention_1(f_am, f_rem)

        ###### layer 2 ######

        f_rei = self.layer_2(f_rei)
        f_rem = self.layer_2(f_om)
        f_am = self.mutex_attention_2(f_rei, f_rem)
        f_om = self.fusion_attention_2(f_am, f_rem)

        ##### layer 3 #######

        f_rei = self.layer_3(f_rei)
        f_rem = self.layer_3(f_om)
        f_am = self.mutex_attention_3(f_rei, f_rem)
        f_om = self.fusion_attention_3(f_am, f_rem)

        #####  layer 4 #######

        f_rei = self.layer_4(f_rei)
        f_rem = self.layer_4(f_om)
        f_am = self.mutex_attention_4(f_rei, f_rem)
        f_om = self.fusion_attention_4(f_am, f_rem)

        # reshape frei and fom to vector for computing cos similarity loss
        reshaped_frei = tf.reshape(f_rei, [1, -1])
        reshaped_fom = tf.reshape(f_om, [1, -1])

        ##### global average pooling ####

        global_avg_ct = self.global_avg_pooling(f_rei)
        global_avg_mutex = self.global_avg_pooling(f_om)

        # dense layer

        ct_output = self.dense(global_avg_ct)
        mutex_output = self.dense(global_avg_mutex)

        # softmax layer
        ct_output = self.last_dense_1(ct_output)
        mutex_output = self.last_dense_2(mutex_output)

        return reshaped_frei, reshaped_fom, ct_output, mutex_output
