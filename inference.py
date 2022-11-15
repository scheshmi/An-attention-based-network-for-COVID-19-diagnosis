from tensorflow import keras
import tensorflow as tf
from dataset import testing_dataset
from model import *


# define  model and load pretrained weights

model = MANet(MutexBlock, FusionBlock)

model.load_weights('./saved_model/')


# define metrics

auc = keras.metrics.AUC()
acc = keras.metrics.CategoricalAccuracy()


def prediction(model: tf.keras.models.Model, test_ds: tf.data.Dataset) -> None:
    '''
    make prediction with model and testset

    param model: model which want to make prediction with
    param test_ds: test dataset

    '''
    # set zeros for mutex input

    mutex_input = tf.zeros(shape=(224, 224, 3), dtype=tf.float32)
    mutex_input = tf.expand_dims(mutex_input, axis=0)

    for input, label in test_ds:

        _, _, pred, _ = model((input, mutex_input), training=False)

        # updating metrics
        acc.update_state(label, pred)
        auc.update_state(label, pred)

    print(f'Accuracy : {acc.result()}')
    print(f'AUC : {auc.result()}')

    acc.reset_states()
    auc.reset_states()


# loading test dataset
test_ds = testing_dataset('./dataset/testing')

prediction(model, test_ds)
