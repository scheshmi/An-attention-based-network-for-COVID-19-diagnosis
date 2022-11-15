import numpy as np
from model import *
from losses import adaptive_loss
from dataset import training_dataset
from tensorflow import keras
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.set_logical_device_configuration(
    physical_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3600)])


# set random seed for python, numpy and tensorflow
keras.utils.set_random_seed(42)

EPOCHS = 3
# loading train dataset
train_ds = training_dataset('./dataset/training')

# set loss function
loss_fn = adaptive_loss

trainset_len = train_ds.cardinality()
# define learning rate scheduler
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=30 * trainset_len,
    decay_rate=0.5)

# set optimizer
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

# define metrics
auc = keras.metrics.AUC()
acc = keras.metrics.CategoricalAccuracy()

# build model
model = MANet(MutexBlock, FusionBlock)

# training model
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch + 1} ")
    for step, (ct, mutex) in enumerate(train_ds):

        # swap ct and mutex with probability of 50%
        if np.random.rand() > 0.5:
            ct, mutex = mutex, ct

        x_ct, y_ct = ct
        x_mutex, y_mutex = mutex

        with tf.GradientTape() as tape:
            reshaped_frei, reshaped_fom, ct_output, mutex_output = model(
                (x_ct, x_mutex), training=True)

            loss = loss_fn(y_ct, ct_output, y_mutex,
                           mutex_output, reshaped_frei, reshaped_fom)

        # compute gradients
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        total_true_y = tf.concat((y_ct, y_mutex), axis=0)
        total_pred_y = tf.concat((ct_output, mutex_output), axis=0)

        auc.update_state(total_true_y, total_pred_y)
        acc.update_state(total_true_y, total_pred_y)

        if step % 10 == 0:
            print(f'step: {step } loss : {loss}')

    print(f'Accuracy : {acc.result()}')
    print(f'AUC : {auc.result()}')

    auc.reset_states()
    acc.reset_states()

# saving model
model.save_weights('./saved_model/')
