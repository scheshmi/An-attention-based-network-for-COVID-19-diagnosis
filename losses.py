from tensorflow import keras
import tensorflow as tf


def classification_loss(y_ture: tf.Tensor, y_pred: tf.Tensor, y_true_m: tf.Tensor, y_pred_m: tf.Tensor) -> tf.Tensor:
    '''
    computing 2 deffirent losses, one for input and one for mutex input
    param y_ture: ground truth of input
    param y_pred : prediction for input
    param y_true_m: ground truth for mutex input
    param y_pred_m: prediction for mutex input

    :return summation of losses
    '''
    input_loss_fn = keras.losses.CategoricalCrossentropy()
    mutex_loss_fn = keras.losses.CategoricalCrossentropy()

    l1 = input_loss_fn(y_ture, y_pred)
    l2 = mutex_loss_fn(y_true_m, y_pred_m)

    return tf.add(l1, l2)


def adaptive_loss_weight(clf_loss: tf.Tensor, cos_loss: tf.Tensor) -> tuple:
    '''
    computing adaptive loss weight with corrensponding losses

    param clf_loss: classification loss
    param cos_loss: cos similarity loss

    :return weight of classification loss and cos similarity loss
    '''
    exp_clf = tf.math.exp(tf.math.divide(1, clf_loss))
    exp_cos = tf.math.exp(tf.math.divide(1, cos_loss))

    denominator = tf.math.add(exp_clf, exp_cos)

    a1 = tf.math.divide(exp_clf, denominator)
    a2 = tf.math.divide(exp_cos, denominator)

    return a1, a2


def adaptive_loss(y_ture: tf.Tensor, y_pred: tf.Tensor, y_true_m: tf.Tensor, y_pred_m: tf.Tensor, frei: tf.Tensor, fom: tf.Tensor) -> tf.Tensor:
    '''
    computing adaptive loss 

    param y_ture: ground truth of input
    param y_pred : prediction for input
    param y_true_m: ground truth for mutex input
    param y_pred_m: prediction for mutex input
    param frei: frei vector from 4th Res-Layer 
    param fom: fom vector from 4th Res-Layer

    :return adaptive loss
    '''
    clf_loss = classification_loss(y_ture, y_pred, y_true_m, y_pred_m)

    cos_loss_fn = keras.losses.CosineSimilarity()
    cos_loss = -1 * cos_loss_fn(fom, frei)

    a1, a2 = adaptive_loss_weight(clf_loss, cos_loss)

    # if a1 goes to infinity, set it to 1
    if tf.math.is_nan(a1):
        a1 = 1
        a2 = 0
    # if a2 goes to infinity, set it to 1
    if tf.math.is_nan(a2):
        a1 = 0
        a2 = 1

    return tf.multiply(a1, clf_loss) + tf.multiply(a2, cos_loss)
