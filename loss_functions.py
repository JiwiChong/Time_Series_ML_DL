import tensorflow as tf
from keras import backend as K


def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)


def custom_loss(y_true, y_pred):
    loss1 = tf.losses.mean_squared_error(y_true, y_pred)
    loss2 = tilted_loss(0.7, y_true, y_pred)
    return (loss1+loss2)/2