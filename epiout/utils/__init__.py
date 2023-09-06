from . import common
import tensorflow as tf
from .utils import peak_str
from .io import df_batch_writer


def logit(x, eps=1e-7):

    # TODO: Value error for x > 1  and x< 0
    x = tf.clip_by_value(x, eps, 1. - eps)

    return -tf.math.log(1. / x - 1.)


def trim_mean(x, proportiontocut=0.125, axis=0):

    nobs = x.shape[axis]

    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut

    if (lowercut > uppercut):
        raise ValueError("Proportion too big.")

    x = tf.sort(x, axis=axis)[lowercut:uppercut]
    return tf.reduce_mean(x, axis=axis)
