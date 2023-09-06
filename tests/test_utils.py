import pytest
import numpy as np
import scipy
import tensorflow as tf
from epiout.utils import logit, trim_mean


def test_logit():
    x = tf.random.uniform(shape=(100, 100))

    assert x.numpy() == pytest.approx(
        tf.math.sigmoid(logit(x)).numpy()
    )

    logit_min = logit(0.)
    assert logit_min < -10
    assert not tf.math.is_inf(logit_min).numpy()

    logit_max = logit(1.)
    assert logit_max > 10
    assert not tf.math.is_inf(logit_max).numpy()


def test_trim_mean():

    mat = np.array([[0,  1,  2,  3],
                    [4,  5,  6,  7],
                    [8,  9, 10, 11],
                    [16, 17, 18, 19],
                    [12, 13, 14, 15]])
    mean_scipy = scipy.stats.trim_mean(
        mat, proportiontocut=0.25, axis=0)

    mean = trim_mean(mat, proportiontocut=0.25, axis=0)

    assert pytest.approx(mean_scipy) == mean.numpy()

    np.random.seed(31)
    mat = np.random.random(size=(5, 5))

    mean_scipy = scipy.stats.trim_mean(
        mat, proportiontocut=0.25, axis=0)

    mean = trim_mean(mat, proportiontocut=0.25, axis=0)
    assert pytest.approx(mean_scipy) == mean.numpy()
