import tensorflow as tf
from epiout.optimizer import backward_linesearch_gd


def test_backward_linesearch_gb():

    def loss_fn(x):
        return x**2 + 10

    x_init = tf.Variable(tf.random.uniform(shape=(100_000, 1),) * 1000.)

    x = backward_linesearch_gd(loss_fn, x_init)

    assert tf.reduce_all(loss_fn(x) < loss_fn(x_init))
    assert tf.reduce_all(x == 0)
