import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from epiout.negative_binomial import NegativeBinomial, NBLoss


@pytest.fixture
def samples_1d_nb():
    mean = 3.
    var = 3.7
    r = mean**2 / (var - mean)
    p = (var - mean) / var
    samples = tfp.distributions.NegativeBinomial(r, p) \
                               .sample(10000, seed=31).numpy()
    return samples


def test_NegativeBinomial_infer_dispersion_1d(samples_1d_nb):
    mean = 3.
    var = 3.7
    dispersion_true = mean**2 / (var - mean)

    samples_1d_nb = tf.reshape(samples_1d_nb, (-1, 1))
    mean = tf.reshape(mean, (-1, 1))

    dispersion_mom = NegativeBinomial() \
        .infer_mean(samples_1d_nb) \
        .infer_dispersion(samples_1d_nb, method='mom') \
        .dispersion
    assert abs(dispersion_true - dispersion_mom) < 1

    dispersion_mle = NegativeBinomial() \
        .infer_mean(samples_1d_nb) \
        .infer_dispersion(samples_1d_nb, method='mle') \
        .dispersion

    dispersion_mom = NegativeBinomial(
        tf.ones_like(samples_1d_nb) * tf.reduce_mean(samples_1d_nb)
    ).infer_dispersion(samples_1d_nb, method='mom').dispersion

    assert abs(dispersion_true - dispersion_mom) < 1


def test_NegativeBinomial_infer_dispersion_2d():

    mean = 3 + tf.random.uniform(shape=(100,), seed=31) * 2
    var = mean + tf.random.uniform(shape=(100,), seed=31)
    dispersion_true = mean**2 / (var - mean)
    p = (var - mean) / var
    samples = tfp.distributions.NegativeBinomial(
        dispersion_true, p).sample(999, seed=31).numpy()

    nb_mom = NegativeBinomial() \
        .infer_mean(samples) \
        .infer_dispersion(samples, method='mom') \

    assert nb_mom.dispersion.shape == dispersion_true.shape

    nb_mle = NegativeBinomial() \
        .infer_mean(samples) \
        .infer_dispersion(samples, method='mle') \

    assert tf.reduce_all((
        tf.reduce_sum(nb_mle.log_prob(samples), axis=0)
        - tf.reduce_sum(nb_mom.log_prob(samples), axis=0)
    ) > -0.1)


def test_NegativeBinomial_pval():

    nb_tf = tfp.distributions.NegativeBinomial \
        .experimental_from_mean_dispersion(15, 1/12)
    samples = nb_tf.sample(10000, seed=31)

    cdf = nb_tf.cdf(samples)
    pdf = nb_tf.prob(samples)
    pval_tf = 2 * tf.minimum(cdf, (1 - cdf + pdf))
    pval_tf = tf.minimum(1, pval_tf)

    pval = NegativeBinomial() \
        .infer_mean(samples) \
        .infer_dispersion(samples) \
        .pval(samples)

    assert tf.reduce_mean(pval_tf - pval).numpy() \
        == pytest.approx(0, abs=1e-2)

    mean = 10 + tf.random.uniform(shape=(5000,), seed=31) * 100
    dispersion = 1 + tf.random.uniform(shape=(5000,), seed=31) * 90

    nb_tf = tfp.distributions.NegativeBinomial \
        .experimental_from_mean_dispersion(mean, 1/dispersion)
    samples = nb_tf.sample(99, seed=31).numpy()

    cdf = nb_tf.cdf(samples)
    pdf = nb_tf.prob(samples)
    pval_tf = 2 * tf.minimum(cdf, (1 - cdf + pdf))
    pval_tf = tf.minimum(1, pval_tf)

    pval = NegativeBinomial() \
        .infer_mean(samples) \
        .infer_dispersion(samples, method='mle') \
        .pval(samples)

    assert pval.shape == samples.shape
    assert tf.reduce_mean(pval - pval_tf).numpy() \
        == pytest.approx(0, abs=1e-2)


def test_NBLoss():

    mean = 10 + tf.random.uniform(shape=(5000,), seed=31) * 100
    dispersion = 1 + tf.random.uniform(shape=(5000,), seed=31) * 90

    nb_tf = tfp.distributions.NegativeBinomial \
        .experimental_from_mean_dispersion(mean, 1/dispersion)
    counts = nb_tf.sample(99, seed=31).numpy()

    nb_loss = NBLoss()
    loss = nb_loss(counts, counts)

    assert loss.numpy() > 0
