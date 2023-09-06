import pytest
import numpy as np
import tensorflow as tf
from epiout.autoencoder import LinearAutoEncoder
from conftest import counts


@pytest.fixture
def lae():
    return LinearAutoEncoder(2, njobs=1)


def test_linear_autoencoder_init_weights(lae, counts):
    counts = counts.values
    X = lae.norm.fit_transform(counts)
    lae.init_weights(counts, X)

    assert lae.encoder.weights[0].shape == (X.shape[1], 2)
    assert lae.decoder.weights[0].shape == (2, X.shape[1])
    assert lae.dispersion.shape == X.shape[1]
    assert lae(counts).shape == X.shape


def test_linear_autoencoder_init_weights_metadata(lae, counts):
    counts = counts.values
    X = lae.norm.fit_transform(counts)
    metadata = np.random.normal(size=(X.shape[0], 3))
    lae.init_weights(counts, X, metadata)

    assert lae.encoder.weights[0].shape == (X.shape[1], 2)
    assert lae.decoder.weights[0].shape == (5, X.shape[1])
    assert lae.dispersion.shape == X.shape[1]
    assert lae(counts, metadata).shape == X.shape


def test_linear_autoencoder_update_dispersion(lae, counts):
    counts = counts.values
    X = lae.norm.fit_transform(counts)
    lae.init_weights(counts, X)
    lae.update_dispersion(counts)

    assert lae.dispersion.shape == X.shape[1]
    assert lae(counts).shape == X.shape


def test_linear_autoencoder_update_decoder(lae, counts):
    counts = counts.values

    X = lae.norm.fit_transform(counts)
    lae.init_weights(counts, X)

    decoder_weights = lae.decoder.weights[0].numpy()
    lae.update_decoder(counts, X)

    assert lae.decoder.weights[0].shape == (2, X.shape[1])
    assert lae(counts).shape == X.shape
    assert not np.allclose(decoder_weights, lae.decoder.weights[0].numpy())


def test_linear_autoencoder_update_encoder(lae, counts):
    counts = counts.values

    X = lae.norm.fit_transform(counts)
    lae.init_weights(counts, X)

    encoder_weights = lae.encoder.weights[0].numpy()
    lae.update_encoder(counts, X)

    assert lae.encoder.weights[0].shape == (X.shape[1], 2)
    assert lae(counts).shape == X.shape
    assert not np.allclose(encoder_weights, lae.encoder.weights[0].numpy())


def test_linear_autoencoder_fit(lae, counts):
    counts = counts.values
    metadata = tf.ones((counts.shape[0], 3))
    lae.fit(counts, metadata, epochs=5, train_encoder=True, train_decoder=True)

    assert lae.encoder.weights[0].shape == (counts.shape[1], 2)
    assert lae.decoder.weights[0].shape == (5, counts.shape[1])
    assert lae.dispersion.shape == counts.shape[1]
    assert lae(counts, metadata).shape == counts.shape
