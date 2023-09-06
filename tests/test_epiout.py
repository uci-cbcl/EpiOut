import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from epiout.epiout import EpiOutResult


@pytest.fixture
def nb_tf():
    mean = 10 + tf.random.uniform(shape=(5000,), seed=31) * 100
    dispersion = 1 + tf.random.uniform(shape=(5000,), seed=31) * 90

    return tfp.distributions.NegativeBinomial \
        .experimental_from_mean_dispersion(mean, 1/dispersion)


def test_EpiOutResult(nb_tf):
    counts = pd.DataFrame(nb_tf.sample(99, seed=31).numpy())
    counts_mean = nb_tf.sample(99, seed=31).numpy()
    epiout_result = EpiOutResult(counts, counts_mean)

    assert epiout_result.pval.shape == counts.shape
    assert epiout_result.padj.shape == counts.shape
    assert epiout_result.l2fc.shape == counts.shape
    assert epiout_result.zscore.shape == counts.shape
    assert epiout_result.outlier.shape == counts.shape

    df_result = epiout_result.results()
    assert np.all(df_result['padj'] < 0.05)


def test_EpiOutResult_save_load(tmp_path, nb_tf):

    counts = pd.DataFrame(nb_tf.sample(99, seed=31).numpy())
    counts.index = counts.index.astype('str')
    counts.columns = counts.columns.astype('str')

    counts_mean = nb_tf.sample(99, seed=31).numpy()
    epiout_result = EpiOutResult(counts, counts_mean)

    filename = tmp_path / 'test.h5ad'
    epiout_result.save(filename)

    epiout_result_loaded = epiout_result.load(filename)

    pd.testing.assert_frame_equal(
        epiout_result_loaded.counts,
        epiout_result.counts
    )
    pd.testing.assert_frame_equal(
        epiout_result_loaded.counts_mean,
        epiout_result.counts_mean
    )
    pd.testing.assert_frame_equal(
        epiout_result_loaded.padj,
        epiout_result.padj
    )
    pd.testing.assert_frame_equal(
        epiout_result_loaded.l2fc,
        epiout_result.l2fc
    )
