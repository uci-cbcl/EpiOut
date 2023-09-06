import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from epiout.inject import inject_outliers


def test_inject_outliers():

    mean = 10 + tf.random.uniform(shape=(5000,), seed=31) * 100
    dispersion = 1 + tf.random.uniform(shape=(5000,), seed=31) * 90

    nb_tf = tfp.distributions.NegativeBinomial \
        .experimental_from_mean_dispersion(mean, 1/dispersion)
    counts = nb_tf.sample(99, seed=31).numpy()
    counts = pd.DataFrame(counts)
    outlier_mask, counts_inj = inject_outliers(counts)

    assert outlier_mask.shape == counts.shape
    assert counts_inj.shape == counts.shape

    assert np.all((counts_inj != counts) == outlier_mask)
