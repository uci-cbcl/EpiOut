import numpy as np
import pandas as pd
import tensorflow as tf
from epiout.sflognorm import SFLogNorm


def _rlnorm(size, inj_mean, inj_sd):
    log_mean = np.log(inj_mean) if inj_mean != 0 else 0
    return np.random.lognormal(mean=log_mean, sigma=np.log(inj_sd), size=size)


def inject_outliers(df_counts, inj_freq=1e-3, inj_mean=3, inj_sd=1):
    '''
    Inject outliers into counts dataframe

    Args:
      df_counts: counts dataframe.
      inj_freq: frequency of outliers.
      inj_mean: mean of log normal distribution.
      inj_sd: standard deviation of log normal distribution.

    Returns:
      Tuple of outlier mask as np.array and injected counts as dataframe.
    '''

    counts = df_counts.astype('float32').values

    norm = SFLogNorm()

    X = norm.fit_transform(counts)

    outlier_mask = np.random.choice(
        [0., -1., 1.],
        size=X.shape,
        p=[1 - inj_freq, inj_freq / 2, inj_freq / 2])

    # insert with log normally distributed zscore in transformed space
    inj_zscores = _rlnorm(size=X.shape, inj_mean=inj_mean, inj_sd=inj_sd)
    sd = np.nanstd(X, ddof=1, axis=0)
    X_inj = X + outlier_mask * inj_zscores * sd

    counts_inj = np.round(norm.inverse_transform(X_inj)).astype('int')

    too_large = (counts_inj / counts.max(axis=0)) > 100
    counts_inj = np.where(too_large, counts, counts_inj)
    outlier_mask = np.where(too_large, 0, outlier_mask)

    df_inj = pd.DataFrame(counts_inj, index=df_counts.index,
                          columns=df_counts.columns)

    return np.abs(outlier_mask), df_inj
