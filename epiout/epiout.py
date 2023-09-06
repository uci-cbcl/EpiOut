import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score
from epiout.inject import inject_outliers
from epiout.autoencoder import LinearAutoEncoder
from epiout.result import EpiOutResult


class EpiOut:

    def __init__(self, bottleneck_size=None, n_jobs=1,
                 train_encoder=False, train_decoder=True, epochs=1):
        self.bottleneck_size = bottleneck_size
        self.n_jobs = n_jobs
        self.train_encoder = train_encoder
        self.train_decoder = train_decoder
        self.epochs = epochs

    def _hyperparam(self, df_counts, steps):
        min_size = 2
        max_size = df_counts.shape[0] // 2

        return np.unique(np.round(np.exp(np.linspace(
            np.log(min_size), np.log(max_size), steps
        )))).astype('int')

    def tune_bottleneck(self, df_counts, confounders, steps=10):
        '''
        '''
        outlier_mask, df_counts_inj = inject_outliers(df_counts)

        df_auc = {'bottleneck_size': [], 'auc': []}

        hyperparams = self._hyperparam(df_counts, steps)

        for bottleneck_size in hyperparams:
            epiout = EpiOut(bottleneck_size)

            epiout_result = epiout(df_counts_inj, confounders)

            df_auc['bottleneck_size'].append(bottleneck_size)
            df_auc['auc'].append(average_precision_score(
                outlier_mask.ravel(),
                1 - epiout_result.pval.values.ravel()
            ))

        df_auc = pd.DataFrame(df_auc)

        self.bottleneck_size = df_auc.loc[
            df_auc['auc'].idxmax(), 'bottleneck_size']

        return df_auc

    def encode_metadata(self, metadata):
        return np.hstack([
            OneHotEncoder(sparse_output=False).fit_transform(
                metadata.select_dtypes(exclude=[np.float, np.int])),
            metadata.select_dtypes(include=[np.float, np.int]).values
        ])

    def __call__(self, df_counts, df_metadata=None):
        '''
        Perform training and prediction of EpiOut. Create results object 
          based on expected and predicted counts. 

        Args:
            df_counts: DataFrame of counts where rows are peaks 
              and columns are samples.
            df_metadata: DataFrame of metadata where 
        '''
        if self.bottleneck_size is None:
            raise ValueError(
                '`bottleneck_size` is not defined. '
                'Define `bottleneck_size` or run '
                '`self.tune_bottleneck(df_counts)` '
                'to tune `bottleneck_size`'
            )

        if df_metadata is not None:
            df_metadata = df_metadata.loc[df_counts.index]

        df_counts = df_counts.astype('float32')
        counts = tf.convert_to_tensor(df_counts.values)

        metadata = None
        if df_metadata is not None:
            metadata = tf.convert_to_tensor(
                self.encode_metadata(df_metadata).astype('float32'))

        lae = LinearAutoEncoder(self.bottleneck_size, self.n_jobs) \
            .fit(counts, metadata, self.epochs,
                 self.train_encoder, self.train_decoder)
        counts_pred = lae(counts, metadata)

        return EpiOutResult(df_counts, counts_pred)
