import tensorflow as tf


class SFLogNorm:
    '''
    Size factor log normalization for samples based on the geometric mean.

    Args:
      mean_norm (bool): Whether to mean normalize the data.
    '''

    def __init__(self, mean_norm=True):
        self.mean_norm = mean_norm

    def fit(self, X):
        '''
        Fit the size factor log normalization.
        '''
        self.size_factor_ = self.size_factor(X)

        if self.mean_norm:
            self.mean_ = tf.reduce_mean(self.sflog(
                X, self.size_factor_), axis=0, keepdims=True)

        return self

    @staticmethod
    def size_factor(X):
        '''
        Size factor for each sample based on the geometric mean.

        Args:
          X (tf.Tensor): Input data as counts.
        '''
        log_gmean = tf.math.reduce_mean(tf.math.log(X), axis=0)

        s = tf.math.log(X) - log_gmean

        mask = (X < 1) | (~tf.math.is_finite(s))
        s = tf.where(mask, float('nan'), s)

        return tf.math.exp(tf.experimental.numpy.nanmean(
            s, axis=1, keepdims=True))

    @staticmethod
    def sflog(X, sf):
        '''
        Size factor log transformation.

        Args:
          X (tf.Tensor): Input data as counts.
          sf (tf.Tensor): Size factor for each sample.
        '''
        return tf.math.log((X + 1) / sf)

    def transform(self, X):
        '''
        Transform the data with the size factor log normalization.

        Args:
          X (tf.Tensor): Input data as counts.
        '''
        X = self.sflog(X, self.size_factor_)

        if self.mean_norm:
            return X - self.mean_

        return X

    def fit_transform(self, X):
        '''
        Fit size factor using count data and transform the data.

        Args:
          X (tf.Tensor): Input data as counts.
        '''
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        '''
        Inverse transform the data back to counts from the size factor 
          log normalization values.

        Args:
          X (tf.Tensor): Input data as size factor log normalization values.
        '''
        if self.mean_norm:
            X = X + self.mean_

        X = (tf.math.exp(X) * self.size_factor_) - 1

        return tf.math.maximum(X, 0)
