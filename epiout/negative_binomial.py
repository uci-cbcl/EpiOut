from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
from epiout.utils import trim_mean
from epiout.optimizer import backward_linesearch_gd


class NegativeBinomial:
    '''
    Negative binomial distribution with mean and dispersion parameters.

    Args:
      mean: Mean parameter of negative binomial distribution.
      dispersion: Dispersion parameter of negative binomial distribution.
      dispersion_min: Minimum possible value of dispersion parameter.
      dispersion_max: Maximum possible value of dispersion parameter.
      mean_min: Minimum possible value of mean parameter.
    '''

    def __init__(self, mean=None, dispersion=None,
                 dispersion_min=0.1, dispersion_max=1000., mean_min=0.5):
        self.mean = mean
        self.dispersion = dispersion

        self.dispersion_min = dispersion_min
        self.dispersion_max = dispersion_max
        self.mean_min = mean_min

    def infer_dispersion(self, counts, method='mom', mle_kwargs=None):
        '''
        Infer dispersion parameter from counts.

        Args:
          counts: Counts of peaks.
          method: Method to infer dispersion parameter. 
            Valid values are `mom` and `mle`.
        '''
        if method == 'mom':
            self.dispersion = self._infer_dispersion_mom(counts)
        elif method == 'mle':
            mle_kwargs = mle_kwargs or dict()
            self.dispersion = self._infer_dispersion_mle(counts, **mle_kwargs)
        else:
            raise ValueError(f'`{method}` is not valid method name.'
                             ' Valid method values are `mom`, `mle`')
        return self

    def infer_mean(self, counts, method='mom'):
        '''
        Infer mean parameter from counts.
        '''
        if method == 'mom':
            self.mean = self._infer_mean_mom(counts)
        else:
            raise ValueError(f'`{method}` is not valid method name.'
                             ' Valid method values are `mom`')
        return self

    def _infer_mean_mom(self, counts):
        return tf.math.maximum(
            trim_mean(counts, proportiontocut=0.125, axis=0),
            self.mean_min)

    def _infer_dispersion_mom(self, counts):
        mean = self.mean

        if counts.shape == mean.shape:
            mean = tf.reduce_mean(mean, axis=0)

        ve = 1.51 * trim_mean(
            (counts - mean) ** 2,
            proportiontocut=0.125, axis=0)

        theta = mean ** 2 / (ve - mean)
        theta = tf.where(theta > 0, theta, self.dispersion_max)

        return tf.math.maximum(
            self.dispersion_min,
            tf.math.minimum(self.dispersion_max, theta))

    def _infer_dispersion_mle(self, counts, iteration=10,
                              decay_rate=0.5, c1=10**-4, c2=0.9):
        def loss_fn(log_dispersion, counts, mean):
            counts = tf.transpose(counts)
            mean = tf.transpose(mean)
            return -tf.reduce_sum(
                NegativeBinomial._log_prob(
                    counts, mean, tf.math.exp(log_dispersion)
                ), axis=0)

        # initial position
        if self.dispersion is None:
            dispersion = self._infer_dispersion_mom(counts)
        else:
            dispersion = self.dispersion

        log_dispersion = backward_linesearch_gd(
            loss_fn,
            initial_position=tf.math.log(dispersion),
            args=(tf.transpose(counts), tf.transpose(self.mean)),
            boundary=(
                tf.math.log(self.dispersion_min),
                tf.math.log(self.dispersion_max)
            ), max_iterations=iteration, decay_rate=decay_rate, c1=c1, c2=c2)

        return tf.clip_by_value(
            tf.math.exp(log_dispersion),
            self.dispersion_min, self.dispersion_max)

    def pval(self, counts):
        '''
        Calculate p-value from counts and inferred mean 
          and dispersion parameters.
        '''
        if self.mean is None:
            raise ValueError('Mean parameter is not inferred.'
                             'First infer mean with `infer_mean` method.')
        if self.dispersion is None:
            raise ValueError('Dispersion parameter is not inferred.'
                             'First infer dispersion with '
                             '`infer_dispersion` method.')

        nb = tfp.distributions.NegativeBinomial \
            .experimental_from_mean_dispersion(
                self.mean, 1 / self.dispersion)

        cdf = nb.cdf(counts)
        pdf = nb.prob(counts)

        pval = 2 * tf.minimum(cdf, (1 - cdf + pdf))
        pval = tf.minimum(1, pval)
        pval = tf.where(tf.math.is_finite(pval), pval, 1)

        return pval

    def log_prob(self, counts, eps=1e-8):
        '''
        Log probability of counts given mean and dispersion parameters.
        '''
        return self._log_prob(counts, self.mean, self.dispersion, eps=eps)

    @staticmethod
    def _log_prob(counts, mean, dispersion, eps=1e-8):
        log_theta_mu_eps = tf.math.log(dispersion + mean + eps)
        return (
            dispersion * (tf.math.log(dispersion + eps) - log_theta_mu_eps)
            + counts * (tf.math.log(mean + eps) - log_theta_mu_eps)
            + tf.math.lgamma(counts + dispersion)
            - tf.math.lgamma(dispersion)
            - tf.math.lgamma(counts + 1)
        )


class NBLoss(tf.keras.losses.Loss):
    '''
    Negative binomial loss function.

    Args:
      dispersion_min: Minimum possible value of dispersion parameter.
      dispersion_max: Maximum possible value of dispersion parameter.
    '''

    def __init__(self, dispersion_min=0.1, dispersion_max=1000.):
        super().__init__()
        self.dispersion_min = dispersion_min
        self.dispersion_max = dispersion_max

    def call(self, y_true, y_pred):
        nb = NegativeBinomial(mean=y_pred,
                              dispersion_min=self.dispersion_min,
                              dispersion_max=self.dispersion_max) \
            .infer_dispersion(y_true, method='mle')

        return -tf.reduce_sum(nb.log_prob(y_true))
