import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import tensorflow_probability as tfp
from sklearn.linear_model import LinearRegression
from epiout.negative_binomial import NegativeBinomial
from epiout.sflognorm import SFLogNorm


class LinearAutoEncoder:
    '''
    Linear autoencoder for count data with negative binomial likelihood.

    Args:
        bottleneck_size: bottleneck size of the autoencoder.
        njobs: number of jobs to run in parallel during training.
    '''

    def __init__(self, bottleneck_size: int, njobs: int = -1):
        self.bottleneck_size = bottleneck_size
        self.norm = SFLogNorm()
        self.njobs = njobs

    def init_weights(self, counts, X, metadata=None):
        '''
        Init layers and their weights with PCA
        '''
        self.encoder = tf.keras.layers.Dense(
            self.bottleneck_size, use_bias=False)
        self.encoder(X)
        pca = PCA(n_components=self.bottleneck_size).fit(X)
        self.encoder.set_weights([pca.components_.T])
        X_h = self.encoder(X)

        if metadata is None:
            weights = pca.components_
        else:
            X_h = np.hstack([X_h, metadata])
            weights = LinearRegression(fit_intercept=False, n_jobs=self.njobs) \
                .fit(X_h, X).coef_.T

        self.decoder = tf.keras.layers.Dense(X.shape[1], use_bias=False)
        self.decoder(X_h)
        self.decoder.set_weights([weights])

        mean = self.norm.inverse_transform(self.decoder(X_h))
        self.dispersion = NegativeBinomial(mean) \
            .infer_dispersion(counts, method='mom').dispersion

    def log_prob(self, counts, nmean, dispersion):
        '''
        Log probability of counts given mean and dispersion.
        '''
        return NegativeBinomial._log_prob(
            counts, tf.maximum(self.norm.inverse_transform(nmean), 1), dispersion)

    def update_dispersion(self, counts, metadata=None):
        '''
        Update dispersion parameter of the negative binomial distribution.
        '''
        self.dispersion = NegativeBinomial(mean=self(counts, metadata),
                                           dispersion=self.dispersion) \
            .infer_dispersion(counts, method='mle').dispersion

    def _loss_grad_encoder(self, counts, X, metadata=None):
        '''
        Loss and gradient functions of the encoder for l-bfgs algorithm.
        '''
        def loss_grad(weight):
            with tf.GradientTape() as tape:
                tape.watch(weight)
                shape = tf.shape(self.encoder.weights[0])

                X_hidden = X @ tf.reshape(weight, shape=shape)
                if metadata is not None:
                    X_hidden = tf.concat([X_hidden, metadata], axis=1)

                loss = -tf.reduce_mean(self.log_prob(
                    counts,
                    self.decoder(X_hidden),
                    self.dispersion
                ))
            grad = tape.gradient(loss, weight)
            return loss, grad

        return loss_grad

    @tf.function
    def _update_encoder(self, counts, X, metadata=None):
        init_position = tf.reshape(self.encoder.weights[0], [-1])
        loss_grad = self._loss_grad_encoder(counts, X, metadata)

        result = tfp.optimizer.lbfgs_minimize(
            loss_grad, initial_position=init_position,
            parallel_iterations=10, f_absolute_tolerance=.1)

        return tf.reshape(result.position,
                          shape=tf.shape(self.encoder.weights[0]))

    def update_encoder(self, counts, X, metadata=None):
        '''
        Update encoder weights with l-bfgs algorithm.
        '''
        self.encoder.set_weights([self._update_encoder(counts, X, metadata)])
        return self

    def _loss_grad_decoder(self, counts, X, metadata=None):
        '''
        Loss and gradient functions of the decoder for l-bfgs algorithm.
        '''
        X_hidden = self.encoder(X)

        if metadata is not None:
            X_hidden = tf.concat([X_hidden, metadata], axis=1)

        def loss_grad(weight):
            with tf.GradientTape() as tape:
                tape.watch(weight)
                loss = -tf.reduce_mean(self.log_prob(
                    counts,
                    X_hidden @ tf.transpose(weight),
                    self.dispersion
                ), axis=0)

            grad = tape.gradient(loss, weight)
            return loss, grad

        return loss_grad

    @tf.function
    def _update_decoder(self, counts, X, metadata=None):
        init_position = tf.transpose(self.decoder.weights[0])
        loss_grad = self._loss_grad_decoder(counts, X, metadata)

        result = tfp.optimizer.lbfgs_minimize(
            loss_grad, initial_position=init_position,
            parallel_iterations=10, f_absolute_tolerance=.1)

        return tf.transpose(result.position)

    def update_decoder(self, counts, X, metadata=None):
        '''
        Update decoder weights with l-bfgs algorithm.
        '''
        self.decoder.set_weights([self._update_decoder(counts, X, metadata)])
        return self

    def fit(self, counts, metadata=None, epochs=0,
            train_encoder=False, train_decoder=False):
        '''
        Fit the autoencoder weights by minimizing the negative log likelihood
          using alternating optimization updates of the encoder, decoder 
          and dispersion weights. 
        '''
        X = self.norm.fit_transform(counts)
        self.init_weights(counts, X, metadata)

        for i in range(epochs):

            if train_encoder or train_decoder:
                self.update_dispersion(counts, metadata)

                if train_decoder:
                    self.update_decoder(counts, X, metadata)
                if train_encoder:
                    self.update_encoder(counts, X, metadata)

        return self

    def __call__(self, counts, metadata=None):
        '''
        Predict reconstructed counts with the autoencoder 
          given counts and metadata.
        '''
        X_hidden = self.encoder(self.norm.transform(counts))

        if metadata is not None:
            X_hidden = tf.concat([X_hidden, metadata], axis=1)

        return self.norm.inverse_transform(self.decoder(X_hidden))
