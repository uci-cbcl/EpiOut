import numpy as np
import pandas as pd
import scipy
import anndata
from statsmodels.stats.multitest import multipletests
from epiout.negative_binomial import NegativeBinomial
from epiout.plot import qq_plot, plot_counts, plot_rank, \
    plot_volcona, plot_umap, plot_corr_heatmap


class EpiOutResult:
    '''
    EpiOut result class for outlier detection to calculate statistics and 
      visualize results.

    Args:
      counts (pd.DataFrame): Counts data.
      counts_mean (np.array): Mean counts data.
      multipletests_method (str): Method for multiple testing correction.
      pval_threshold (float): P-value threshold for outlier detection.
      l2fc_threshold (float): Log2 fold change threshold for outlier detection.
      min_count_threshold (int): Minimum count threshold for outlier detection.

    Attributes:
        counts (pd.DataFrame): Counts data.
        counts_mean (np.array): Mean counts data.
        nb (NegativeBinomial): Negative binomial distribution.
        pval (pd.DataFrame): P-values.
        padj (pd.DataFrame): Adjusted p-values.
        l2fc (pd.DataFrame): Log2 fold change.
        zscore (pd.DataFrame): Z-score.
        outlier (pd.DataFrame): Outlier detection.
        log_padj (pd.DataFrame): -log10 adjusted p-values.

    Examples:
        >>> from epiout import EpiOutResult
        >>> result = EpiOutResult.load('result.h5ad')
        >>> result.outlier
        >>> result.log_padj
        >>> df_results = result.results()
        >>> result.qq_plot('chr1:100-200')
        >>> result.plot_counts('chr1:100-200')
        >>> result.plot_volcona('chr1:100-200')
    '''

    def __init__(self, counts: pd.DataFrame, counts_mean: np.array,
                 multipletests_method='fdr_by', pval_threshold=0.05,
                 l2fc_threshold=0.5, min_count_threshold=50):

        self._counts = counts
        self._counts_mean = self._df(counts_mean)
        self.multipletests_method = multipletests_method
        self.pval_threshold = pval_threshold
        self.l2fc_threshold = l2fc_threshold
        self.min_count_threshold = min_count_threshold

        self._nb = None
        self._pval = None
        self._padj = None
        self._l2fc = None
        self._zscore = None

    def _df(self, mat):
        return pd.DataFrame(mat,
                            columns=self.counts.columns,
                            index=self.counts.index)

    @property
    def counts(self):
        return self._counts

    @property
    def counts_mean(self):
        return self._counts_mean

    @property
    def nb(self):
        if self._nb is None:
            self._nb = NegativeBinomial(mean=self.counts_mean.values) \
                .infer_dispersion(self.counts.values, method='mle')
        return self._nb

    @property
    def pval(self):
        if self._pval is None:
            self._pval = self._df(self.nb.pval(self.counts.values).numpy())
        return self._pval

    @property
    def padj(self):
        if self._padj is None:
            fn = (lambda pvals: multipletests(
                pvals, method=self.multipletests_method)[1])
            self._padj = self._df(np.apply_along_axis(fn, 0, self.pval.values))
        return self._padj

    @property
    def l2fc(self):
        if self._l2fc is None:
            self._l2fc = self._df(np.log2(self.counts.values + 1)
                                  - np.log2(self.counts_mean.values + 1))
        return self._l2fc

    @property
    def zscore(self):
        if self._zscore is None:
            self._zscore = self._df(scipy.stats.zscore(self.l2fc.values))
        return self._zscore

    @property
    def outlier(self):
        return self._df(
            (self.padj.values < self.pval_threshold)
            & (np.abs(self.l2fc.values) > self.l2fc_threshold)
            & ((self.counts.values > self.min_count_threshold)
               | (self.counts_mean.values > self.min_count_threshold))
        )

    @property
    def log_padj(self):
        return self._df(-np.log10(self.padj.values))

    def results(self):
        '''
        Results of outlier detection for statistically significant outliers  
          as dataframe with columns: `peak`, 
          `sample`, `count`, `count_expected`, `pval`, `padj`, `l2fc`.
        '''
        outlier = self.outlier.values
        rows, cols = np.where(outlier)

        return pd.DataFrame({
            'peak': self.counts.columns[cols],
            'sample': self.counts.index[rows],
            'count': self.counts.values[outlier],
            'count_expected': self.counts_mean.values[outlier],
            'pval': self.pval.values[outlier],
            'padj': self.padj.values[outlier],
            'l2fc': self.l2fc.values[outlier]
        })

    def results_all(self):
        '''
        Results of utlier detection for all samples and peaks
          regardless of statistical significance
          as dataframe with columns: `peak`, 
          `sample`, `count`, `count_expected`, `pval`, `padj`, `l2fc`.
        '''
        return pd.DataFrame({
            'peak': np.repeat([self.counts.columns],
                              self.counts.shape[0], axis=0).ravel(),
            'sample': np.repeat([self.counts.index],
                                self.counts.shape[1], axis=1).ravel(),
            'count': self.counts.values.ravel(),
            'count_expected': self.counts_mean.values.ravel(),
            'pval': self.pval.values.ravel(),
            'padj': self.padj.values.ravel(),
            'l2fc': self.l2fc.values.ravel(),
            'outlier': self.outlier.values.ravel()
        })

    def qq_plot(self, peak, ci=0.95, eps=10e-15):
        '''
        QQ-plot for p-values of a peak.

        Args:
          peak (str): Peak name across samples.
        '''
        qq_plot(self.pval[peak], highlight=self.padj[peak] < self.pval_threshold,
                ci=ci, eps=eps)

    def plot_counts(self, peak):
        '''
        Counts plot for a peak across samples.

        Args:
          peak (str): Peak name.
        '''
        plot_counts(self.counts[peak], self.counts_mean[peak],
                    self.padj[peak] < self.pval_threshold)

    def plot_rank(self, peak, stats='l2fc'):
        '''
        Rank plot for a peak across samples.

        Args:
          peak (str): Peak name.
        '''
        if stats == 'l2fc':
            val = self.l2fc[peak]
        elif stats == 'zscore':
            val = self.zscore[peak]
        else:
            raise ValueError("Only `l2fc` and `zscore` supported stats")

        plot_rank(val, stats_name=stats,
                  highlight=self.padj[peak] < self.pval_threshold)

    def plot_volcona(self, peak, zscore_cutoff=(-1, 1)):
        '''
        Volcano plot for a peak across samples.

        Args:
          peak (str): Peak name.
        '''
        plot_volcona(self.padj[peak], self.zscore[peak],
                     pval_cutoff=self.pval_threshold,
                     zscore_cutoff=zscore_cutoff)

    def plot_umap(self, color=None, legend_title='color'):
        '''
        UMAP plot for samples.

        Args:
          color (str): Column name of color.
        '''
        plot_umap(self.counts, color=color, legend_title=legend_title)

    def plot_corr_heatmap(self, count_type='raw', row_var=None,
                          cbar_pos=(-.06, .45, .03, .2),
                          cmap='twilight_shifted', vmin=-1, vmax=1):
        '''
        Correlation heatmap between samples.

        Args:
          count_type: `raw`, `corrected`
          row_var (str): Row variable name.
          cbar_pos (tuple): Colorbar position.
          cmap (str): Colormap name.
          vmin (float): Minimum value for colormap.
          vmax (float): Maximum value for colormap.
        '''
        if count_type == 'raw':
            counts = self.counts
        elif count_type == 'corrected':
            counts = self.counts - self.counts_mean

        return plot_corr_heatmap(counts, row_var=row_var, cbar_pos=cbar_pos,
                                 cmap=cmap, vmin=vmin, vmax=vmax)

    def save(self, filename):
        '''
        Save result to h5ad file.

        Args:
          filename (str): File name.
        '''
        adata = anndata.AnnData(self.counts)
        adata.layers['counts'] = self.counts.values
        adata.layers['counts_mean'] = self.counts_mean.values
        adata.layers['pval'] = self.pval.values
        adata.layers['padj'] = self.padj.values
        adata.layers['l2fc'] = self.l2fc.values
        adata.write(filename)

    @classmethod
    def load(cls, filename):
        '''
        Save result to h5ad file.

        Args:
          filename (str): File name.
        '''
        adata = anndata.read_h5ad(filename)
        result = cls(adata.to_df('counts'), adata.layers['counts_mean'])
        result._pval = adata.to_df('pval')
        result._padj = adata.to_df('padj')
        result._l2fc = adata.to_df('l2fc')
        return result
