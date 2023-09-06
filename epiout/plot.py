import pyBigWig
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import beta
import seaborn as sns
import matplotlib.pyplot as plt
from epiout.dataclasses import Peak


def coverage_bigwig(bw, peak: Peak, rolling_percent=.1):
    '''
    Calculate coverage of a bigwig file for a given peak.

    Args:
        bw: Path to bigwig file.
        peak: Peak object.
        rolling_percent: Percent of peak width for rolling mean.

    Returns:
        Pandas dataframe with columns `position` and `coverage`.
    '''

    cov = pyBigWig.open(str(bw)).values(
        peak.chrom, peak.start, peak.end)

    rolling = int(peak.width * rolling_percent)

    return pd.DataFrame({
        'position': np.arange(peak.start, peak.end),
        'coverage': pd.Series(cov).rolling(rolling).mean()
    })


def plot_coverage_line(df_bigwig, sample, peak: Peak, rolling_percent=.1):
    '''
    Plot coverage of a bigwig files for a given peak.

    Args:
      df_bigwig: Pandas dataframe with columns `sample` and `bigwig`.
      sample: Sample name.
      peak: Peak object for the region.
      rolling_percent: Percent of peak width for rolling mean.
    '''

    if isinstance(peak, str):
        peak = Peak.from_str(peak)

    if not any(df_bigwig['sample'] == sample):
        raise ValueError('Sample is not included in the metadata')

    for row in tqdm(df_bigwig.itertuples()):
        df = coverage_bigwig(row.bigwig, peak, rolling_percent)

        if row.sample == sample:
            df_sample = df
        else:
            sns.lineplot(data=df, x='position', y='coverage',
                         color="0.2", linewidth=1, alpha=.1)

    sns.lineplot(data=df_sample, x='position', y='coverage',
                 color='red', linewidth=2)

    sns.despine(trim=True, left=True, bottom=True)
    plt.xticks([])

    plt.xlabel(str(peak))
    plt.ylabel('Coverage')


class CoveragePlot:
    '''
    Plot coverage of a given region for samples.

    Args:
      samples: Dictionary of sample names and bigwig paths.
      region: Region string.
      highlight_samples: List of sample names to highlight.
      color: Color of the coverage plot.
      highlight_color: Color of the highlighted coverage plot.
      color: Color of the coverage plot.
      highlight_color: Color of the highlighted coverage plot.
      label: Whether to label the samples.
      aspect: Aspect ratio of the plot.
    '''

    def __init__(self, samples, region, highlight_samples=None,
                 color='#5a5a83', highlight_color='#b1615c',
                 label=True, aspect=1):
        self.samples = samples
        self.region = region

        self.highlight_samples = set()
        if not isinstance(highlight_samples, type(None)):
            self.highlight_samples = highlight_samples

        self.color = color
        self.highlight_color = highlight_color
        self.label = label
        self.aspect = aspect

    def data(self):
        '''
        '''
        df = list()

        for sample, path in self.samples.items():
            _df = coverage_bigwig(path, Peak.from_str(self.region))
            _df['sample'] = sample
            df.append(_df)

        return pd.concat(df)

    def _label(self, data, color):
        ax = plt.gca()
        sample = data['sample'][0]
        ax.text(0, .1, sample, color='black', ha="left",
                va="center", transform=ax.transAxes)

    def _fill(self, data, color):
        ax = plt.gca()
        line = ax.lines[0]
        x, y = line.get_xydata().T

        if data['sample'][0] in self.highlight_samples:
            color = self.highlight_color
        else:
            color = self.color

        ax.fill_between(x, 0, y, color=color, alpha=.7)

    def plot(self):
        df = self.data()

        sns.set_theme(style="white",
                      rc={"axes.facecolor": (0, 0, 0, 0)})

        pal = sns.cubehelix_palette(10, rot=-.2, light=.7)
        g = sns.FacetGrid(df, row="sample", height=1, palette=pal,
                          aspect=len(self.samples) / self.aspect)

        g.map(sns.lineplot, "position", 'coverage',
              errorbar=None, color="w", lw=3)
        g.refline(y=0, linewidth=2, linestyle="-",
                  color='black', clip_on=False)

        g.map_dataframe(self._fill)

        if self.label:
            g.map_dataframe(self._label)

        g.figure.subplots_adjust(hspace=-.75)

        g.set_titles("")
        g.set(xticks=[], xlabel=self.region)
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        return g


def plot_coverage(samples, region, highlight_samples=None, aspect=1,
                  color='#5a5a83', highlight_color='#b1615c', label=True):
    '''
    Plot coverage of a given region for samples.

    Args:
      samples: Dictionary of sample names and bigwig paths.
      region: Region string.
      highlight_samples: List of sample names to highlight.
      color: Color of the coverage plot.
      highlight_color: Color of the highlighted coverage plot.
      color: Color of the coverage plot.
      highlight_color: Color of the highlighted coverage plot.
      label: Whether to label the samples.
      aspect: Aspect ratio of the plot.
    '''
    return CoveragePlot(
        samples, region, highlight_samples=highlight_samples, aspect=aspect,
        color=color, highlight_color=highlight_color, label=label
    ).plot()


class QQPlot:
    '''
    QQ plot for p-values.

    Args:
      pvalues: as dataframe where each row is a sample and each column 
        is a p-value.
      highlight: List of sample names to highlight. 
      ci: Confidence interval for the expected p-values.
      eps: Epsilon value to avoid zero p-values.
    '''

    def __init__(self, pvalues, highlight=None, ci=0.95, eps=10e-15):
        self.pvalues = pvalues
        self.highlight = highlight
        self.ci = ci
        self.eps = eps

    def data(self):
        pvalues = np.array(self.pvalues) + self.eps
        n = len(self.pvalues)

        df = pd.DataFrame({
            'observed': -np.log10(self.pvalues)
        })

        if not isinstance(self.highlight, type(None)):
            highlight = np.array(self.highlight)
            assert self.pvalues.shape == self.highlight.shape
            df['highlight'] = self.highlight

        df = df.sort_values('observed', ascending=False)

        df['expected'] = -np.log10(np.arange(1, n+1)/(n+1))
        df['clower'] = -np.log10(beta.ppf(
            (1 - self.ci) / 2, np.arange(1, n+1), np.arange(n, 0, -1)
        ))
        df['cupper'] = -np.log10(beta.ppf(
            (1 + self.ci) / 2, np.arange(1, n+1), np.arange(n, 0, -1)
        ))
        return df

    def plot(self):
        df = self.data()

        plt.figure(figsize=(4, 4), dpi=300)
        plt.scatter(df.expected, df.observed, color='black', alpha=.5)
        plt.fill_between(df.expected, df.clower, df.cupper,
                         alpha=0.5, color='grey')

        if not isinstance(self.highlight, type(None)):
            _df = df[df['highlight']]
            plt.scatter(_df.expected, _df.observed, color='red')

        lim = min(df.expected.max(), df.observed.max())
        plt.plot([0, lim], [0, lim], color='black', linestyle='--')

        plt.xlabel("$-log_{10}(P_{Expected})$")
        plt.ylabel("$-log_{10}(P_{Observed})$")
        sns.despine()


def qq_plot(pvalues, highlight=None, ci=0.95, eps=10e-15):
    '''
    QQ plot for p-values.

    Args:
      pvalues: as dataframe where each row is a sample and each column 
        is a p-value.
      highlight: List of sample names to highlight. 
      ci: Confidence interval for the expected p-values.
      eps: Epsilon value to avoid zero p-values.
    '''
    QQPlot(pvalues, highlight=highlight, ci=ci, eps=eps).plot()


class CountsPlot:
    '''
    Plot observed vs expected counts.

    Args:
      counts: Observed counts as dataframe where each row is a sample and 
        columns are peaks.
      counts_mean: Expected counts as dataframe where each row is a sample 
        and columns are peaks.
      highlight: List of sample names to highlight.
    '''

    def __init__(self, counts, counts_mean, highlight=None):
        self.counts = counts
        self.counts_mean = counts_mean
        self.highlight = highlight

    def data(self):
        df = pd.DataFrame({
            'counts': self.counts,
            'counts_mean': self.counts_mean
        })

        if not isinstance(self.highlight, type(None)):
            df['highlight'] = self.highlight

        return df

    def plot(self):
        df = self.data()

        plt.figure(figsize=(4, 4), dpi=300)
        plt.scatter(df['counts'], df['counts_mean'],
                    color='black', alpha=.5)

        if not isinstance(self.highlight, type(None)):
            _df = df[df['highlight']]
            plt.scatter(_df['counts_mean'], _df['counts'], color='red')

        lim = max(df['counts'].max(), df['counts_mean'].max())
        plt.plot([0, lim], [0, lim], color='black', linestyle='--')

        plt.xlabel('Expected Counts')
        plt.ylabel('Observed Counts')
        sns.despine()


def plot_counts(counts, counts_mean, highlight=None):
    '''
    Plot observed vs expected counts.

    Args:
      counts: Observed counts as dataframe where each row is a sample and 
        columns are peaks.
      counts_mean: Expected counts as dataframe where each row is a sample 
        and columns are peaks.
      highlight: List of sample names to highlight.
    '''
    CountsPlot(counts, counts_mean, highlight=highlight).plot()


class RankPlot:
    '''
    Rank plot for samples for a given statistic.

    Args:
      stats: Statistics as a list.
      stats_name: Name of the statistic.
      highlight: List of sample names to highlight.
    '''

    def __init__(self, stats, stats_name='score', highlight=None):
        self.stats = stats
        self.stats_name = stats_name
        self.highlight = highlight

    def data(self):
        df = pd.DataFrame({
            self.stats_name: self.stats
        })

        if not isinstance(self.highlight, type(None)):
            df['highlight'] = self.highlight

        df = df.sort_values(self.stats_name)
        df['Rank'] = df[self.stats_name].rank()

        return df

    def plot(self):
        df = self.data()

        plt.figure(figsize=(4, 4), dpi=300)

        plt.scatter(df['Rank'], df[self.stats_name],
                    color='black', alpha=.5)

        if not isinstance(self.highlight, type(None)):
            _df = df[df['highlight']]
            plt.scatter(_df['Rank'], _df[self.stats_name], color='red')

        plt.ylabel(self.stats_name)
        sns.despine()


def plot_rank(stats, stats_name='score', highlight=None):
    '''
    Rank plot for samples for a given statistic.

    Args:
      stats: Statistics as a list.
      stats_name: Name of the statistic.
      highlight: List of sample names to highlight.
    '''
    RankPlot(stats, stats_name=stats_name, highlight=highlight).plot()


class VolconaPlot:
    '''
    Volcano plot for p-values and z-scores.

    Args:
      pval: P-values as a list.
      zscore: Z-scores as a list.
      pval_cutoff: P-value cutoff for significance.
      zscore_cutoff: Z-score cutoff for significance.
    '''

    def __init__(self, pval, zscore, pval_cutoff=.05, zscore_cutoff=(-1, 1)):
        self.pval = pval
        self.zscore = zscore
        self.pval_cutoff = pval_cutoff
        self.zscore_cutoff = zscore_cutoff

    def data(self):
        df = pd.DataFrame({
            '-log(padj)': -np.log10(self.pval),
            'zscore': self.zscore
        })

        return df

    def plot(self):
        df = self.data()

        plt.figure(figsize=(5, 5), dpi=300)

        plt.scatter(df['zscore'], df['-log(padj)'], color='gray', alpha=.5)

        plt.axhline(-np.log10(self.pval_cutoff), color='black',
                    linestyle='--', alpha=.25)
        plt.axvline(self.zscore_cutoff[0], color='black',
                    linestyle='--', alpha=.25)
        plt.axvline(self.zscore_cutoff[1], color='black',
                    linestyle='--', alpha=.25)

        _df = df[(df['zscore'] < self.zscore_cutoff[0])
                 & (df['-log(padj)'] > -np.log10(self.pval_cutoff))]
        plt.scatter(_df['zscore'], _df['-log(padj)'], color='blue')

        _df = df[(df['zscore'] > self.zscore_cutoff[1])
                 & (df['-log(padj)'] > -np.log10(self.pval_cutoff))]
        plt.scatter(_df['zscore'], _df['-log(padj)'], color='red')

        plt.xlabel('z-score')
        plt.ylabel('$-log_{10}(P_{adj})$')


def plot_volcona(pval, zscore, pval_cutoff=.05, zscore_cutoff=(-1, 1)):
    '''
    Volcano plot for p-values and z-scores.

    Args:
      pval: P-values as a list.
      zscore: Z-scores as a list.
      pval_cutoff: P-value cutoff for significance.
      zscore_cutoff: Z-score cutoff for significance.
    '''
    VolconaPlot(pval, zscore, pval_cutoff=pval_cutoff,
                zscore_cutoff=zscore_cutoff).plot()


def plot_umap(df_counts, color=None, legend_title='color'):
    '''
    Plot umap plot for counts.

    Args:
      df_counts: Counts as dataframe where each row is a sample and
        columns are peaks.
      color: Color of the points.
      legend_title: Title of the legend.
    '''
    try:
        import umap
    except ImportError:
        raise ImportError(
            'Install umap package with '
            '`pip install umap-learn` to plot umap plot')

    emb = umap.UMAP(random_state=42).fit(df_counts.values)
    _df = pd.DataFrame({
        'UMAP1': emb.embedding_[:, 0],
        'UMAP2': emb.embedding_[:, 1],
    })

    if isinstance(color, type(None)):
        ax = sns.scatterplot(data=_df, x="UMAP1", y="UMAP2")
    else:
        _df[legend_title] = color
        ax = sns.scatterplot(data=_df, x="UMAP1", y="UMAP2", hue=legend_title)


def plot_corr_heatmap(df_counts, row_var=None, cbar_pos=(-.06, .45, .03, .2),
                      cmap='twilight_shifted', vmin=-1, vmax=1):
    '''
    Plot correlation heatmap between samples from counts.

    Args:
      df_counts: Counts as dataframe where each row is a sample and
        columns are peaks.
      row_var: Variable to color the rows.
      cbar_pos: Position of the colorbar.
      cmap: Colormap.
      vmin: Minimum value for the colorbar.
      vmax: Maximum value for the colorbar.
    '''
    corr = df_counts.T.corr()

    if not isinstance(row_var, type(None)):
        _colors = sns.hls_palette(len(set(row_var)), l=0.5, s=0.8)
        lut = dict(zip(set(row_var), _colors))
        row_colors = row_var.map(lut)
    else:
        row_colors = None

    ax = sns.clustermap(corr, cmap=cmap, col_colors=row_colors, cbar_pos=cbar_pos,
                        dendrogram_ratio=.1, yticklabels=False, xticklabels=False,
                        vmin=vmin, vmax=vmax)

    if not isinstance(row_var, type(None)):
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=lut[name]) for name in lut]

        plt.legend(handles, lut, bbox_to_anchor=(0, 1),
                   bbox_transform=plt.gcf().transFigure, loc='upper left')

    return ax
