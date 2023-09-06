from warnings import warn
from pathlib import Path
import pysam
import pandas as pd
import pyranges as pr
from joblib import Parallel, delayed
from epiout.utils.common import chroms


class EpiOutDataset:
    '''
    Dataset object to read bed and bam files and count reads for each peak.

    Args:
      bed: path to bed file or pyranges object.
      alignments: path to metadata file or list of paths to bam files 
          or dict of sample name and path to bam file.
      njobs: number of jobs to run in parallel during counting.
      slack: slack to merge peaks.
      subset_chrom: subset chromosomes to only those in the bam files.
    '''

    def __init__(self, bed, alignments, njobs=1, slack=200, subset_chrom=False):
        self.bed = self.read_bed(bed, slack, subset_chrom)
        self.alignments = self.read_alignments(alignments)
        self.njobs = njobs

    def read_bed(self, bed, slack=200, subset_chrom=False):
        '''
        Read bed file and overlapping merge peaks with slack of 
          by default 200bp, subset chromosomes of 
          chr1, chr2, ..., chrX, chrY, chrM, if `subset_chrom` is True, 
          and sort by chromosome and start position.  
        '''
        if isinstance(bed, str):
            bed = pr.read_bed(bed)
        bed = bed.drop().merge(strand=False, slack=slack).sort()

        if subset_chrom:
            bed = bed[bed.Chromosome.isin(chroms)]

        self.valid_bed(bed)
        return bed

    def read_alignments(self, alignments):
        '''
        Read alignments file and return dict of sample name 
          and path to bam file.
        '''
        if isinstance(alignments, dict):
            pass
        elif isinstance(alignments, list):
            alignments = {
                Path(path).stem: path
                for path in alignments
            }
        elif isinstance(alignments, str):
            df = pd.read_csv(alignments, header=None, sep='\t')
            if df.columns.size == 1:
                alignments = {
                    Path(path).stem: path
                    for path in df.iloc[:, 0]
                }
            elif df.columns.size == 2:
                alignments = {
                    row.iloc[0]: row.iloc[1]
                    for _, row in df.iterrows()
                }
            else:
                raise ValueError(
                    'Metadata file must have 1 or 2 columns.'
                    '1st column is sample name and 2nd column is path to bam.'
                )
        else:
            raise ValueError(
                'Metadata must be dict of sample name and path to bam,'
                'or list of paths to bam,'
                'or path to metadata file that contains columns of sample name and path to bam.'
            )
        self._valid_alignments(alignments)
        return alignments

    @staticmethod
    def _valid_alignments(alignments: dict):
        for bam in alignments.values():
            if not Path(bam).exists():
                raise FileNotFoundError(f'Bam file `{bam}` not found.')

    @staticmethod
    def _valid_bed(bed: pr.PyRanges):
        if bed.empty:
            raise ValueError('Bed file is empty.')

    @staticmethod
    def count_reads(gr, bam, mapq=10):
        '''
        Read bam file and count reads for each peak.

        Args:
          gr: pyranges object of peaks.
          bam: path to bam file or pysam.AlignmentFile object.
          mapq: minimum mapping quality.
        '''
        if isinstance(bam, str):
            bam = pysam.AlignmentFile(bam, "rb")

        counts = dict()

        for chrom, df_chrom in iter(gr):

            reads = filter(lambda r: r.mapping_quality > mapq,
                           bam.fetch(chrom))
            try:
                read = next(reads)
            except StopIteration:
                warn(f'{chrom} not exist in the bam skipping.')
                continue

            peaks = (
                (peak, f'{peak.Chromosome}:{peak.Start}-{peak.End}')
                for peak in df_chrom.itertuples(index=False)
            )
            peak, peak_name = next(peaks)
            counts[peak_name] = 0

            while True:
                try:
                    if read.reference_end <= peak.Start:
                        read = next(reads)
                    elif read.reference_start >= peak.End:
                        peak, peak_name = next(peaks)
                        counts[peak_name] = 0
                    else:
                        counts[peak_name] += 1
                        read = next(reads)
                except StopIteration:
                    break

        return counts

    def _count_samples(self, mapq=10):
        count_samples = Parallel(n_jobs=self.njobs)(
            delayed(self.count_reads)(self.bed, bam, mapq)
            for bam in self.alignments.values()
        )
        count_samples = dict(zip(self.alignments.keys(), count_samples))

        df = pd.DataFrame(count_samples).fillna(0).astype(int)
        df.index.name = 'peak'

        return df

    @staticmethod
    def _filters(df_raw, min_count=100, min_percent_sample=0.5):
        '''
        min_count: minimum count at least one sample.
        min_num_sample: minimum number of sample peak with at least one read.
        '''
        min_num_sample = df_raw.shape[1] * min_percent_sample

        filter_read_in_max = df_raw.max(axis=1) > min_count
        filter_min_read = (df_raw > 0).sum(axis=1) > min_num_sample

        return filter_read_in_max & filter_min_read

    def count(self, mapq=10, min_count=100, min_percent_sample=0.5):
        '''
        Count reads for each peak and filter peaks with minimum count and
            minimum number of samples with at least one read.

        Args:
          mapq: minimum mapping quality.
          min_count: minimum count at least one sample.
          min_percent_sample: minimum number of sample peak with 
            at least one read.
        '''
        df_raw = self._count_samples(mapq)

        filters = self._filters(df_raw, min_count, min_percent_sample)

        self.df_counts = df_raw[filters]
        self.bed_filtered = self.bed[filters]

        return df_raw
