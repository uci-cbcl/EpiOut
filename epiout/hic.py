from typing import List
import numpy as np
from tqdm import tqdm


def import_hicstraw():
    '''
    Import optional dependency hic-straw if installed in env, 
      otherwise raise an error.
    '''
    try:
        import hicstraw  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            'hicstraw not installed. '
            'It is optional dependency for hic file reading.\n'
            'Please install it to use this feature with conda:\n'
            '\t`conda install -c bioconda hic-straw` \n'
            'or download with pip:\n'
            '\t`conda install -c conda-forge curl` \n'
            '\t`pip install hicstraw`'
        ) from exc
    return hicstraw


class HicReader:
    '''
    Read chromotain contacts information from Hi-C data from hic files.

    Args:
        hic_file_paths: List of hic file paths.
        chrom_sizes: Chromosome sizes file path.
        normalization: Normalization method for Hi-C data (default=None). 
          Valid normalization methods defined in hic-straw package.
        binsize: Bin size for Hi-C data (default=5_000).
        bin_vicinity: Number of bins to consider around 
          the diagonal (default=200) so defaults consider 1 Mb vicinity.
        batch_size: Batch size for reading Hi-C data (default=500_000). 
          Too large may cause memory issues and 
          too small may cause performance issues.
    '''

    def __init__(
        self,
        hic_file_paths: List[str],
        chrom_sizes: str,
        normalization="NONE",
        binsize=5_000,
        bin_vicinity=200,
        batch_size=500_000,
    ):
        self.hic_file_paths = hic_file_paths
        self.chrom_sizes = self._read_chrom_sizes(chrom_sizes)
        self.normalization = normalization
        self.binsize = binsize
        self.bin_vicinity = bin_vicinity
        assert bin_vicinity > 0
        self.batch_size = batch_size

    @staticmethod
    def _read_chrom_sizes(chrom_sizes):
        '''
        Read chromosome sizes from file.

        Returns:
            Dictionary of chromosome sizes.
        '''
        sizes = dict()
        with open(chrom_sizes) as f:
            for line in f:
                chrom, size = line.strip().split()
                sizes[chrom] = int(size)
        return sizes

    @staticmethod
    def match_chrom(chrom, chroms):
        '''
        Match chromosome name with the chromosome names in the hic file.

        Args:
            chrom: Chromosome name.
            chroms: Chromosome names in the hic file.

        Returns:
            Matched chromosome name.
        '''
        if chrom not in chroms:
            if chrom.startswith("chr"):
                chrom = chrom.replace("chr", "")
            else:
                chrom = "chr" + chrom

        if chrom not in chroms:
            raise ValueError(f"{chrom} not found in hic file.")

        return chrom

    def _read_contacts(self, chrom):
        '''
        Read contacts from hic file for given chromosome.

        Args:
            chrom: Chromosome name.

        Yields:
            Contact matrix for given chromosome as numpy array
        '''
        hicstraw = import_hicstraw()

        for f in self.hic_file_paths:
            hic = hicstraw.HiCFile(f)
            chroms = set(i.name for i in hic.getChromosomes())
            chrom = self.match_chrom(chrom, chroms)

            yield hic.getMatrixZoomData(
                chrom, chrom, "observed", self.normalization, "BP", self.binsize
            )

    def contact_scores(self, chrom):
        '''
        Calculate contact scores for given chromosome.

        Args:
            chrom: Chromosome name.

        Returns:
            Contact scores as numpy array.
        '''
        scores = list()

        chrom_size = self.chrom_sizes[self.match_chrom(
            chrom, self.chrom_sizes)]
        win_size = self.binsize * self.bin_vicinity

        hics = list(self._read_contacts(chrom))

        for i in tqdm(range(chrom_size // self.batch_size + 1)):
            pos = i * self.batch_size

            win1 = (pos, pos + self.batch_size - 1)
            win2 = (max(pos - win_size, 0), pos + win_size + self.batch_size)

            mat = np.sum([hic.getRecordsAsMatrix(*win1, *win2)
                         for hic in hics], axis=0)

            if mat.shape == (1, 1):
                mat = np.zeros(
                    (
                        self.batch_size // self.binsize,
                        (2 * win_size + self.batch_size) // self.binsize + 1,
                    )
                )

            if (pos - win_size) < 0:
                mat = np.hstack(
                    [np.zeros(
                        (mat.shape[0], abs(pos - win_size) // self.binsize)), mat]
                )

            for i, row in enumerate(mat):
                scores.append(row[i: i + 2 * self.bin_vicinity + 1])

        return np.array(scores)
