import pandas as pd
import pytest
from epiout.annotate import EpiAnnot

bam1 = 'tests/data/test1.bam'
bam2 = 'tests/data/test2.bam'
bam3 = 'tests/data/test3.bam'

bed = 'tests/data/test_peaks.bed'
config = 'tests/data/annotation.config.yaml'

ccres_pls = 'tests/data/ccres_pls_chr17.bed'

hic = 'tests/data/ENCFF860TRU_chr17.hic'

chipseq = [
    'tests/data/ENCFF031UQO_chr17.bed',
    'tests/data/ENCFF246QVL_chr17.bed',
    'tests/data/ENCFF493FAP_chr17.bed',
    'tests/data/ENCFF579MGE_chr17.bed',
    'tests/data/ENCFF605EAL_chr17.bed',
    'tests/data/ENCFF811BQV_chr17.bed'
]

gtf = 'tests/data/test_chr17.gtf'
chrom_sizes = 'tests/data/chrom.sizes'
count_table = 'tests/data/test_counts.csv'

epiout_h5ad = 'tests/data/test.h5ad'
epiout_result = 'tests/data/test.result.csv'


@pytest.fixture
def counts():
    return pd.read_csv(count_table, index_col=0).T.astype('float32')


@pytest.fixture
def epi_annot():
    df_counts = pd.DataFrame(
        {
            "peak": [
                "chr22:40000000-40001000",
                "chr22:40010000-40012000",
                "chr22:40110000-40115000",
            ],
            "s1": [1, 1, 2],
            "s2": [50, 100, 200],
        }
    ).set_index("peak")
    return EpiAnnot(config, gtf=gtf, chrom_sizes=chrom_sizes,
                    counts=df_counts, hic_min_read=1)
