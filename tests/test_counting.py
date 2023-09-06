import pytest
import pandas as pd
import pyranges as pr
from epiout.dataset import EpiOutDataset
from conftest import bam1, bam2, bam3


@pytest.fixture
def gr_peaks():
    return pr.from_dict({
        'Chromosome': ['chr1', 'chr1', 'chr17'],
        'Start': [10032487, 100351106, 10251039],
        'End': [10033539, 100353427, 10252020]
    })


def test_EpiOutDataset_count_reads(gr_peaks):

    counts = EpiOutDataset.count_reads(gr_peaks, bam1)

    counts_expected = {
        'chr1:10032487-10033539': 1346,
        'chr1:100351106-100353427': 3217,
        'chr17:10251039-10252020': 330
    }
    assert counts == counts_expected


def test_EpiOutDataset_filters():

    df_raw = pd.DataFrame({
        'test1': [1346, 3217, 50, 200],
        'test2': [70, 60, 90, 0],
        'test3': [208, 717, 0, 0],
        'peak': ['chr1:10032487-10033539',
                 'chr1:100351106-100353427',
                 'chr17:10251039-10252020',
                 'chr19:10251039-10252020']
    }).set_index('peak')

    filters = EpiOutDataset._filters(
        df_raw, min_count=100, min_percent_sample=0.5)

    assert filters.tolist() == [True, True, False, False]


def test_EpiOutDataset_count(gr_peaks):

    alignments = [bam1, bam2, bam3]
    dataset = EpiOutDataset(gr_peaks, alignments, njobs=2)

    df_raw = dataset.count(min_percent_sample=0.9)

    df_raw_expected = pd.DataFrame({
        'test1': [1346, 3217, 330],
        'test2': [1110, 3381, 367],
        'test3': [208, 717, 0],
        'peak': ['chr1:10032487-10033539',
                 'chr1:100351106-100353427',
                 'chr17:10251039-10252020']
    }).set_index('peak')

    pd.testing.assert_frame_equal(df_raw, df_raw_expected)

    df_expected = pd.DataFrame({
        'test1': [1346, 3217],
        'test2': [1110, 3381],
        'test3': [208, 717],
        'peak': [
            'chr1:10032487-10033539',
            'chr1:100351106-100353427']
    }).set_index('peak')
    pd.testing.assert_frame_equal(dataset.df_counts, df_expected)
