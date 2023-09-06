import pytest
import pandas as pd
import pyranges as pr
from epiout.annotate import AnnotationConfig, EpiAnnot, GTFAnnot
from conftest import config, epi_annot, bed, gtf, chrom_sizes


@pytest.fixture
def annot_config():
    return AnnotationConfig(config)


def test_AnnotationConfig_to_dict(annot_config):
    annot = annot_config.to_dict()
    assert annot == {
        "H3K27ac": [
            "tests/data/ENCFF579MGE_chr17.bed",
            "tests/data/ENCFF031UQO_chr17.bed",
        ],
        "H3K4me1": [
            "tests/data/ENCFF605EAL_chr17.bed",
            "tests/data/ENCFF493FAP_chr17.bed",
        ],
        "H3K4me3": [
            "tests/data/ENCFF811BQV_chr17.bed",
            "tests/data/ENCFF246QVL_chr17.bed",
        ],
        "ccres_pls": ["tests/data/ccres_pls_chr17.bed"],
        "hic": ["tests/data/test_chr22.hic", "tests/data/test_chr22.hic"],
    }


def test_EpiAnnot__read(epi_annot):
    assert set(epi_annot.grs.keys()) == {
        "H3K27ac", "H3K4me1", "H3K4me3", "ccres_pls"}
    epi_annot.grs["H3K27ac"].df.shape == (4175, 3)
    epi_annot.grs["H3K4me1"].df.shape == (6099, 3)
    epi_annot.grs["H3K4me3"].df.shape == (1296, 3)
    epi_annot.grs["ccres_pls"].df.shape == (2405, 3)


def test_EpiAnnot_annotate(epi_annot):
    df_annot, df_gtf = epi_annot.annotate(bed)

    assert df_annot.loc[df_annot["H3K27ac"] == 1].shape == (420, 5)
    assert df_annot.loc[df_annot["H3K4me3"] == 1].shape == (96, 5)
    assert df_annot.loc[df_annot["H3K4me1"] == 1].shape == (406, 5)
    assert df_annot.loc[df_annot["ccres_pls"] == 1].shape == (94, 5)

    assert df_gtf.shape == (1675, 3)

    annot = EpiAnnot(config, gtf=gtf, chrom_sizes=chrom_sizes)
    df_annot, df_gtf = annot.annotate(bed)

    annot = EpiAnnot(config, chrom_sizes=chrom_sizes)
    df_annot, df_gtf = annot.annotate(bed)


def test_EpiAnnot_annotate_hic(epi_annot):
    gr = pr.PyRanges(
        chromosomes="chr22",
        starts=(40_000_000, 40_010_000, 40_110_000),
        ends=[40_001_000, 40_012_000, 40_115_000],
    )

    df = pd.concat(epi_annot.annotate_hic(gr))

    df_expected = pd.DataFrame({
        'peak': ['chr22:40000000-40001000', 'chr22:40010000-40012000', 'chr22:40010000-40012000',
                 'chr22:40110000-40115000', 'chr22:40110000-40115000'],
        'peak_other': ['chr22:40010000-40012000', 'chr22:40000000-40001000',
                       'chr22:40110000-40115000', 'chr22:40000000-40001000', 'chr22:40010000-40012000'],
        'distance': [10500, 10500, 101500, 112000, 101500],
        'count': [101, 51, 202, 51, 101],
        'hic_score': [2.0, 4.0, 2.0, 2.0, 2.0],
        'co_outlier': [False, False, False, False, False],
        'abc_score': [1.0, 0.3355263157894737, 0.6644736842105263,
                      0.3355263157894737, 0.6644736842105263]}
    )
    pd.testing.assert_frame_equal(df.reset_index(), df_expected,
                                  check_dtype=False, check_exact=False)


def test_GTFAnnot():
    gtf_annot = GTFAnnot(gtf)
    df = gtf_annot.annotate(bed)
    assert set(df.Feature) == {
        "tes",
        "UTR",
        "upstream",
        "tss",
        "exon",
        "intron",
        "intergenic",
    }
    assert df.columns.tolist() == ["Feature", "gene_id", "gene_type"]
