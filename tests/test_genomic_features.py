import pyranges as pr
from epiout.utils.genomic_features import tss_genes, tes_genes
from conftest import gtf


def test_tss_genes():
    gr_gtf = pr.read_gtf(gtf)
    gr = tss_genes(gr_gtf)
    assert gr.df.shape[0] == sum(gr_gtf.Feature == "gene")


def test_tes_genes():
    gr_gtf = pr.read_gtf(gtf)
    gr = tes_genes(gr_gtf)
    assert gr.df.shape[0] == sum(gr_gtf.Feature == "gene")
