import pandas as pd
from epiout.gene import GeneLink
from epiout.result import EpiOutResult
from conftest import epi_annot, bed, epiout_h5ad


def test_gene(tmp_path, epi_annot):
    df_annot, df_gtf = epi_annot.annotate(bed)
    df_hic = pd.DataFrame({
        'peak': ['chr17:50033046-50033554'],
        'peak_other': ['chr17:59892254-59893918'],
        'distance': [100_000],
        'hic_score': [10],
        'co_outlier': [False],
        'abc_score': [15]
    })
    annotation = tmp_path / "annot.csv"
    df_annot.to_csv(annotation)

    annotation_gtf = tmp_path / "gtf.csv"
    df_gtf.to_csv(annotation_gtf)

    interaction = tmp_path / "interaction.csv"
    df_hic.to_csv(interaction)

    gene_link = GeneLink(annotation, annotation_gtf, interaction)

    result = EpiOutResult.load(epiout_h5ad)
    features = gene_link.predict(result)

    assert list(features.columns) == [
        'promoter_outlier', 'l2fc', '-log(pval)', 'Proximal-Enhancer', 'Distal-Enhancer', 'Score'
    ]
    assert list(features.index.names) == ['gene_id', 'sample']
