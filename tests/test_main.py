import anndata
import pyranges as pr
import pandas as pd
from click.testing import CliRunner
from epiout.main import cli_epiannot, cli_epiannot_create, cli_epiannot_list, cli_epicount, cli_epiout
from conftest import count_table, bed, config, chrom_sizes, epiout_h5ad, gtf


def test_cli_epicount(tmp_path):

    csv = str(tmp_path / 'config.csv')
    pd.DataFrame({
        'sample': ['test1', 'test2', 'test3'],
        'bam': [
            'tests/data/test1.bam',
            'tests/data/test2.bam',
            'tests/data/test3.bam'
        ]
    }).to_csv(csv, index=False, header=False, sep='\t')

    bed = str(tmp_path / 'intervals.bed')
    pr.from_dict({
        'Chromosome': ['chr1', 'chr1', 'chr17'],
        'Start': [10032487, 100351106, 10251039],
        'End': [10033539, 100353427, 10252020]
    }).to_bed(bed)

    output_prefix = (tmp_path / 'test')
    output_prefix.mkdir()

    runner = CliRunner()
    result = runner.invoke(cli_epicount, [
        '--bed', bed,
        '--metadata', csv,
        '--output_prefix', str(output_prefix),
        '--min_count', '500',
        '--file_format', 'parquet'
    ])
    assert result.exit_code == 0

    df_bed_raw = pr.read_bed(str(tmp_path / 'test.raw.bed'), as_df=True)
    df_counts_raw = pd.read_parquet(tmp_path / 'test.raw_counts.parquet')
    df_bed = pr.read_bed(str(tmp_path / 'test.bed'), as_df=True)
    df_counts = pd.read_parquet(tmp_path / 'test.counts.parquet')

    assert df_bed_raw.shape[0] == 3
    assert df_bed.shape[0] == 2
    assert df_counts_raw.shape == (3, 3)
    assert df_counts.shape == (2, 3)

    runner = CliRunner()
    result = runner.invoke(cli_epicount, [
        '--bed', bed,
        '--metadata', csv,
        '--output_prefix', str(output_prefix),
        '--min_count', '500',
        '--file_format', 'csv'
    ])
    assert result.exit_code == 0
    df_counts = pd.read_csv(tmp_path / 'test.counts.csv')
    assert df_counts.shape == (2, 4)

    csv = str(tmp_path / 'config.csv')
    pd.DataFrame({
        'bam': [
            'tests/data/test1.bam',
            'tests/data/test2.bam',
            'tests/data/test3.bam'
        ]
    }).to_csv(csv, index=False, header=False, sep='\t')

    runner = CliRunner()
    result = runner.invoke(cli_epicount, [
        '--bed', bed,
        '--metadata', csv,
        '--output_prefix', str(output_prefix),
        '--min_count', '500',
        '--file_format', 'csv'
    ])
    assert result.exit_code == 0
    df_counts = pd.read_csv(tmp_path / 'test.counts.csv')
    assert df_counts.shape == (2, 4)


def test_cli_epiout(tmp_path, counts):
    prefix = str(tmp_path / 'outliers')
    confounders_path = str(tmp_path / 'confounders.tsv')
    pd.DataFrame({
        'sample': counts.index.tolist(),
        'groups': ['a'] * 10 + ['b'] * 10
    }).to_csv(confounders_path, index=False, header=False, sep='\t')

    result = CliRunner().invoke(cli_epiout, [
        '--count_table', count_table,
        '--confounders', str(confounders_path),
        '--output_prefix', prefix,
    ])
    assert result.exit_code == 0

    adata = anndata.read_h5ad(prefix + '.h5ad')
    assert set(adata.layers.keys()) == {
        'counts', 'counts_mean', 'l2fc', 'padj', 'pval'
    }
    assert adata.n_obs == 20
    assert adata.n_vars == 1107
    assert adata.layers['pval'].shape == (20, 1107)

    df_result = pd.read_csv(prefix + '.result.csv')
    assert all(df_result['padj'] < .05)


def test_cli_epiannot_create(tmp_path):
    output_dir = (tmp_path / 'motor_neuron')
    result = CliRunner().invoke(cli_epiannot_create, [
        '--tissue', 'motor_neuron',
        '--output_dir', str(output_dir),
        '--tf',
        '--ccres'
    ])
    assert result.exit_code == 0


def test_cli_epiannot_list(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli_epiannot_list, [])
    assert result.exit_code == 0
    assert len(result.output.split('\n')) == 214


def test_cli_epiannot(tmp_path):

    output_prefix = (tmp_path / 'test')

    runner = CliRunner()
    result = runner.invoke(cli_epiannot, [
        '--bed', bed,
        '--config', config,
        '--gtf', gtf,
        '--output_prefix', str(output_prefix),
        '--chrom_sizes', chrom_sizes,
        '--counts', epiout_h5ad
    ])
    assert result.exit_code == 0

    df_annot = pd.read_csv(f'{output_prefix}.annotation.csv')
    df_gtf = pd.read_csv(f'{output_prefix}.gtf.csv')
    df_interact = pd.read_csv(f'{output_prefix}.interaction.csv')
    df_genes = pd.read_csv(f'{output_prefix}.genes.csv')

    assert set(df_genes.columns) == {
        'gene_id', 'sample', 'promoter_outlier', 'l2fc', '-log(pval)',
        'Proximal-Enhancer', 'Distal-Enhancer', 'Score'
    }
