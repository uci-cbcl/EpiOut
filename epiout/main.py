from pathlib import Path
import click
import pandas as pd


@click.command()
@click.option(
    "--bed",
    help="bed file containing accessible regions",
    required=True
)
@click.option(
    "--alignments",
    help="file containing bam files for each samples. "
    "Either each line is bam file (Ex: `a.bam\nb.bam\c.bam`) or "
    "first columns is sample and second column is path to bam "
    "Ex: (a\ta.bam\nb\tb.bam\nc\tc.bam) ",
    required=True
)
@click.option(
    "--output_prefix",
    help="Output prefix for {prefix}.raw_counts.csv, {prefix}.counts.csv, {prefix}.bed",
    required=True,
)
@click.option(
    "--cores",
    help="Number of cores to use for paralell counting.",
    default=-1,
    type=int,
)
@click.option(
    "--mapq",
    help="Minimum read quality of reads.",
    default=10, type=int
)
@click.option(
    "--min_count", help="Minimum count at least one sample.", default=100, type=int
)
@click.option(
    "--min_percent_sample",
    help="minimum percentange of sample peak with at least one read.",
    default=0.5,
    type=click.FloatRange(0, 1),
)
@click.option("--subset_chrom", help="Subset chr1 to 22 and X,Y", is_flag=True)
@click.option("--file_format", help="File format of count table", default="parquet")
def cli_epicount(
    bed,
    alignments,
    output_prefix,
    cores=-1,
    mapq=10,
    min_count=100,
    min_percent_sample=0.5,
    subset_chrom=False,
    file_format="parquet",
):
    from epiout.dataset import EpiOutDataset
    dataset = EpiOutDataset(bed, alignments, njobs=cores,
                            subset_chrom=subset_chrom)
    df_raw = dataset.count(mapq, min_count, min_percent_sample)

    if file_format == "parquet":
        df_raw.to_parquet(f"{output_prefix}.raw_counts.parquet")
        dataset.df_counts.to_parquet(f"{output_prefix}.counts.parquet")
    elif file_format == "csv":
        df_raw.to_csv(f"{output_prefix}.raw_counts.csv")
        dataset.df_counts.to_csv(f"{output_prefix}.counts.csv")
    else:
        raise ValueError("`csv` and `parquet` is supported file formats.")

    dataset.bed.to_bed(f"{output_prefix}.raw.bed")
    dataset.bed_filtered.to_bed(f"{output_prefix}.bed")


@click.command()
@click.option(
    "--count_table",
    help="Path to count table csv containing peaks (rows) x samples (columns)",
    required=True,
)
@click.option(
    "--output_prefix",
    help="Output prefix for `{prefix}.h5ad and` and `{prefix}.result.csv`.",
    required=True,
)
@click.option(
    "--confounders",
    help="Known confounding factors as tsv file where "
    "each row is a sample and each column is a confounding factor.",
)
@click.option(
    "--bottleneck_size",
    help="Predefined bottleneck size. If not provide, will be tuned.",
    type=int,
)
@click.option(
    "--cores", help="Number of cores to use for paralell counting.", default=1, type=int
)
def cli_epiout(count_table, output_prefix, confounders=None, bottleneck_size=None, cores=1):

    if count_table.endswith(".parquet"):
        df_counts = pd.read_parquet(count_table).T
    elif count_table.endswith(".csv"):
        df_counts = pd.read_csv(count_table, index_col=0).T
    else:
        raise ValueError("`csv` and `parquet` is supported file formats.")

    if confounders is not None:
        confounders = pd.read_csv(
            confounders, sep="\t", index_col=0, header=None)

    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(cores)

    from epiout.epiout import EpiOut
    epiout = EpiOut(bottleneck_size=bottleneck_size)

    if bottleneck_size is None:
        df_auc = epiout.tune_bottleneck(df_counts, confounders)
        df_auc.to_csv(f"{output_prefix}.bottleneck_auc.csv", index=False)

    result = epiout(df_counts, confounders)

    result.save(f"{output_prefix}.h5ad")
    df_result = result.results()
    df_result.to_csv(f"{output_prefix}.result.csv", index=False)


@click.command()
@click.option(
    "--bed", help="bed file containing accessible regions to annotate", required=True
)
@click.option(
    "--config",
    help="config file containing bed and hic files to annotate against",
    required=True,
)
@click.option(
    "--output_prefix",
    help="Output prefix for `{prefix}.annotation.csv`, `{prefix}.contact.csv`,"
    "`{prefix}.gtf.csv`, `{prefix}.genes.csv` files.",
    required=True,
)
@click.option("--gtf", help="GTF files to obtain genes.")
@click.option(
    "--chrom_sizes",
    help="Chrom sizes files `faidx {fasta} -i chromsizes > {chrom_sizes}`",
)
@click.option("--counts", help="Path count table of accessible reads.")
@click.option(
    "--gene_types",
    help="List of gene types separated by comma.",
    default="protein_coding",
)
@click.option(
    "--hic_bin_vicinity",
    help="Number of bin distance between peaks to consider in conduct analysis.",
    type=int,
    default=200,
)
@click.option(
    "--hic_normalization", help="Hic normalization method.", type=str, default="NONE"
)
@click.option(
    "--hic_binsize", help="Hic bin size.", type=int, default=5_000
)
@click.option(
    "--hic_min_read", help="Minimum number of hic min reads to include", type=int, default=10
)
def cli_epiannot(
    bed,
    config,
    output_prefix,
    gtf=None,
    chrom_sizes=None,
    counts=None,
    gene_types="protein_coding",
    hic_bin_vicinity=200,
    hic_normalization="NONE",
    hic_binsize=5_000,
    hic_min_read=10,
):
    gene_types = gene_types.split(",")
    from epiout.annotate import EpiAnnot
    annot = EpiAnnot(
        config=config,
        gtf=gtf,
        chrom_sizes=chrom_sizes,
        counts=counts,
        gene_types=gene_types,
        hic_bin_vicinity=hic_bin_vicinity,
        hic_normalization=hic_normalization,
        hic_binsize=hic_binsize,
        hic_min_read=hic_min_read,
    )
    annot.save_annotate(bed, output_prefix)

    annotation = Path(f'{output_prefix}.annotation.csv')
    gtf_annot = Path(f'{output_prefix}.gtf.csv')
    contact = Path(f'{output_prefix}.interaction.csv')

    if annotation.exists() and gtf_annot.exists() and contact.exists():
        if counts.endswith('.h5ad'):
            from epiout.gene import GeneLink
            from epiout.result import EpiOutResult
            result = EpiOutResult.load(counts)
            gene_link = GeneLink(annotation, gtf_annot, contact)
            df_link = gene_link.predict(result)
            df_link.to_csv(f"{output_prefix}.genes.csv")


@click.command()
@click.option(
    "--tissue",
    help="Tissue, cell line, primiary or differentialed cell "
)
@click.option(
    "--output_dir",
    help="Output directory of create config and download files",
    required=True
)
@click.option(
    "--tf",
    help="If include TF-ChIP-seq marks",
    is_flag=True
)
@click.option(
    "--hic",
    help="If inclice Hi-C",
    is_flag=True
)
@click.option(
    "--ccres",
    help="If include ccres",
    is_flag=True
)
def cli_epiannot_create(tissue, output_dir, tf=False, hic=False, ccres=False):
    from epiout.annotation_config import create_config
    create_config(tissue, output_dir, tf=tf, hic=hic, ccres=ccres)


@click.command()
def cli_epiannot_list():
    '''
    List available tissues, cell lines, primary or differentiated cells.
    '''
    from epiout.annotation_config import read_hic, read_histone, read_tf
    df_histone = read_histone()
    tissues = set(df_histone.index)

    df_tf = read_tf()
    tissues = tissues.union(set(df_tf.index))

    df_hic = read_hic()
    tissues = tissues.union(set(df_hic.index))

    for tissue in tissues:
        print(f'- {tissue}')
