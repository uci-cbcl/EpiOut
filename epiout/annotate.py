import warnings
from typing import Union
import numpy as np
import pandas as pd
import pyranges as pr
from tqdm import tqdm
from epiout.utils import peak_str, df_batch_writer
from epiout.epiout import EpiOutResult
from epiout.hic import HicReader
from epiout.utils.genomic_features import tss_genes, tes_genes
from epiout.annotation_config import AnnotationConfig


class EpiAnnot:
    """
    Annotate peaks with epigenetic data.

    Args:
      config: Path to annotation config file.
      gtf: Path to GTF file.
      counts: Path to counts table.
      gene_types: Genes to use in the annotation.
      hic_bin_vicinity: Distance to search for HIC bins.
      hic_normalization: HIC normalization method.
      hic_binsize: HIC binsize.
      hic_min_read: 10.
    """

    def __init__(
        self,
        config: str,
        gtf: str = None,
        chrom_sizes: str = None,
        counts: str = None,
        gene_types: set = ("protein_coding",),
        hic_bin_vicinity=200,
        hic_normalization="NONE",
        hic_binsize=5_000,
        hic_min_read=10,
    ):
        if isinstance(config, str):
            config = AnnotationConfig(config)

        self.config = config
        self.gtf = gtf
        self.chrom_sizes = chrom_sizes
        self.counts = counts
        self.hic_bin_vicinity = hic_bin_vicinity
        self.hic_normalization = hic_normalization
        self.hic_binsize = hic_binsize
        self.hic_min_read = hic_min_read

        self.grs, self.hic = self._read()

        if gtf is not None:
            self.gtf = GTFAnnot(self.gtf, gene_types=gene_types)

        if counts is not None:
            self.counts, self.co_outlier = self._read_counts(self.counts)
        else:
            self.counts = dict()

    def _read(self):
        grs = dict()
        hic = None

        for name, files in self.config.to_dict().items():
            assert (
                self.chrom_sizes is not None
            ), "Chrom sizes must be provided to annotate contacts"
            if name == "hic":
                hic = HicReader(
                    files,
                    self.chrom_sizes,
                    self.hic_normalization,
                    self.hic_binsize,
                    self.hic_bin_vicinity,
                )
            else:
                grs[name] = self._read_bed(files)
        return grs, hic

    def _read_bed(self, files):
        return pr.PyRanges(
            pd.concat([pr.read_bed(i, as_df=True) for i in files])
        ).merge()

    def _read_counts(self, counts):
        co_outlier = set()

        if isinstance(counts, pd.DataFrame):
            df = counts
        elif counts.endswith(".h5ad"):
            result = EpiOutResult.load(counts)
            df = result.counts.T
            co_outlier = self._co_outliers(result.outlier)
        elif counts.endswith(".parquet"):
            df = pd.read_parquet(counts)
        elif counts.endswith(".csv"):
            df = pd.read_csv(counts, index_col=0)
        else:
            raise ValueError(
                "Only `.adata`, `.parquet`, `.csv` file " "formats are supported"
            )

        return df.sum(axis=1).to_dict(), co_outlier

    def _co_outliers(self, outliers):
        co_outlier = set()

        for _, row in outliers.iterrows():
            peaks = row[row].index
            for peak in peaks:
                for peak_other in peaks:
                    co_outlier.add((peak, peak_other))

        return co_outlier

    def annotate(self, gr):
        """
        Annotate peaks for chromatin features.

        Args:
          gr: genomic intervals as pyranges
        """
        if isinstance(gr, str):
            gr = pr.read_bed(gr).drop()

        df = self._annotate_grs(gr)

        if self.gtf is not None:
            df_gtf = self.gtf.annotate(gr)

            if (
                self._has_chipseq(df, "H3K4me3")
                & self._has_chipseq(df, "H3K27ac")
                & self._has_chipseq(df, "H3K4me1")
            ):
                df = self._annotate_feature(df, df_gtf)
        else:
            df_gtf = None

        return df, df_gtf

    def _annotate_feature(self, df, df_gtf):
        tss, proximal = self._tss_proximal(df_gtf)

        h3k4me3 = self._chipseq_marks(df, "H3K4me3")
        h3k27ac = self._chipseq_marks(df, "H3K27ac")
        h3k4me1 = self._chipseq_marks(df, "H3K4me1")

        active = h3k4me1 & h3k27ac
        poised = h3k4me1 & ~h3k27ac

        df["annotation"] = "NA"

        df["annotation"] = np.where(
            active, "active-enhancer", df["annotation"])
        df["annotation"] = np.where(
            poised, "poised-enhancer", df["annotation"])

        df["annotation"] = (
            np.where(
                df["annotation"].str.endswith("enhancer"),
                np.where(df.index.isin(proximal), "proximal-", "distal-"),
                "",
            )
            + df["annotation"]
        )

        df["annotation"] = np.where(
            h3k4me3 & df.index.isin(tss), "promoter", df["annotation"]
        )

        return df

    def _tss_proximal(self, df_gtf):
        tss = set(df_gtf[df_gtf["Feature"].isin(
            {"tss", "five_prime_utr"})].index)
        proximal = set(df_gtf[df_gtf["Feature"] != "intergenic"].index)
        return tss, proximal

    def _has_chipseq(self, df, mark):
        return df.columns.str.startswith(mark).any()

    def _chipseq_marks(self, df, mark):
        mark_cols = df.columns[df.columns.str.startswith(mark)]
        return df[mark_cols].sum(axis=1).astype("bool").astype("int")

    def _annotate_grs(self, gr):
        for name, _gr in self.grs.items():
            gr = gr.count_overlaps(_gr, overlap_col=name)

        df = gr.df
        df = peak_str(df)

        for name in self.grs:
            df[name] = df[name].astype("bool").astype("int")

        return df[["peak", *self.grs.keys()]].set_index("peak")

    def annotate_hic(self, gr):
        """
        Annotation of HIC contacts.

        Args:
          gr: genomic intervals as pyranges
        """
        if isinstance(gr, str):
            gr = pr.read_bed(gr).drop()

        df = gr.df

        for chrom, df_chrom in df.groupby("Chromosome"):
            df_nn = self._neighbors(pr.PyRanges(df_chrom))

            try:
                print(f"\t Annotating contacts {chrom}:")
                hic_scores = self.hic.contact_scores(chrom)
            except ValueError:
                warnings.warn(f"{chrom} not found in hic file. Skipping...")
                continue

            df_nn["peak_other"] = (
                df_nn["Chromosome"].astype(str)
                + ":"
                + df_nn["Start_b"].astype(str)
                + "-"
                + df_nn["End_b"].astype(str)
            )

            df_nn = df_nn[df_nn["peak"] != df_nn["peak_other"]]

            if len(self.co_outlier):
                df_nn["co_outlier"] = [
                    pair in self.co_outlier
                    for pair in tqdm(zip(df_nn["peak"], df_nn["peak_other"]))
                ]
            else:
                df_nn["co_outlier"] = False

            center = (df_nn.Start + df_nn.End) // 2
            center_other = (df_nn.Start_b + df_nn.End_b) // 2

            bins = center // self.hic_binsize
            bins_other = center_other // self.hic_binsize
            bins_index = hic_scores.shape[1] // 2 + (bins_other - bins)

            in_vicinity = bins_index < hic_scores.shape[1]

            bins = bins[in_vicinity]
            bins_index = bins_index[in_vicinity]
            df_nn = df_nn.loc[in_vicinity]

            df_nn["hic_score"] = np.array(
                [
                    hic_scores[
                        np.minimum(np.maximum(0, bins + i),
                                   hic_scores.shape[0] - 1),
                        np.minimum(
                            np.maximum(0, bins_index +
                                       j), hic_scores.shape[1] - 1
                        ),
                    ]
                    for i in range(-1, 2)
                    for j in range(-1, 2)
                ]
            ).max(axis=0)

            df_nn["distance"] = abs(center_other - center)

            df_nn["count"] = df_nn["peak_other"].map(self.counts).fillna(0)

            df_nn = df_nn.set_index("peak")[
                [
                    "peak_other",
                    "distance",
                    "count",
                    "hic_score",
                    "co_outlier",
                ]
            ]
            df_nn = self._abc_score(df_nn)

            yield df_nn[(df_nn["hic_score"] >= self.hic_min_read) | df_nn["co_outlier"]]

    def _neighbors(self, gr):
        df_nn = gr.join(gr, slack=self.hic_bin_vicinity * self.hic_binsize).df
        return peak_str(df_nn)

    def _abc_score(self, df):
        df["abc_score"] = df["hic_score"] * df["count"]
        df["norm"] = df.reset_index().groupby("peak")["abc_score"].sum()
        df["abc_score"] = (df["abc_score"] / df["norm"]).fillna(0)
        del df["norm"]
        return df

    def save_annotate(self, gr, output_prefix):
        '''
        Annotate peaks and save results.
        '''
        print("Annotating peaks...")
        df, df_gtf = self.annotate(gr)

        df.to_csv(output_prefix + ".annotation.csv")

        if not isinstance(df_gtf, type(None)):
            df_gtf.to_csv(output_prefix + ".gtf.csv")

        print("Annotating hic contacts...")
        if self.hic is not None:
            df_batch_writer(self.annotate_hic(
                gr), output_prefix + ".interaction.csv")


class GTFAnnot:
    def __init__(
        self,
        gtf_file: str,
        gene_types: set = ("protein_coding",),
        tss_upstream=1000,
        upsteam=10_000,
        downstream=1000,
    ):
        """
        Annotate intervals based on gtf file

        Args:
          gtf_file: Path to gtf file.
          gene_type: Gene types to consider `protein_coding`.
        """
        self.gtf_file = gtf_file
        self.gene_types = set(gene_types)
        self.tss_upstream = tss_upstream
        self.upsteam = upsteam
        self.downstream = downstream
        self.gr = self.read_features(gtf_file, gene_types)

    def read_features(self, gtf_file, gene_types):
        """
        Read features from gtf file

        Args:
          gtf_file: Path to gtf file.
          gene_type: Gene types to consider `protein_coding`.
        """
        gr = pr.read_gtf(gtf_file)

        if not isinstance(gene_types, type(None)):
            gr = gr[gr.gene_type.isin(gene_types)]

        df_intron = (
            gr.features.introns().merge(
                by=["Feature", "gene_id", "gene_type"]).df
        )

        gr_tss = tss_genes(gr)
        df_tss = gr_tss.extend({"5": self.tss_upstream}).df
        df_tes = tes_genes(gr).extend({"3": self.downstream}).df

        df_upstream = (
            gr_tss.extend({"5": self.upsteam})
            .extend({"5": -self.tss_upstream})
            .df.assign(Feature="upstream")
        )

        features = {"UTR", "exon", "five_prime_UTR", "three_prime_UTR"}
        df = (
            gr[gr.Feature.isin(features)]
            .merge(by=["Feature", "gene_id", "gene_type"])
            .df
        )
        df = pd.concat([df, df_intron, df_tss, df_tes, df_upstream])

        cols = [
            "Chromosome",
            "Start",
            "End",
            "Strand",
            "Feature",
            "gene_id",
            "gene_type",
        ]
        return pr.PyRanges(df[cols].drop_duplicates())

    def annotate(self, gr: Union[str, pr.PyRanges]):
        """
        Annotate peaks for gtf features.
        """
        if isinstance(gr, str):
            gr = pr.read_bed(gr)

        df = gr.join(self.gr, strandedness=False, apply_strand_suffix=False).df
        df = peak_str(df)

        cols = ["peak", "Feature", "gene_id", "gene_type"]
        df = df[cols]

        df_intergenic = gr.df
        df_intergenic = peak_str(df_intergenic)

        df_intergenic = (
            df_intergenic[~df_intergenic["peak"].isin(set(df["peak"]))]
            .assign(Feature="intergenic")
            .assign(gene_id=None)
            .assign(gene_type=None)[cols]
        )

        return pd.concat([df, df_intergenic]).drop_duplicates().set_index("peak")
