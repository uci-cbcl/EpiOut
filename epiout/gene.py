import numpy as np
import pandas as pd
from pkg_resources import resource_filename


def import_onnxruntime():
    '''
    Import optional dependency onnxruntime if installed in env,
      otherwise raise an error.
    '''
    try:
        import onnxruntime as rt
    except ImportError as exc:
        raise ImportError(
            'onnxruntime is not installed.'
            'Please install it by running:'
            'pip install onnxruntime'
        ) from exc
    return rt


class GeneLink:
    '''
    Predict aberrant genes based on the features from promoters, proximal 
      and distal enhancers.
    '''

    def __init__(self, annotation, annotation_gtf, interaction):
        self.df_annotation = pd.read_csv(annotation)
        self.df_gtf = pd.read_csv(annotation_gtf)

        self.df_gtf = self.df_gtf[self.df_gtf["gene_type"] == "protein_coding"]
        self.df_interaction = pd.read_csv(interaction)

        self.df_peak_gene = self._peak_gene()

    def _peak_gene(self):
        df = (
            self.df_annotation.set_index("peak")[["annotation"]]
            .join(self.df_gtf.set_index("peak")[["Feature", "gene_id"]])
            .reset_index()
            .drop_duplicates()
            .set_index("peak")
        )

        df = df[~df["gene_id"].isna()]
        return df

    def _promoter_gene(self):
        return self.df_peak_gene[
            (self.df_peak_gene["annotation"] == "promoter")
            & (self.df_peak_gene["Feature"].isin({"tss", "five_prime_utr"}))
        ]

    def promoter_gene(self, df_result):
        '''
        Calculate promoter outlier scores for each gene.
        '''
        return df_result.set_index("peak").join(self._promoter_gene(), how="inner")

    def _promoters(self):
        return self._promoter_gene().index

    def _melt(self, df, value_name):
        return (
            df[self._promoters()]
            .reset_index()
            .melt(id_vars="index", value_name=value_name)
            .rename(columns={"index": "sample"})
            .set_index(["sample", "peak"])
        )

    def promoter_features(self, result, eps=1e-16):
        '''
        Calculate promoter outlier scores for each gene.
        '''
        df = pd.concat(
            [
                self._melt(result.outlier, "promoter_outlier").astype("int"),
                self._melt(result.l2fc.abs(), "l2fc"),
                self._melt(-np.log10(np.maximum(result.pval, eps)),
                           "-log(pval)"),
            ],
            axis=1,
        )
        return (
            df.reset_index(level=0)
            .join(self._promoter_gene(), how="inner")
            .reset_index()
            .groupby(["gene_id", "sample"])
            .agg({
                "promoter_outlier": "max",
                "l2fc": "max",
                "-log(pval)": "max",
            })
        )

    def _proximal_gene(self):
        return self.df_peak_gene[
            ~(
                (self.df_peak_gene["annotation"] == "promoter")
                & (self.df_peak_gene["Feature"].isin({"tss", "five_prime_utr"}))
            )
        ]

    def proximal_gene(self, df_result):
        '''
        Calculate proximal enhancer scores for each gene.
        '''
        return df_result.set_index("peak").join(self._proximal_gene(), how="inner")

    def proximal_features(self, df_result):
        df_proximal = self.proximal_gene(df_result)
        df_proximal["Proximal-Enhancer"] = df_proximal["l2fc"].abs()
        return df_proximal.groupby(["gene_id", "sample"])[["Proximal-Enhancer"]].max()

    def _distal_gene(self):
        promoters = set(self._promoter_gene().index)
        df = (
            self.df_interaction[self.df_interaction["peak"].isin(promoters)]
            .set_index("peak")
            .join(self.df_peak_gene)
            .rename(columns={"peak_other": "peak"})
        )
        del df["annotation"]
        return (
            df[["abc_score", "distance", "gene_id", "peak"]]
            .drop_duplicates()
            .set_index("peak")
        )

    def distal_gene(self, df_result):
        '''
        Calculate distal enhancer scores for each gene.
        '''
        df_promoter = self._promoter_gene().set_index("gene_id", append=True)
        df_proximal = self._proximal_gene().set_index("gene_id", append=True)

        promoter_proximal = set(df_promoter.index).union(df_proximal.index)

        df_distal = self._distal_gene().set_index("gene_id", append=True)
        df_distal = df_distal[~df_distal.index.isin(promoter_proximal)]

        return df_result.set_index("peak").join(
            df_distal.reset_index(level=1), how="inner"
        )

    def distal_features(self, df_result):
        '''
        Calculate distal enhancer scores for each gene.
        '''
        df_distal = self.distal_gene(df_result)
        df_distal["Distal-Enhancer"] = df_distal["abc_score"] * \
            df_distal["l2fc"].abs()
        return df_distal.groupby(["gene_id", "sample"])[["Distal-Enhancer"]].sum()

    def features(self, result):
        '''
        Prepare features for prediction from promoters, proximal and 
          distal enhancers.
        '''
        df_result = result.results()
        return (
            self.promoter_features(result)
            .join(self.proximal_features(df_result), how="outer")
            .join(self.distal_features(df_result), how="outer")
            .fillna(0)
        )

    def predict(self, result):
        '''
        Predict aberrant genes based on the features from promoters, 
          proximal and distal enhancers.
        '''
        features = self.features(result)

        rt = import_onnxruntime()
        sess = rt.InferenceSession(
            resource_filename('epiout', 'models/epigene.onnx'))
        onnx_pred = sess.run(None, {
            feature: features[feature].values.astype(np.float32)
            for feature in features.columns
        })
        features['Score'] = onnx_pred[1][:, 1]
        return features
