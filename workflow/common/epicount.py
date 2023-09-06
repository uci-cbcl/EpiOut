import pandas as pd
from epiout.dataset import EpiOutDataset


gr = EpiOutDataset.read_bed(snakemake.input['bed'])
counts = EpiOutDataset.count_reads(gr, snakemake.input['bam'])

pd.Series(counts).to_csv(snakemake.output['counts'])
