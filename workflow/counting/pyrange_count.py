import pyranges as pr


gr_bed = pr.read_bed(snakemake.input['bed']) \
    .drop().merge(strand=False, slack=200)

gr_bam = pr.read_bam(snakemake.input['bam'], mapq=10)

gr = gr_bed.count_overlaps(gr_bam, overlap_col='count')

gr.drop(drop=['Name', 'Score', 'Strand']) \
  .to_bed(snakemake.output['counts'])
