import pandas as pd
import pyranges as pr


def tss_genes(gr: pr.PyRanges):
    '''
    Return a PyRanges object with TSSs of genes.
    '''
    gr = gr[gr.Feature == "gene"]
    df_pos = gr[gr.Strand == "+"].df
    df_neg = gr[gr.Strand == "-"].df

    df_pos['End'] = df_pos['Start'] + 1
    df_neg['Start'] = df_neg['End'] - 1

    df = pd.concat([df_pos, df_neg])
    df['Feature'] = 'tss'

    return pr.PyRanges(df)


def tes_genes(gr: pr.PyRanges):
    '''
    Return a PyRanges object with TESs of genes.
    '''
    gr = gr[gr.Feature == "gene"]
    df_pos = gr[gr.Strand == "+"].df
    df_neg = gr[gr.Strand == "-"].df

    df_pos['Start'] = df_pos['End'] - 1
    df_neg['End'] = df_neg['Start'] + 1

    df = pd.concat([df_pos, df_neg])
    df['Feature'] = 'tes'

    return pr.PyRanges(df)
