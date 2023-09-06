import pandas as pd
import pyranges as pr
from typing import Union


def _peak_str(df):
    return df.Chromosome.astype(str) + ':' \
        + df.Start.astype(str) + '-' \
        + df.End.astype(str)


def peak_str(df: Union[pd.DataFrame, pr.PyRanges]):
    '''
    Convert genomic intervals to peak string

    Args:
        df: genomic intervals as pyranges or dataframe
    '''

    if isinstance(df, pr.PyRanges):
        return df.assign('peak', _peak_str)
    elif isinstance(df, pd.DataFrame):
        return df.assign(peak=_peak_str)
    else:
        raise TypeError('df must be pyranges or dataframe')