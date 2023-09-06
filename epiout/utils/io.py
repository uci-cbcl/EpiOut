import pandas as pd


def df_batch_writer(df_iter, output):
    '''
    Write a batch of dataframes to a csv file.
    '''
    df = next(df_iter)
    with open(output, "w") as f:
        df.to_csv(f)

    with open(output, "a") as f:
        for df in df_iter:
            df.to_csv(f, header=False)
