#%%

## Script for Girls' Day 3rd April, 2025

### Data: in folder "output" in cwd (current working directory 

### JGU Jupyter Webserver: https://cbdm-01.zdv.uni-mainz.de/~muro/teaching/p4b/mod4-2/SoSe23/c0_set_up/c0_jgu_jupyter_notebook_server.html

#%%
# imports (pre-installed libraries)
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import scipy
import os
import re

#%%

#Folder that contains the data we will use
output_folder = './output'
#%%

#We need a function to load the multi-index data

def load_multi_index_csv_as_df(path, index_cols):
    """
    Loads a CSV and applies transform_df_types, then sets MultiIndex.
    """
    df = pd.read_csv(path)

    # Protect index columns from being cast to int
    preserve_cols = index_cols + ['time_s','batch', 'session', 'rp_rm']
    df = transform_df_types(df, preserve_cols=preserve_cols)

    df.set_index(index_cols, inplace=True)
    return df

def transform_df_types(df, preserve_cols=None):
    """
    Casts all columns to int, except for columns in preserve_cols (kept as strings).
    Ignores non-numeric values.
    """
    if preserve_cols is None:
        preserve_cols = ['batch', 'session', 'rp_rm']

    for col in df.columns:
        if col in preserve_cols:
            df[col] = df[col].astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df


#%%
