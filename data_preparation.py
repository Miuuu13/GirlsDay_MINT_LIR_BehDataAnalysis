# This file contains the code to get the actual data to do the behavioral analysis

# It is not part of the code, that will be used for the GirlsDay

# The script I use for the GirlsDay is called "behavioral_data_analysis.py"
#%%

## Script to bring data from .xlsx files into a form that can easily be analysed

## Data uploaded in folder "s4-7_extDays"

## JGU Jupyter Webserver 

#https://cbdm-01.zdv.uni-mainz.de/~muro/teaching/p4b/mod4-2/SoSe23/c0_set_up/c0_jgu_jupyter_notebook_server.html

#%%
# impports
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import scipy
import os
import re
#Folder paths 

# relative path to folder in cwd that contains all .xlsx files (behavioral data)
input_path = './input' #s4-7_extDays'

output_folder = './output'

#%%
# 1.) Parse filename and extract metadata 
def parse_filename(filename):
    """ Parses the filename to get metadata """
    # Remove the file extension (.xlsx)
    name = os.path.splitext(filename)[0]

    #filename schema: id_batch_s4/s5/s6/s7_rp/rm_7/12_4kHz
    #id can be 3 or 4 digit, bacht either A or B (capital letter always), 
    #s stands for session 4 to 7, 7_4 or 12_4 means 7,4 or 12,4 kHz

    # Adjusted regex to parse the file name properly
    # Example base filename: 934_B_s4_rp_12_4kHz, 1037_B_s4_rp_12_4kHz
    pattern = r'^(\d+)_([AB])_s(\d+)_([rp]{2}|rm)_(\d+)_([0-9]+)kHz$' #regex
    match = re.match(pattern, name)
    
    if match:
        return match.groups()  # returns a tuple
    else:
        print(f"Filename '{filename}' didn't match pattern!")
        return None

#%%
# 2.) 
# Initialize the dictionary to store processed DataFrames
extDays_dict = {}

# Get all the Excel files in the directory with .xlsx
files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

#%%
# check dict
extDays_dict # empty dict
#%%

# 3.) Process each Excel file
for file in files:
    file_path = os.path.join(input_path, file)
    
    # Extract metadata from the filename
    parsed = parse_filename(file)
    if not parsed:
        continue  # skip files that don't match the naming pattern
    
    # Read the row 39 as header (zero-indexed = row 38)
    headers_df = pd.read_excel(file_path, skiprows=38, nrows=0)
    headers = headers_df.columns.tolist()
    
    # Read the data starting from row 41 (zero-indexed = skip 40 rows)
    df = pd.read_excel(file_path, skiprows=40, header=None)
    
    # Assign correct headers
    df.columns = headers
    
    # Select only the required columns
    columns_needed = ["Trial time", "Mobility state(Immobile)", "CS+", "CS-"]

    df = df[columns_needed]


    
    # Rename columns
    df = df.rename(columns={
        "Trial time": "time_s",
        "Mobility state(Immobile)": "immobile",
        "CS+": "csp",
        "CS-": "csm"
    })

    #     # Check for non-numeric entries in 'immobile' (e.g. '-')
    # non_numeric_mask = ~df['immobile'].apply(lambda x: str(x).strip().isdigit())
    # non_numeric_values = df[non_numeric_mask]['immobile']
    # non_numeric_indices = df[non_numeric_mask].index

    # # Print summary
    # print(f"\nFile: {file}")
    # print(f"Non-numeric 'immobile' entries found: {len(non_numeric_values)}")
    # print("First few non-numeric entries and their indices:")
    # print(non_numeric_values.head())
    # print(non_numeric_indices[:5].tolist())

    #NOTE Always first 2 lines contain non numeric str values like "-" -> 
    #can be replaced by 0

    # Convert 'immobile' to numeric, replace non-numeric with 0
    df['immobile'] = pd.to_numeric(df['immobile'], errors='coerce').fillna(0).astype(int)

    
    # Add metadata as columns
    df['id'] = parsed[0]
    df['batch'] = parsed[1]
    df['session'] = parsed[2]
    df['rp_rm'] = parsed[3]
    # df['freq_1'] = parsed[4] #would save "7" or "12"; not needed
    # df['freq_2'] = parsed[5] #would save "4"; not needed
    
    # Store in dictionary
    key = os.path.splitext(file)[0]
    extDays_dict[key] = df


#%%

# 4. Combine all DataFrames into one big DataFrame
dfs = list(extDays_dict.values())
combined_df = pd.concat(dfs, keys=extDays_dict.keys(), names=['file', 'index'])

#%%

# 5.)  Save to CSV or Parquet
os.makedirs(output_folder, exist_ok=True)  # in case the output folder does not excist, it is created
combined_df_reset = combined_df.reset_index()  # flatten multi-index
combined_df_reset.to_csv(os.path.join(output_folder, 'combined_data.csv'), index=False)
#combined_df_reset.to_parquet(os.path.join(output_folder, 'combined_data.parquet'))

#Parquet is faster and more efficient to save large df to csv (if needed) - not pre-installed!

#%%
print("Processing complete. Combined DataFrame shape:", combined_df_reset.shape)

#%%

""" Save as checkpoint """

def save_df_as_csv(df, path):
    """Save a DataFrame with MultiIndex to CSV, preserving index columns."""
    df_to_save = df.reset_index()  # flatten index into columns
    df_to_save.to_csv(path, index=False)

def load_csv_as_df_outdated(path, index_cols):
    """Load a CSV and set specified columns as MultiIndex."""
    df = pd.read_csv(path)
    df.set_index(index_cols, inplace=True)
    return df
# combined_df has a MultiIndex with levels ['file', 'index']
save_df_as_csv(combined_df, './output/combined_with_index.csv') #NOTE renamed to checkpoint....

#%%
# Later, reload it with the same MultiIndex
df_reloaded = load_csv_as_df_outdated('./output/combined_with_index.csv', ['file', 'index'])

print(df_reloaded.index.names)  # ['file', 'index']

#%%
