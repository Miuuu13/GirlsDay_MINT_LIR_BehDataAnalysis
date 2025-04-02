# Girls'Day 3rd April, 2025
# Script to analyze behavioral data of mice.

# The checkpoint (file: "checkpoint_with_index.csv") should be placed in the current working directory. Link to the JGU Jupyter Webserver:
# https://cbdm-01.zdv.uni-mainz.de/~muro/teaching/p4b/mod4-2/SoSe23/c0_set_up/c0_jgu_jupyter_notebook_server.html

#%%

# imports (pre-installed libraries)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import mcolors
import scipy
import os
import re

#%%

# We need a function to load the multi-index correctly:

def load_multi_index_csv_as_df(path: str, index_cols: list) -> pd.DataFrame:
    """
    Loads a .csv file and applies the helper function transform_df_types to cast columns to the preferred data type.
    Additionally, the multi index is set.
    """
    # Force 'file' column to string directly on load
    dtype_overrides = {col: str for col in index_cols}
    df = pd.read_csv(path)
    df = transform_df_types(df)
    df.set_index(index_cols, inplace=True)
    return df

def transform_df_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Casts numeric columns to int (0,1), casts 'time_s' to float,
    and keeps certain columns as strings.
    """
    pres_cols = ['time_s', 'batch', 'session', 'rp_rm']

    for col in df.columns:
        if col == 'time_s':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
        elif col in pres_cols:
            df[col] = df[col].astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    return df

#%%

# 1.) Load the data 

data = load_multi_index_csv_as_df('./checkpoint_with_index.csv', ['file', 'index'])
print(data.index.names) # displays index 

#%%

# check the data type

def print_column_dtypes(df):
    print("\n Column and data type:")
    print("-" * 38) # just ----
    for col, dtype in df.dtypes.items():
        print(f"{col:<25} : {dtype}")
        print("-" * 38)
        
        
print_column_dtypes(data)

# Informations about the data
# The keys
# The data contains keys like "990_B_s4_rm_12_4kHz". This is a unique identifier for a session, where the animals behavior has been recorded.
# The first part "990_... is the animal id (number that identifies the animal)
# The second part "B" is the batch
# The third part "s4" is the session
# The fourth part is the info to which group the animal belongs to: "rm" = R- (suscepptible) and "rp" = R+ (resilient)
# The fifth part is the frequency that has been used for the tone "12_4kHz" or "7_4kHz"

#%%

# Take a look on the data:
data
# %%
data.head() # another function, that only displays the head of the data

#%%

import numpy as np
import pandas as pd

def add_freezing_column_for_cutoff(data: pd.DataFrame, freezing_cutoff: int = 25) -> pd.DataFrame:
    """
    Adds a 'freezing' column to the DataFrame:
    - 1: true freezing event (immobile streak >= freezing_cutoff)
    - 0: all else
    Assumes data has a MultiIndex (file, frame) and an 'immobile' column.
    """
    data = data.sort_index()
    data['freezing'] = 0

    for file, df in data.groupby(level=0):
        immobile = df['immobile'].values
        indices = df.index.get_level_values(1).values

        # Add padding to detect edges of streaks
        padded = np.pad(immobile, (1, 1), constant_values=(0, 0))
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0] - 1

        lengths = ends - starts + 1

        # Apply label 1 for valid freezing streaks
        for s, e, l in zip(starts, ends, lengths):
            if l >= freezing_cutoff:
                data.loc[(file, slice(indices[s], indices[e])), 'freezing'] = 1

    return data
# %%

# The columns we need:
# immobile - 1 if the mouse does not move
# csm, csp - 1 if tone is on
# CS- (csm) is the "safe tone" - nothing bad happened when this tone was on
# CS+ (csp) is the "fear tone" - traumatic event is linked to this tone
# We are interested in the behavior before, during and after the tone is on.
# For our analysis we want to get the freezing behavior
# Freezing refers to the mouse being in enormous fear
# It is defined as "immobile state lasting for at least 2 seconds"
# Meaning: we need all freezing events, that last 2 seconds or longer for each key (one animal recorded in one session)

# 2.) Add a new column called "freezing" to our data

# We now want to add a column that extracts the "real freezings" for immobile = 1
# 1 = real freezing event (immobile streak ≥ 25 frames → 1 second)
# 0 = fake/short freezing event (immobile streak < 25)
# 0 = not freezing (immobile == 0)
# Note: MultiIndex slicing in pandas: 
# you can’t slice unless the index is lexsorted (i.e., sorted properly by all levels)



def add_freezing_column(data: pd.DataFrame, freezing_cutoff: int = 25) -> pd.DataFrame:
    """
    Adds a 'freezing' column to the DataFrame:
    - 1: true freezing event (immobile streak >= freezing_cutoff)
    - 2: short immobile streak (immobile streak < freezing_cutoff)
    - 0: not immobile; also no freezing
    """
    data = data.sort_index()
    data['freezing'] = 0  # Initialize with zeros

    # Store updates in a list to apply in bulk
    freezing_updates = []

    for file in data.index.levels[0]:
        try:
            df = data.loc[file]
        except KeyError:
            continue

        immobile = df['immobile'].values
        idxs = df.index.values
        start = None
        duration = 0

        for i in range(len(immobile)):
            if immobile[i] == 1:
                if start is None:
                    start = idxs[i]
                    duration = 1
                else:
                    duration += 1
            else:
                if start is not None:
                    end = idxs[i - 1]
                    label = 1 if duration >= freezing_cutoff else 2
                    freezing_updates.append(((file, slice(start, end)), label))
                    start = None
                    duration = 0

        # Handle case where freezing extends to the end
        if start is not None:
            end = idxs[-1]
            label = 1 if duration >= freezing_cutoff else 2
            freezing_updates.append(((file, slice(start, end)), label))

    # Bulk apply all updates
    for (file, index_slice), label in freezing_updates:
        data.loc[(file, index_slice), 'freezing'] = label

    return data


#%%
data = add_freezing_column_for_cutoff(data)
#%%
print(data['freezing'].value_counts())

#%%

# 3.) Filter the data: for R+/R- and the session

def filter_group_session(df, group='rp', session='4'):
    """
    Filters a MultiIndex dataframe for a given group (rp or rm) and session.
    indef, file as levels
    Parameters:
    - df: MultiIndex DataFrame with 'rp_rm' and 'session' columns
    - group: 'rp' or 'rm'
    - session: e.g., '4', '5', etc. (can also pass as int)

    Returns:
    - Filtered DataFrame
    """
    return df[(df['rp_rm'] == group) & (df['session'] == str(session))].copy()

#%%
# filter all sessions in R+

# R+ for all 4 sessions
rp_s4_df = filter_group_session(data, group='rp', session='4')
rp_s5_df = filter_group_session(data, group='rp', session='5')
rp_s6_df = filter_group_session(data, group='rp', session='6')
rp_s7_df = filter_group_session(data, group='rp', session='7')

# TODO: same for R-
# R- for all 4 sessions
rm_s4_df = filter_group_session(data, group='rm', session='4')
rm_s5_df = filter_group_session(data, group='rm', session='5')
rm_s6_df = filter_group_session(data, group='rm', session='6')
rm_s7_df = filter_group_session(data, group='rm', session='7')


# %%

# 4.) Visualization

def compute_freezing_by_tone(df, n_csp=12, n_csm=4):
    """
    Computes total and normalized freezing counts during CS+ and CS− tones only.

    @params:
    - df: filtered MultiIndex dataframe (e.g., rp_s4_df)
    - n_csp: number of CS+ tones in session; 12 by default
    - n_csm: number of CS− tones in session; 4 by default

    @returns:
    - Dictionary with total freezing counts and the average freezing per tone
    """
    # Ensure correct types (esp. if loaded from .csv format) - remove?
    df['freezing'] = df['freezing'].astype(int)
    df['csp'] = df['csp'].astype(int)
    df['csm'] = df['csm'].astype(int)

    #Only count freezing frames DURING tones (csp, csm column ==1 (True))
    csp_freeze_count = df[(df['freezing'] == 1) & (df['csp'] == 1)].shape[0]
    csm_freeze_count = df[(df['freezing'] == 1) & (df['csm'] == 1)].shape[0]

    csp_per_tone = csp_freeze_count / n_csp if n_csp else 0
    csm_per_tone = csm_freeze_count / n_csm if n_csm else 0

    return {
        'csp_freeze_total': csp_freeze_count,
        'csm_freeze_total': csm_freeze_count,
        'csp_per_tone': csp_per_tone,
        'csm_per_tone': csm_per_tone
    }
    
#%%
# Sessions to plot
sessions = ['4', '5', '6', '7']
rp_csp_per_tone = []

# per session
for session in sessions:
    rp_df = filter_group_session(data, group='rp', session=session)
    stats = compute_freezing_by_tone(rp_df)
    rp_csp_per_tone.append(stats['csp_per_tone'])

# Plot
plt.figure(figsize=(8, 5))
plt.plot(sessions, rp_csp_per_tone, marker='o', label='R+ (CS+ freezing)')
plt.xlabel('Session')
plt.ylabel('CS+ Freezing per Tone')
plt.title('R+ (rp) CS+ Freezing Across Sessions')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
sessions = ['4', '5', '6', '7']
rp_csp, rp_csm = [], []
rm_csp, rm_csm = [], []

for session in sessions:
    rp_df = filter_group_session(data, group='rp', session=session)
    rm_df = filter_group_session(data, group='rm', session=session)

    rp_stats = compute_freezing_by_tone(rp_df)
    rm_stats = compute_freezing_by_tone(rm_df)

    rp_csp.append(rp_stats['csp_per_tone'])
    rp_csm.append(rp_stats['csm_per_tone'])
    rm_csp.append(rm_stats['csp_per_tone'])
    rm_csm.append(rm_stats['csm_per_tone'])

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# R+ subplot
axes[0].plot(sessions, rp_csp, marker='o', label='CS+')
axes[0].plot(sessions, rp_csm, marker='o', label='CS−')
axes[0].set_title('R+ (rp) Freezing per Tone')
axes[0].set_xlabel('Session')
axes[0].set_ylabel('Freezing per Tone')
axes[0].legend()
axes[0].grid(True)

# R− subplot
axes[1].plot(sessions, rm_csp, marker='o', label='CS+')
axes[1].plot(sessions, rm_csm, marker='o', label='CS−')
axes[1].set_title('R− (rm) Freezing per Tone')
axes[1].set_xlabel('Session')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Freezing per Tone Over Sessions (CS+ vs CS−)', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
""" tone dictionary """

fps = 25  # Frames per second

# Tone (CS−, habituation)
tone1_start = 180 * fps
tone1_duration = 30 * fps
tone1_break = 60 * fps
tone1_reps = 4

# Tone (CS+, main tones)
tone2_start = tone1_start + tone1_reps * (tone1_duration + tone1_break)
tone2_duration = 30 * fps
tone2_break = 60 * fps
tone2_reps = 12

# Calculate CS− epochs
csm_epochs = []
for i in range(tone1_reps):
    start = tone1_start + i * (tone1_duration + tone1_break)
    end = start + tone1_duration
    csm_epochs.append((int(start), int(end)))

# Calculate CS+ epochs
csp_epochs = []
for i in range(tone2_reps):
    start = tone2_start + i * (tone2_duration + tone2_break)
    end = start + tone2_duration
    csp_epochs.append((int(start), int(end)))

# Combine into shared dictionary
shared_tone_epochs = {
    'csp': csp_epochs,
    'csm': csm_epochs
}

#check
print("CS− epochs:", csm_epochs)
print("CS+ epochs:", csp_epochs)
print("Master file epochs:", shared_tone_epochs)
# %%

# To get a list of the animals
def list_animals(df):
    files = df.index.levels[0]
    animal_ids = sorted(set(f.split('_')[0] + '_' + f.split('_')[1] for f in files))
    return animal_ids

# filter R+ and R- animals
rp_animals = data[data['rp_rm'] == 'rp']
rm_animals = data[data['rp_rm'] == 'rm']
#%%
list_animals(rm_animals)
#%%
list_animals(rp_animals)

#%%
# 5.) Heatmap

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------
# FUNCTION: Extract freezing for one animal
# ----------------------------------------

def extract_cs_freezing_for_animal(df, animal_key, tone_epochs, tone_type='csp'):
    """
    Returns a list of freezing counts for each CS+ or CS− tone for a single animal.

    Parameters:
    - df: full data (with MultiIndex)
    - animal_key: full file name (e.g. '990_B_s4_rp_12_4kHz')
    - tone_epochs: dict with 'csp' and 'csm' epoch lists
    - tone_type: 'csp' or 'csm'

    Returns:
    - List of freezing frame counts for each tone
    """
    try:
        animal_df = df.loc[animal_key]
    except KeyError:
        return [np.nan] * len(tone_epochs[tone_type])

    freezing = animal_df['freezing']
    freezing_counts = []

    for start, end in tone_epochs[tone_type]:
        if start in freezing.index and end in freezing.index:
            tone_freezing = freezing.loc[start:end]
        else:
            tone_freezing = freezing.loc[start:]
        freezing_counts.append(tone_freezing.loc[start:end].sum())

    return freezing_counts

# ----------------------------------------
# FUNCTION: Build freezing matrix for group
# ----------------------------------------

def build_group_freezing_matrix(df, group_label, tone_epochs, tone_type='csp'):
    """
    Builds a matrix of CS+ freezing counts per tone for all animals in a group.

    Parameters:
    - df: full dataset with 'rp_rm' column
    - group_label: 'rp' or 'rm'
    - tone_epochs: dictionary from shared_tone_epochs
    - tone_type: 'csp' or 'csm'

    Returns:
    - DataFrame: rows = animals, columns = tone number
    """
    df = df.copy()
    df.index = df.index.set_levels(df.index.levels[0].astype(str), level=0)  # ensure file index is string
    group_df = df[df['rp_rm'] == group_label]
    files = group_df.index.levels[0]

    result = {}

    for file in files:
        counts = extract_cs_freezing_for_animal(df, file, tone_epochs, tone_type)
        result[file] = counts

    return pd.DataFrame(result).T  # rows = animals, cols = tone #s

# ----------------------------------------
# FUNCTION: Plot side-by-side heatmaps
# ----------------------------------------

def plot_cs_heatmaps(rp_matrix, rm_matrix, tone_type='CS+'):
    """
    Plots two heatmaps side-by-side for R+ and R− groups.

    Parameters:
    - rp_matrix: DataFrame from build_group_freezing_matrix (R+)
    - rm_matrix: DataFrame from build_group_freezing_matrix (R−)
    - tone_type: 'CS+' or 'CS−' (for title)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.heatmap(rp_matrix, ax=axes[0], cmap="YlGnBu", cbar=False)
    axes[0].set_title(f'R+ (Resilient) - {tone_type}')
    axes[0].set_ylabel('Animal')
    axes[0].set_xlabel('Tone #')

    sns.heatmap(rm_matrix, ax=axes[1], cmap="YlGnBu")
    axes[1].set_title(f'R− (Susceptible) - {tone_type}')
    axes[1].set_xlabel('Tone #')

    plt.suptitle(f'Freezing During {tone_type} Tones', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# %%
# Assuming shared_tone_epochs was already defined earlier
rp_cs_matrix = build_group_freezing_matrix(data, group_label='rp', tone_epochs=shared_tone_epochs, tone_type='csp')
rm_cs_matrix = build_group_freezing_matrix(data, group_label='rm', tone_epochs=shared_tone_epochs, tone_type='csp')

plot_cs_heatmaps(rp_cs_matrix, rm_cs_matrix, tone_type='CS+')

# %%
""" new freezing heatmap """

fps = 25  # frames per second
tone_duration = 30 * fps  # all tones are 30 seconds

# normalized freezing per tone
def extract_normalized_freezing_per_tone(df, animal_key, tone_epochs, tone_type='csp'):
    """
    Returns normalized freezing per tone (freezing frames / tone length).
    """
    try:
        animal_df = df.loc[animal_key]
    except KeyError:
        return [np.nan] * len(tone_epochs[tone_type])

    freezing = animal_df['freezing']
    results = []

    for start, end in tone_epochs[tone_type]:
        if start in freezing.index and end in freezing.index:
            tone_freezing = freezing.loc[start:end].sum()
        else:
            tone_freezing = freezing.loc[start:].sum()
        results.append(tone_freezing / (end - start + 1))  # normalize
    return results

def build_normalized_freezing_matrix(df, group_label, tone_epochs, tone_type='csp'):
    df = df.copy()
    df.index = df.index.set_levels(df.index.levels[0].astype(str), level=0)
    group_df = df[df['rp_rm'] == group_label]
    files = group_df.index.levels[0]

    matrix = []

    for file in files:
        row = extract_normalized_freezing_per_tone(df, file, tone_epochs, tone_type)
        matrix.append(row)

    result_df = pd.DataFrame(matrix).T  # Transpose: rows = tones, columns = animals
    result_df.columns = files
    result_df.index = [f"Tone {i+1}" for i in range(result_df.shape[0])]
    return result_df


def plot_session_heatmaps_by_tone(data, group_label='rp', tone_type='csp', title_prefix='CS+'):
    sessions = ['4', '5', '6', '7']
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)

    for i, session in enumerate(sessions):
        session_df = data[data['session'] == session]
        matrix = build_normalized_freezing_matrix(session_df, group_label, shared_tone_epochs, tone_type)

        sns.heatmap(matrix, ax=axes[i], cmap="YlGnBu", vmin=0, vmax=1)
        axes[i].set_title(f"Session {session}")
        axes[i].set_xlabel('Animal')
        if i == 0:
            axes[i].set_ylabel('Tone #')
        else:
            axes[i].set_ylabel('')

    plt.suptitle(f'{group_label.upper()} - Freezing During {title_prefix} Tones (Normalized)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#%%
plot_session_heatmaps_by_tone(data, group_label='rp', tone_type='csp', title_prefix='CS+')

# %%
