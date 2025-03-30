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
