#!/usr/bin/env python3

# Load all packages needed
import pandas as pd
import numpy as np
import pickle
import dill
from os import chdir, getcwd
from scipy.stats import zscore
from scipy.signal import detrend
import matplotlib.pyplot as plt # For plotting
from scipy.stats import zscore
import seaborn as sns
from pyspi.calculator import Calculator
from matplotlib import colors


# Load in raw BOLD fMRI time-series data for example subject
# TODO: make this reproducible
BOLD_fMRI_TS_data = np.load("/media/sf_Shared_Folder/github/CNS_2022_pyspi/tutorial_example_data/BOLD_fMRI_TS_example.npy")
ROI_info = pd.read_csv("/media/sf_Shared_Folder/github/CNS_2022_pyspi/tutorial_example_data/ROI_info.csv")

# Plotting the data
def plot_data(z, labels):
    plt.subplots()
    plt.pcolormesh(z,cmap=sns.color_palette('icefire_r',as_cmap=True))
    plt.colorbar()

    ticks = [t+0.5 for t in range(len(labels))]
    plt.yticks(ticks=ticks, labels=labels)
    plt.xlabel('Time (fMRI frame)')
    plt.ylabel('Brain Region')
    plt.show()

plot_data(BOLD_fMRI_TS_data, labels = ROI_info.ROI)

# These two lines show the main usage of the calculator: simply instantiate and compute.
calc = Calculator(BOLD_fMRI_TS_data, fast=True)
calc.compute()

with open('/media/sf_Shared_Folder/github/CNS_2022_pyspi/tutorial_example_data/pyspi_calc.pkl', 'wb') as f:
    dill.dump(calc, f)

# Load saved calc object
with open('/media/sf_Shared_Folder/github/CNS_2022_pyspi/tutorial_example_data/pyspi_calc.pkl','rb') as f:
    calc = dill.load(f)

# We can now inspect the results table, which includes hundreds of pairwise interactions
print(calc.table)


def plot_mpi(S,identifier,labels,ax=None):
    """ Plot a given matrix of pairwise interactions, annotating the process labels and identifier
    """
    if ax is None:
        _, ax = plt.subplots()
    plt.sca(ax)

    # Use a diverging cmap if our statistic goes negative (and a sequential cmap otherwise)
    if np.nanmin(S) < 0.:
        maxabsval = max(abs(np.nanmin(S)),abs(np.nanmax(S)))
        norm = colors.Normalize(vmin=-maxabsval, vmax=maxabsval)
        plt.imshow(S,cmap='coolwarm',norm=norm)
    else:
        plt.imshow(S,cmap='Reds',vmin=0)

    plt.xticks(ticks=range(len(labels)),labels=labels,rotation=45)
    plt.yticks(ticks=range(len(labels)),labels=labels)
    plt.xlabel('Symbol')
    plt.ylabel('Symbol')
    plt.title(identifier)
    plt.colorbar()

# Iterate through the three methods (covariance, dynamic time warping, Granger causality, and convergent cross-mapping), extract and plot their matrices
spis = ["cov_EmpiricalCovariance", 'anm','pec']


for identifier in spis:
    # Simply index an SPI in the output table, which will give you an MxM dataframe of pairwise interactions (where M is the number of processes)
    S = calc.table[identifier]

    # Plot this dataframe
    plot_mpi(S = S, identifier = identifier, labels = ROI_info.ROI)
    
    
#### Extracting the data for downstream purposes
with open('/media/sf_Shared_Folder/github/CNS_2022_pyspi/tutorial_example_data/pyspi_calc_table.pkl', 'wb') as f:
    dill.dump(calc.table, f)


#### Exporting the data to R