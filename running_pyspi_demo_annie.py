#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:37:57 2022

@author: osboxes
"""

# Load all packages needed
import pandas as pd
import numpy as np
from os import chdir, getcwd
from scipy.stats import zscore
from scipy.signal import detrend
import matplotlib.pyplot as plt # For plotting
from scipy.stats import zscore
import seaborn as sns

# Load in raw BOLD fMRI time-series data for example subject
# TODO: make this reproducible
BOLD_fMRI_TS_data = np.load("/media/sf_Shared_Folder/github/CNS_2022_pyspi/tutorial_example_data/BOLD_fMRI_TS_example.npy")


def plot_data(z):
    plt.subplots()
    plt.pcolormesh(z,vmin=-1.5,vmax=1.5,cmap=sns.color_palette('icefire_r',as_cmap=True))
    plt.colorbar()

    # ticks = [t+0.5 for t in range(len(labels))]
    # plt.yticks(ticks=ticks, labels=labels)
    plt.xlabel('Time (fMRI frame)')
    plt.ylabel('Brain Region')
    plt.show()

plot_data(BOLD_fMRI_TS_data)
