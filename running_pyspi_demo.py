#!/usr/bin/env python3
# -*- coding: utf-8 -*-


""" First, we need to set up some tools for downloading and minimally processing the data 
"""
import datetime, warnings
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.signal import detrend

def download(symbols,start,days):
    """ Download financial data from one of two sources, Yahoo Finance (yahoo) and the St. Lois Federal Reserve (fred)
    """
    end = start + datetime.timedelta(days=days)

    startstr = start.strftime('%Y-%m-%d')
    endstr = end.strftime('%Y-%m-%d')

    print(f'Obtaining {symbols} data from {startstr} to {endstr}...')
    close = yf.download(symbols, start=startstr, end=endstr, progress=True)['Close']

    # Match data up with weekdays
    weekdays = pd.date_range(start=startstr, end=endstr, freq='B')
    close = close.reindex(weekdays)

    # For any NaN's, propogate last valid value forward (and remove first value) 
    z = close.fillna(method='ffill').values.T[:,2:]

    # Make sure to always detrend and normalise your data, otherwise most statistics will give spurious results.
    return detrend(zscore(z,ddof=1,nan_policy='omit',axis=1))

# The FAANG tickers (Facebook/Meta, Amazon, Apple, Netflix, Google)
stocks = ['FB','AMZN','AAPL','NFLX','GOOGL'] 

# We'll download 140 days of data (corresponding to ~100 observations from business days)
ndays = 140 

# Set a recent(ish) starting date for the period
start_datetime = datetime.datetime.strptime('2014-01-01', '%Y-%m-%d') # Earliest date we will sample

print('Begin data download.')
z = download(stocks,start_datetime,ndays)
print(f'Done. Obtained MTS array of size {z.shape}')


""" Now we've got our data, we can inspect it to make sure everything looks OK.
"""
import matplotlib.pyplot as plt # For plotting
from scipy.stats import zscore
import seaborn as sns

def plot_data(z,labels):
    plt.subplots()
    plt.pcolormesh(z,vmin=-1.5,vmax=1.5,cmap=sns.color_palette('icefire_r',as_cmap=True))
    plt.colorbar()

    ticks = [t+0.5 for t in range(len(labels))]
    plt.yticks(ticks=ticks, labels=labels)
    plt.xlabel('Time')
    plt.ylabel('Symbol')
    plt.show()

plot_data(z,stocks)

""" Now that we have our data, and inspected it to make sure it looks OK, we can compute all pairwise interactions.
"""

from pyspi.calculator import Calculator

# These two lines show the main usage of the calculator: simply instantiate and compute.
calc = Calculator(z, fast=True)
calc.compute()

# We can now inspect the results table, which includes hundreds of pairwise interactions
print(calc.table)

""" One purpose of the calculator is that we can now extract pairwise matrices for every type of interaction.

For instance, below we will examine how covariance, dynamic time warping, and Granger causality differs when computing the relationship between the FAANG stock-market data.
"""

import matplotlib.pyplot as plt
from matplotlib import colors

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

    plt.xticks(ticks=range(len(stocks)),labels=labels,rotation=45)
    plt.yticks(ticks=range(len(stocks)),labels=labels)
    plt.xlabel('Symbol')
    plt.ylabel('Symbol')
    plt.title(identifier)
    plt.colorbar()

# Iterate through the three methods (covariance, dynamic time warping, Granger causality, and convergent cross-mapping), extract and plot their matrices
spis = ["cov_EmpiricalCovariance", "anm"]
for identifier in spis:
    # Simply index an SPI in the output table, which will give you an MxM dataframe of pairwise interactions (where M is the number of processes)
    S = calc.table[identifier]

    # Plot this dataframe
    plot_mpi(S,identifier,stocks)
    
""" First, we need to download several instances of the data type for training/testing our classifier.

This may take a while..
"""

# Add in some more symbols for the market rates
forex = ['DEXJPUS','DEXUSEU','DEXCHUS','DEXUSUK','DEXCAUS']

ninstances = 20

# Dict to store our results (given by the key (type,starting date))
database = {}
for i in range(ninstances):
    # Iterate the start datetime by ndays for each instance
    start_datetime = start_datetime + datetime.timedelta(days=ndays)
    print(f'[{i}/{ninstances-1}] Downloading data for the 140 period starting at {start_datetime}.')

    # Download the stock data (and plot)
    database[('stocks',start_datetime)] = download(stocks,start_datetime,ndays)
    plot_data(database[('stocks',start_datetime)],stocks)
    
    # Download the forex data (and plot)
    database[('forex',start_datetime)] = download(forex,start_datetime,ndays)
    plot_data(database[('forex',start_datetime)],forex)

print(f'Successfully downloaded {ninstances} datasets of both types of data.')