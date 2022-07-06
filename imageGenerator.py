#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:30:55 2022

@author: yasindu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()  #if you want to use seaborn themes with matplotlib functions
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVR
import scipy
import scipy.signal
import itertools
from itertools import chain
import math
from math import sqrt;
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image 
import PIL 

from scipy import signal
#%%
#df = pd.read_excel('/home/yasindu/Desktop/Soil_Spec/library_KS.xlsx')
df = pd.read_csv('/home/yasindu/Desktop/Datasets/lucasSoil.csv', low_memory=False)

#%%
dfTeml = df
df = df.drop(df.iloc[:, 0:4],axis = 1)
df = df.drop(df.iloc[:, 4207:],axis = 1)
df = df.drop(df.iloc[:, 4200:4207],axis = 1)
#df = df.drop(df.iloc[:, 1:8],axis = 1)

print("Readnig images - Done!")
#%%
# Data pre-processing
df2 = df
val = 0

#Remove outliers
# thresh = 2
# df3= df[df['EOC']<thresh]
# OC = df3['EOC'].tolist()
# df3 = df3.iloc[0:, 1:]

#create a list with spectral values without OC
# dataList = df3.values.tolist()
#dataArray = np.array(dataList[0])

# Get spectrogram using Hann window
# Then transform that spectrogram to log spectrogram


dataList = df.values.tolist()

data_list = []
flatten_list = []
for i in range(0, 5):
    dataArray = np.array(dataList[i])
    nperSeg = 100
    window = scipy.signal.get_window('hann', nperSeg)
    freqs, times, spec = scipy.signal.spectrogram(dataArray,fs=1,window=window,nperseg=nperSeg,noverlap=50)
    log_specgram = np.log(spec)
    plt.grid(False)
    
   # Un comment to plot the spectral data 
    #plt.plot(dataArray)
    #plt.xlabel('Wavelength')
    #plt.ylabel('Frequency')
    #plt.plot()

    plt.pcolormesh(times, freqs, log_specgram)
    plt.savefig("/home/yasindu/Desktop/Datasets/soilDelete/soil"+str(i)+".png")
    plt.close()

print("done")