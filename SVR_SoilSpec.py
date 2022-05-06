#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:07:38 2022

@author: yasindu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  #if you want to use seaborn themes with matplotlib functions
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVR
import scipy
import scipy.signal
import itertools
from itertools import chain
import math
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image 
import PIL 
df = pd.read_excel('/home/yasindu/Desktop/Soil_Spec/library_KS.xlsx')

#%%
# Data pre-processing
df2 = df
val = 0

#Remove outliers
thresh = 2
df3= df[df['OC']<thresh]
OC = df3['OC'].tolist()
df3 = df3.iloc[0:, 1:]

#create a list with spectral values without OC
dataList = df3.values.tolist()
dataArray = np.array(dataList[0])

# Get spectrogram using Hann window
# Then transform that spectrogram to log spectrogram


window = scipy.signal.get_window('hann', 100)
sample_rate=16000
window_size=20
step_size=10
eps=1e-10
nperseg = int(round(window_size * sample_rate / 1e3))
noverlap = int(round(step_size * sample_rate / 1e3))

data_list = []
flatten_list = []
for i in range(len(OC)):
    dataArray = np.array(dataList[i])
    freqs, times, spec = scipy.signal.spectrogram(dataArray,fs=1,window=window,nperseg=100,noverlap=50)
    log_specgram = np.log(spec)
    data_list.append(log_specgram)
    flatten_list.append(list(chain.from_iterable(data_list[i])))
    imgLog = log_specgram
    imgLog = np.flipud(imgLog)
    plt.imshow(imgLog)
    plt.savefig("/home/yasindu/Desktop/Datasets/soilimages/soil"+str(val)+".png")
    val = val +1


#sns.set_style('whitegrid')
# Example view of pair plot
#sns.pairplot(df3[df3.columns])

#%%
#Splitting the data
rand_state = 1000
#y = df['OC']
#X = df.drop('OC', axis=1) #Anything but OC
y = OC
X = flatten_list;

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rand_state) # 70-30%


#%%
#SVM Regression
from sklearn.svm import SVR
# Fitting SVM regression to the Training set
SVM_regression = SVR() #by default the kernel is rbf.
SVM_regression.fit(X_train, y_train)


# Predicting the Test set results
y_hat = SVM_regression.predict(X_test)

predictions = pd.DataFrame({ 'y_test':y_test,'y_hat':y_hat})
score_c = r2_score(y_test, y_hat) score_cv = r2_score(y_test, y_hat) # Calculate mean squared error for calibration and cross validation mse_c = mean_squared_error(y, y_c) mse_cv = mean_squared_error(y, y_cv) RMSE_c = math.sqrt(mse_c) RMSE_cv = math.sqrt(mse_cv) rpd_c = y.std()/np.sqrt(mse_c) rpd_cv = y.std()/np.sqrt(mse_cv) 
print(score_c)

#%%
#Evaluation of the model


sns.scatterplot(x=y_test, y=y_hat, alpha=0.6)
sns.lineplot(y_test, y_test)

plt.xlabel('Actual OC', fontsize=14)
plt.ylabel('Prediced  OC', fontsize=14)
plt.title(f'Actual vs Predicted  OC (test set) T = {thresh}', fontsize=17)
plt.show()




