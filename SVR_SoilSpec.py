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


df = pd.read_excel('/home/yasindu/Desktop/Soil_Spec/library_KS.xlsx')

#%%
# Data pre-processing
df2 = df
df3 = df2.iloc[0:, 0:5]
sns.set_style('whitegrid')
# Example view of pair plot
sns.pairplot(df3[df3.columns])
#%%
thresh = 40
df4 = df[df['OC']<thresh]
df = df4
#%%
#Splitting the data
rand_state = 1000
y = df['OC']
X = df.drop('OC', axis=1) #Anything but OC

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
print(predictions.head())

#%%
#Evaluation of the model


sns.scatterplot(x=y_test, y=y_hat, alpha=0.6)
sns.lineplot(y_test, y_test)

plt.xlabel('Actual OC', fontsize=14)
plt.ylabel('Prediced  OC', fontsize=14)
plt.title(f'Actual vs Predicted  OC (test set) T = {thresh}', fontsize=17)
plt.show()




