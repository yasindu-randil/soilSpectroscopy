#%%
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.signal
# Read from the CSV file
df = pd.read_csv('/home/yasindu/Desktop/Soil_Spec/MIR_spec.csv.part')


val = 1
colNames = list( df[df.columns[pd.Series(df.columns).str.startswith('X')]])


df2 = df.iloc[1:, 37:]
dataList = df2.values.tolist()

dataArray = np.array(dataList[0])

 
#plt.plot(dataArray)
hannTransform = scipy.signal.spectrogram(dataArray, window=("hann"))

dataList2 = hannTransform[2]
#plt.plot(dataList2)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.show()
#plt.show(dataList2)

#convert to image
#minVal = min(dataList2)

for x in range (22):
    dataArray = np.append(dataArray, [0.0])

minVal = min(dataArray)    
img = np.reshape(dataArray, (60, 60))
img = img + abs(minVal)
maxVal = img.max()
img = img/ maxVal
img = img * 255
plt.imshow(img)
print(df2.head)

#%%

#%% define a zero series
#padded = np.zeros((1943,1))
#for padCol in range(3600 - 3578):
#   colNames.append("padded" + str(padCol))
#   df["padded" + str(padCol)] = padded
#%%
#subDf = df[colNames]
    
#%%
def convertToImage(x):
    row = subDf.iloc[x].to_numpy()
    minVal = min(row)
    img = np.reshape(row, (60, 60))
    img = img + abs(minVal)
    maxVal = img.max()
    img = img/ maxVal
    img = img * 255
    plt.imshow(img,  cmap='gray', vmin=0, vmax=255)
    plt.savefig("/home/yasindu/Desktop/Soil_Spec/testImages/soil"+str(val)+".png")


#for x in range (padded.shape[0]):
#    img = convertToImage(x)
#    val = val + 1