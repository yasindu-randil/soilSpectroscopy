#%%
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.signal
import itertools
# Read from the CSV file
df = pd.read_csv('/home/yasindu/Desktop/Soil_Spec/MIR_spec.csv.part')


val = 1
colNames = list( df[df.columns[pd.Series(df.columns).str.startswith('X')]])


#  Plot Wavelength values for each row

df2 = df.iloc[1:, 37:]
dataList = df2.values.tolist()
dataArray = np.array(dataList[0])
#plt.ylabel('Reflectance')
#plt.xlabel('Wavelength')
#plt.plot(dataArray)

#ndArray = scipy.signal.windows.hann(100)
window = scipy.signal.get_window('hann', 128)
#plt.plot(window)
hannTransform = scipy.signal.spectrogram(dataArray,1.0, 'hann')


# Get spectrogram using Hann window
# Then transform that spectrogram to log spectrogram

sample_rate=16000
window_size=20
step_size=10
eps=1e-10
nperseg = int(round(window_size * sample_rate / 1e3))
noverlap = int(round(step_size * sample_rate / 1e3))
freqs, times, spec = scipy.signal.spectrogram(dataArray,fs=1,window='hann',nperseg=100,noverlap=50)
log_specgram = np.log(spec.T.astype(np.float32) + eps)
#plt.plot(log_specgram)
imgLog = log_specgram
plt.imshow(imgLog)


dataList2 = hannTransform[2]
dataList3 = list(itertools.chain.from_iterable(dataList2))
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.plot(dataList2)
#plt.show()
#plt.show(dataList2)

#convert to image

for x in range (22):
    dataArray = np.append(dataArray, [0.0])

minVal = min(dataArray)    
img = np.reshape(dataArray, (60, 60))
img = img + abs(minVal)
maxVal = img.max()
img = img/ maxVal
img = img * 255
#plt.imshow(img)


#convert to image after hann window

for x in range (15):
    dataList3 = np.append(dataList3, [0.0])

minVal = min(dataList3)    
img2 = np.reshape(dataList3, (39, 50))
img2 = img2 + abs(minVal)
maxVal = img2.max()
img2 = img2/ maxVal
img2 = img2 * 255
#plt.imshow(img2)
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