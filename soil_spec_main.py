# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read from the CSV file
df = pd.read_csv('/home/yasindu/Downloads/MIR_spec.csv.part')
#%%

val = 1
colNames = list( df[df.columns[pd.Series(df.columns).str.startswith('X')]])


#%% define a zero series
padded = np.zeros((1943,1))
for padCol in range(3600 - 3578):
   colNames.append("padded" + str(padCol))
   df["padded" + str(padCol)] = padded
#%%
subDf = df[colNames]



# def convertToImage(strCol):


#     #print(df.to_string()) 

#     randCol = df.loc[:,strCol]

#     randCol = randCol.to_numpy()

#     colMat = randCol.reshape(29,67)

#     minVal = min(randCol)


#     colMat2 = colMat + abs(minVal)

#     maxVal = colMat2.max()

#     colMat2 = colMat2/ maxVal

#     colMat2 = colMat2 * 255

#     plt.imshow(colMat2,  cmap='gray', vmin=0, vmax=255)
#     # plt.savefig("/home/yasindu/Desktop/Soil_Spec/testImages/soil"+str(val)+".png")
#     return colMat2

# 

    
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

#%%
# for x in range(padded.shape[0]):
#     img = convertToImage(x)
#     val = val + 1

for x in range (padded.shape[0]):
    img = convertToImage(x)
    val = val + 1