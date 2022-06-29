#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 08:10:40 2022

@author: yasindu
"""
#%%

import numpy as np
import pandas as pd
from pathlib import Path
import os.path

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import r2_score

import glob

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()  #if you want to use seaborn themes with matplotlib functions
#%% Create DataFrame


filepaths = pd.Series(list(glob.iglob('/home/yasindu/Desktop/Datasets/resize/*.png' )), name='Filepath').astype(str)
df = pd.read_csv('/home/yasindu/Desktop/Datasets/VNIR_spec.csv', low_memory=False)
#%%
df = df.drop(df.iloc[:, 0:34],axis = 1)
df = df.drop(df.iloc[:, 1:8],axis = 1)
df = df.drop(df.index[10000:])
dataCols =  df['EOC']
thresh = 2
#df2= df[df['EOC']<thresh]

       
OC = pd.Series(df['EOC'].tolist(), name= 'OC')
images = pd.concat([filepaths, OC], axis=1)
images = images.dropna()
images= images[images['OC']<thresh]


#%%
#print(images)


# # Split the Dataset


train_df, test_df = train_test_split(images, train_size=0.7, shuffle=True, random_state=1)


# # Load Images
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='OC',
    target_size=(32, 32),
    color_mode='rgb',
    class_mode='raw',
    batch_size=512,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='OC',
    target_size=(32, 32),
    color_mode='rgb',
    class_mode='raw',
    batch_size=512,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='OC',
    target_size=(32, 32),
    color_mode='rgb',
    class_mode='raw',
    batch_size=64,
    shuffle=False
)


#%% Training
inputs = tf.keras.Input(shape=(32, 32, 3))
# Extract 16 features
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)



x = tf.keras.layers.MaxPool2D()(x)
#After that inpout was down sampled to 79,59

# Second Convolutional layer to extract 32 features from the downsampled image
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)

#Downsample again
x = tf.keras.layers.MaxPool2D()(x) #(38x28)


#Get the final 64 features
test = tf.keras.layers.GlobalAveragePooling2D()(x)

#@ Hidden layer neural network
x = tf.keras.layers.Dense(64, activation='relu')(test)
x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)


# model = models.Sequential()
# model.add(layers.Dense(16, input_shape=(126,126,3)))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(dropout_rate))

# model.add(layers.Dense(32, activation = 'relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(dropout_rate))

# model.add(layers.Dense(1, activation='relu'))

# The loss function is 'mse', since it is regression
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(
    optimizer=optimizer,
    loss='mse'
)


history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# # Results

#%%    
#output of the model is the prediction
# Results
#output of the model is the prediction
predicted_OC = np.squeeze(model.predict(test_images))

true_OC = test_images.labels
df10 = pd.DataFrame({'true_OC' : true_OC, 'predicted_OC' : predicted_OC})   
df10.to_csv('/home/yasindu/Desktop/Datasets/predictions.csv', index=False, encoding='utf-8')
# SQroot of Mean squared error
rmse = np.sqrt(model.evaluate(test_images, verbose=0))
print("     Test RMSE: {:.5f}".format(rmse))


r2 = r2_score(true_OC, predicted_OC)
print("Test R^2 Score: {:.5f}".format(r2))
#plt.figure()
#sns.scatterplot(x=true_OC, y=predicted_OC, alpha=0.6)
#sns.lineplot(true_OC, predicted_OC)
#plt.xlabel('Actual OC', fontsize=14)
#plt.ylabel('Prediced  OC', fontsize=14)
#plt.title(f'Actual vs Predicted  OC (test set) T = {thresh}', fontsize=17)
#plt.show()

print(predicted_OC.max()-predicted_OC.min())
#sns.regplot(x=true_OC, y=predicted_OC)

true_OC_max = true_OC.max()
true_OC_min = true_OC.min()

ax = sns.scatterplot(x=predicted_OC, y=true_OC)
ax.set(ylim=(true_OC_min, true_OC_max))
ax.set(xlim=(true_OC_min, true_OC_max))
ax.set_xlabel("Predicted value of OC")
ax.set_ylabel("Observed value of OC")

X_ref = Y_ref = np.linspace(true_OC_min, true_OC_max, 100)
plt.plot(X_ref, Y_ref, color='red', linewidth=1)
plt.show()

#%%
his = history.history
plt.figure()
plt.plot(his['loss'])
plt.plot(his['val_loss'])

