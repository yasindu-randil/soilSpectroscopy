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

from sklearn.metrics import r2_score

import glob

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()  #if you want to use seaborn themes with matplotlib functions
# # Create DataFrame


filepaths = pd.Series(list(glob.iglob('/home/yasindu/Desktop/Datasets/soilimages2/*.png' )), name='Filepath').astype(str)
df = pd.read_csv('/home/yasindu/Desktop/Datasets/VNIR_spec.csv', low_memory=False)
df = df.drop(df.iloc[:, 0:34],axis = 1)
df = df.drop(df.iloc[:, 1:8],axis = 1)
df = df.drop(df.index[4500:])
dataCols =  df['EOC']
thresh = 2
#df2= df[df['EOC']<thresh]

       
OC = pd.Series(df['EOC'].tolist(), name= 'OC')
images = pd.concat([filepaths, OC], axis=1)
images = images.dropna()
images= images[images['OC']<thresh]


#%%
print(images)


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
    target_size=(160, 120),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='OC',
    target_size=(160, 120),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='OC',
    target_size=(160, 120),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)

# # Training
inputs = tf.keras.Input(shape=(160, 120, 3))
# Extract 16 features
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)

x = tf.keras.layers.MaxPool2D()(x)
#After that inpout was down sampled to 79,59

# Second Convolutional layer to extract 32 features from the downsampled image
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)

#Downsample again
x = tf.keras.layers.MaxPool2D()(x) #(38x28)

# Second Convolutional layer to extract 32 features from the downsampled image
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)

#Downsample again
x = tf.keras.layers.MaxPool2D()(x) 

#Get the final 64 features
x = tf.keras.layers.GlobalAveragePooling2D()(x)

#@ Hidden layer neural network
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# The loss function is 'mse', since it is regression
model.compile(
    optimizer='adam',
    loss='mse'
)


history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# # Results
    
#output of the model is the prediction
predicted_OC = np.squeeze(model.predict(test_images))

true_OC = test_images.labels

# SQroot of Mean squared error
rmse = np.sqrt(model.evaluate(test_images, verbose=0))
print("     Test RMSE: {:.5f}".format(rmse))


r2 = r2_score(true_OC, predicted_OC)
print("Test R^2 Score: {:.5f}".format(r2))

sns.scatterplot(x=true_OC, y=predicted_OC, alpha=0.6)
sns.lineplot(true_OC, predicted_OC)

plt.xlabel('Actual OC', fontsize=14)
plt.ylabel('Prediced  OC', fontsize=14)
plt.title(f'Actual vs Predicted  OC (test set) T = {thresh}', fontsize=17)
plt.show()





