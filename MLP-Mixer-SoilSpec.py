#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import pandas as pd
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score
import glob
import seaborn as sns
import matplotlib.pyplot as plt

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
# Split the Dataset
train_df, test_df = train_test_split(images, train_size=0.7, shuffle=True, random_state=1)

#%%
# Load Images
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
    target_size=(32, 32), # target size was (160,120)
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
    target_size=(32, 32),  # target size was (160,120)
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
    target_size=(32, 32),  # target size was (160,120)
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)


#%%
#MLP with Two Dense Layers
# Multilayer Perceptron with GeLU ( Gaussian Linear Units ) activation
def mlp( x , hidden_dims ):
    y = tf.keras.layers.Dense( hidden_dims )( x )
    y = tf.nn.gelu( y )
    y = tf.keras.layers.Dense( x.shape[ -1 ] )( y )
    y = tf.keras.layers.Dropout( 0.4 )( y )
    return y

#%%
#Mixer layer consisting of token mixing MLPs and channel mixing MLPs
# input shape -> ( batch_size , channels , num_patches )
# output shape -> ( batch_size , channels , num_patches )
def mixer( x , token_mixing_mlp_dims , channel_mixing_mlp_dims ):
    # inputs x of are of shape ( batch_size , num_patches , channels )
    # Note: "channels" is used instead of "embedding_dims"
    
    # Add token mixing MLPs
    token_mixing_out = token_mixing( x , token_mixing_mlp_dims )
    # Shape of token_mixing_out -> ( batch_size , channels , num_patches )

    token_mixing_out = tf.keras.layers.Permute( dims=[ 2 , 1 ] )( token_mixing_out )
    # Shape of transposition -> ( batch_size , num_patches , channels )
    
    #  Add skip connection
    token_mixing_out = tf.keras.layers.Add()( [ x , token_mixing_out ] )

    # Add channel mixing MLPs
    channel_mixing_out = channel_mixing( token_mixing_out , channel_mixing_mlp_dims )
    # Shape of channel_mixing_out -> ( batch_size , num_patches , channels )
    
    # Add skip connection
    channel_mixing_out = tf.keras.layers.Add()( [ channel_mixing_out , token_mixing_out ] )
    # Shape of channel_mixing_out -> ( batch_size , num_patches , channels )

    return channel_mixing_out

#%%
#Token Mixing
# Token Mixing MLPs : Allow communication within patches.
def token_mixing( x , token_mixing_mlp_dims ):
    # x is a tensor of shape ( batch_size , num_patches , channels )
    x = tf.keras.layers.LayerNormalization( epsilon=1e-6 )( x )
    x = tf.keras.layers.Permute( dims=[ 2 , 1 ] )( x ) 
    # After transposition, shape of x -> ( batch_size , channels , num_patches )
    x = mlp( x , token_mixing_mlp_dims )
    return x

#%%
#Channel Mixing
# Channel Mixing MLPs : Allow communication within channels ( features of embeddings )
def channel_mixing( x , channel_mixing_mlp_dims ):
    # x is a tensor of shape ( batch_size , num_patches , channels )
    x = tf.keras.layers.LayerNormalization( epsilon=1e-6 )( x )
    x = mlp( x , channel_mixing_mlp_dims )
    return x

#%%
hidden_dims = 128
token_mixing_mlp_dims = 64
channel_mixing_mlp_dims = 128
patch_size = 9
num_classes = 10
num_mixer_layers = 4
input_image_shape = ( 32 , 32 , 3 )

inputs = tf.keras.layers.Input( shape=input_image_shape )

# Conv2D to extract patches
patches = tf.keras.layers.Conv2D( hidden_dims , kernel_size=patch_size , strides=patch_size )( inputs )
# Resizing the patches
patches_reshape = tf.keras.layers.Reshape( ( patches.shape[ 1 ] * patches.shape[ 2 ] , patches.shape[ 3 ] ) )( patches )

x = patches_reshape
for _ in range( num_mixer_layers ):
    x = mixer( x , token_mixing_mlp_dims , channel_mixing_mlp_dims )

# Classifier head
x = tf.keras.layers.LayerNormalization( epsilon=1e-6 )( x )
x = tf.keras.layers.GlobalAveragePooling1D()( x )
# x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense( 1, activation='linear')( x )

model = tf.keras.models.Model( inputs , outputs )

# The loss function is 'mse', since it is regression
model.compile(
    optimizer='adam',
    loss='mse'
)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=10,
    # callbacks=[
    #     tf.keras.callbacks.EarlyStopping(
    #         monitor='val_loss',
    #         patience=5,
    #         restore_best_weights=True
    #     )
    # ]
)
#model.summary()

#%%
# Results
#output of the model is the prediction
predicted_OC = np.squeeze(model.predict(test_images))

true_OC = test_images.labels

# SQroot of Mean squared error
rmse = np.sqrt(model.evaluate(test_images, verbose=0))
print("     Test RMSE: {:.5f}".format(rmse))


r2 = r2_score(true_OC, predicted_OC)
print("Test R^2 Score: {:.5f}".format(r2))
plt.figure()
sns.scatterplot(x=true_OC, y=predicted_OC, alpha=0.6)
sns.lineplot(true_OC, predicted_OC)
plt.xlabel('Actual OC', fontsize=14)
plt.ylabel('Prediced  OC', fontsize=14)
plt.title(f'Actual vs Predicted  OC (test set) T = {thresh}', fontsize=17)
plt.show()
#%%
his = history.history
plt.figure()
plt.plot(his['loss'])
plt.plot(his['val_loss'])
