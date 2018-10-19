# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:36:26 2018

@author: Zed_Luz
"""
#%%
#-*- coding:utf-8 -*-
import numpy as np
from scipy.io import loadmat, savemat
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten
from keras.layers import SimpleRNN, LSTM, GRU, TimeDistributed, BatchNormalization
from keras.losses import categorical_crossentropy
from keras import regularizers
import keras.backend as K
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence, to_categorical, plot_model
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
import csv
#from imu_preprocess_utils import *
#import tsne as tsne

#%% Import data and save as dict and list
OnlineDatapath = r'C:\Users\Zed_Luz\OneDrive - 南方科技大学\BigScaleFiles\1-datasets for bilateral lower limb neuromechanical signals\2-Data\AB185\Processed\AB185_Circuit_001_post.csv'
data_in = list()
with open(OnlineDatapath) as csvfile:
   spamreader = csv.reader(csvfile)
   for i, row in enumerate(spamreader):
       if i == 0:
           categories = row
       else:
           data_in.append(row)

data_in = np.array(data_in)
data_in_trigger = [row[-8:] for row in data_in] # only trigger 
data_in_notrigger = [row[:-8] for row in data_in] # without trigger 
data_in_notrigger_float = list()
data_in_trigger_float = list()
for i in range(0,len(data_in_notrigger)):
    buff = np.float32(data_in_notrigger[i])
    data_in_notrigger_float.append(buff)
## transfer trigger data into float
#for i in range(0,len(data_in_trigger)):
#    buff = np.float32(data_in_trigger[i])
#    data_in_trigger_float.append(buff)
array_data_in_notrigger_float = np.array(data_in_notrigger_float)
rowcount = array_data_in_notrigger_float.shape[0]
colcount = array_data_in_notrigger_float.shape[1]
dict_data_in = dict()
for col in range(0,colcount):
    dict_data_in[categories[col]] = array_data_in_notrigger_float[:,col]
#%%
label = dict_data_in['Mode']
label_prep = to_categorical(label)
exp_dur = rowcount/500 # sample rate: 500Hz
nfeat = array_data_in_notrigger_float.shape[1]
t = np.linspace(0, exp_dur, rowcount)
plt.figure(figsize=(10, 6))
#plt.plot(t, label)
plt.plot(t, dict_data_in[categories[0]])
plt.plot(t, label_prep*300)
plt.plot(t, array_data_in_notrigger_float[:,0:6])
plt.show()
#%%
lr_model = Sequential()
lr_model.add(Dense(32, input_dim=nfeat, activation='relu'))
lr_model.add(Dense(7))
lr_model.add(Activation('softmax'))
lr_model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
lr_model.fit(array_data_in_notrigger_float, label_prep,validation_split=0.33, epochs=300, batch_size=64)
#%%
preds = lr_model.predict(array_data_in_notrigger_float)
#print(preds)
plt.figure(figsize=(15, 9))
#plt.plot(t, X[:, 56], label='Right Foot Tilt')
#plt.plot(t, X[:, 57], label='Right Shank Tilt')
#plt.plot(t, X[:, 58], label='Right Thigh Tilt')
plt.plot(t, preds[:, 0]*4, label='Sitting')
plt.plot(t, preds[:, 1]*4, label='Level Ground Walking')
plt.plot(t, preds[:, 2]*4, label='Ramp Ascent')
plt.plot(t, preds[:, 3]*4, label='Ramp Descent')
plt.plot(t, preds[:, 4]*4, label='Stair Ascent')
plt.plot(t, preds[:, 5]*4, label='Stair Descent')
plt.plot(t, preds[:, 6]*4, label='Standing')
#plt.plot(t, label_prep*4)
plt.plot(t, array_data_in_notrigger_float[:,0], label=categories[0])
plt.plot(t, array_data_in_notrigger_float[:,1], label=categories[1])
plt.plot(t, array_data_in_notrigger_float[:,2], label=categories[2])
#plt.xlim([26, 46])
#plt.ylim([-50, 50])
plt.ylabel('Features')
plt.xlabel('Time (s)')
plt.title('Locomotion Mode Prediction Using Logistic Regression')
plt.grid()
plt.legend()
plt.show()