
# coding: utf-8

# In[1]:


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


# ## Data input

# In[2]:


#%% Import data and save as dict and list
DatasetPath = r'C:\Users\Zed_Luz\OneDrive - 南方科技大学\BigScaleFiles\1-datasets for bilateral lower limb neuromechanical signals\2-Data'
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
for col in range(colcount):
    dict_data_in[categories[col]] = array_data_in_notrigger_float[:,col]


# ## Select data in

# In[7]:


print(len(categories),'All of Data types:',categories)
selectedchannels = [
                    'Right_Ankle_Velocity', 'Right_Knee_Velocity',
                    'Left_Ankle_Velocity', 'Left_Knee_Velocity']
selectedindex = [categories.index(selectedchannels[i]) 
                    for i in range(len(selectedchannels))]
selecteddatain = array_data_in_notrigger_float[:,selectedindex]
selectedchannelsNum = len(selectedchannels)
# ## Display data

# In[3]:


label = dict_data_in['Mode']
label_prep = to_categorical(label)
exp_dur = rowcount/500 # sample rate: 500Hz
nfeat = selectedchannelsNum
t = np.linspace(0, exp_dur, rowcount)
plt.figure(figsize=(10, 6))
plt.plot(t, label_prep*300)
plt.plot(t, selecteddatain)
plt.show()


# ## First Model-Regression

# In[35]:


lr_model = Sequential()
lr_model.add(Dense(32, input_dim=nfeat, activation='relu'))
lr_model.add(Dense(label_prep.shape[1]))
lr_model.add(Activation('softmax'))
lr_model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
lr_model.fit(selecteddatain, label_prep, validation_split=0.33, epochs=300, batch_size=64)


# # Second Model-Regression

# In[27]:


lr_model1 = Sequential()
lr_model1.add(Dense(32, input_dim=nfeat, activation='relu'))
lr_model1.add(Dense(label_prep.shape[1]))
lr_model1.add(Activation('softmax'))
lr_model1.compile(optimizer='rmsprop',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
lr_model1.fit(array_data_in_notrigger_float, label_prep, validation_split=0.33, epochs=30, batch_size=64)


# ## Third Model-Regression

# In[28]:


lr_model2 = Sequential()
lr_model2.add(Dense(32, input_dim=nfeat, activation='sigm`oid'))
lr_model2.add(Dense(label_prep.shape[1]))
lr_model2.add(Activation('softmax'))
lr_model2.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
lr_model2.fit(array_data_in_notrigger_float, label_prep, validation_split=0.33, epochs=30, batch_size=64)


# ## Plot-1st Model-Regression

# In[31]:


#%%
preds = lr_model.predict(array_data_in_notrigger_float)
labelcategories = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent','Standing']
#print(preds)
plt.figure(figsize=(15, 9))
cmap = plt.get_cmap('jet_r')
labelpreds0, = plt.plot(t, preds[:, 0]*450, label=labelcategories[0], color = cmap(0/len(labelcategories)))
labelpreds1, = plt.plot(t, preds[:, 1]*450, label=labelcategories[1], color = cmap(1/len(labelcategories)))
labelpreds2, = plt.plot(t, preds[:, 2]*450, label=labelcategories[2], color = cmap(2/len(labelcategories)))
labelpreds3, = plt.plot(t, preds[:, 3]*450, label=labelcategories[3], color = cmap(3/len(labelcategories)))
labelpreds4, = plt.plot(t, preds[:, 4]*450, label=labelcategories[4], color = cmap(4/len(labelcategories)))
labelpreds5, = plt.plot(t, preds[:, 5]*450, label=labelcategories[5], color = cmap(5/len(labelcategories)))
labelpreds6, = plt.plot(t, preds[:, 6]*450, label=labelcategories[6], color = cmap(6/len(labelcategories)))
labelgt0, = plt.plot(t, label_prep[:, 0]*(-450),marker='.', label=[labelcategories[0],'-gt'], color = cmap(0/len(labelcategories)))
labelgt1, = plt.plot(t, label_prep[:, 1]*(-450),marker='.', label=[labelcategories[1],'-gt'], color = cmap(1/len(labelcategories)))
labelgt2, = plt.plot(t, label_prep[:, 2]*(-450),marker='.', label=[labelcategories[2],'-gt'], color = cmap(2/len(labelcategories)))
labelgt3, = plt.plot(t, label_prep[:, 3]*(-450),marker='.', label=[labelcategories[3],'-gt'], color = cmap(3/len(labelcategories)))
labelgt4, = plt.plot(t, label_prep[:, 4]*(-450),marker='.', label=[labelcategories[4],'-gt'], color = cmap(4/len(labelcategories)))
labelgt5, = plt.plot(t, label_prep[:, 5]*(-450),marker='.', label=[labelcategories[5],'-gt'], color = cmap(5/len(labelcategories)))
labelgt6, = plt.plot(t, label_prep[:, 6]*(-450),marker='.', label=[labelcategories[6],'-gt'], color = cmap(6/len(labelcategories)))

firstlegend = plt.legend(handles=[labelpreds0,labelpreds1,labelpreds2,labelpreds3,labelpreds4,labelpreds5,labelpreds6],
                         bbox_to_anchor=(0.5,0.7),ncol=3)
ax = plt.gca().add_artist(firstlegend)
secondlegend = plt.legend(handles=[labelgt0,labelgt1, labelgt2, labelgt3, labelgt4, labelgt5, labelgt6],
                          bbox_to_anchor=(0.65,0.2),ncol=3)
ax = plt.gca().add_artist(secondlegend)
#labelpreds, = plt.plot(t, label_prep*4)
data1, = plt.plot(t, array_data_in_notrigger_float[:,-2-1], label=categories[-2-1-8])
#plt.plot(t, array_data_in_notrigger_float[:,1], label=categories[1])
#plt.plot(t, array_data_in_notrigger_float[:,2], label=categories[2])
plt.legend(handles=[data1],bbox_to_anchor=(0,0.6,1, 0.4) )
#plt.xlim([26, 46])
#plt.ylim([-50, 50])
plt.ylabel('Features')
plt.xlabel('Time (s)')
plt.title('Locomotion Mode Prediction Using Logistic Regression')
plt.grid()
plt.show()


# ## Plot-2nd Model-Regression

# In[15]:


#%%
preds = lr_model1.predict(array_data_in_notrigger_float)
labelcategories = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent','Standing']
#print(preds)-Regression
plt.figure(figsize=(15, 9))
cmap = plt.get_cmap('jet_r')
labelpreds0, = plt.plot(t, preds[:, 0]*450, label=labelcategories[0], color = cmap(0/len(labelcategories)))
labelpreds1, = plt.plot(t, preds[:, 1]*450, label=labelcategories[1], color = cmap(1/len(labelcategories)))
labelpreds2, = plt.plot(t, preds[:, 2]*450, label=labelcategories[2], color = cmap(2/len(labelcategories)))
labelpreds3, = plt.plot(t, preds[:, 3]*450, label=labelcategories[3], color = cmap(3/len(labelcategories)))
labelpreds4, = plt.plot(t, preds[:, 4]*450, label=labelcategories[4], color = cmap(4/len(labelcategories)))
labelpreds5, = plt.plot(t, preds[:, 5]*450, label=labelcategories[5], color = cmap(5/len(labelcategories)))
labelpreds6, = plt.plot(t, preds[:, 6]*450, label=labelcategories[6], color = cmap(6/len(labelcategories)))
labelgt0, = plt.plot(t, label_prep[:, 0]*(-450),marker='.', label=[labelcategories[0],'-gt'], color = cmap(0/len(labelcategories)))
labelgt1, = plt.plot(t, label_prep[:, 1]*(-450),marker='.', label=[labelcategories[1],'-gt'], color = cmap(1/len(labelcategories)))
labelgt2, = plt.plot(t, label_prep[:, 2]*(-450),marker='.', label=[labelcategories[2],'-gt'], color = cmap(2/len(labelcategories)))
labelgt3, = plt.plot(t, label_prep[:, 3]*(-450),marker='.', label=[labelcategories[3],'-gt'], color = cmap(3/len(labelcategories)))
labelgt4, = plt.plot(t, label_prep[:, 4]*(-450),marker='.', label=[labelcategories[4],'-gt'], color = cmap(4/len(labelcategories)))
labelgt5, = plt.plot(t, label_prep[:, 5]*(-450),marker='.', label=[labelcategories[5],'-gt'], color = cmap(5/len(labelcategories)))
labelgt6, = plt.plot(t, label_prep[:, 6]*(-450),marker='.', label=[labelcategories[6],'-gt'], color = cmap(6/len(labelcategories)))

firstlegend = plt.legend(handles=[labelpreds0,labelpreds1,labelpreds2,labelpreds3,labelpreds4,labelpreds5,labelpreds6],
                         bbox_to_anchor=(0.5,0.7),ncol=3)
ax = plt.gca().add_artist(firstlegend)
secondlegend = plt.legend(handles=[labelgt0,labelgt1, labelgt2, labelgt3, labelgt4, labelgt5, labelgt6],
                          bbox_to_anchor=(0.65,0.2),ncol=3)
ax = plt.gca().add_artist(secondlegend)
#labelpreds, = plt.plot(t, label_prep*4)
data1, = plt.plot(t, array_data_in_notrigger_float[:,-2-1], label=categories[-2-1-8])
#plt.plot(t, array_data_in_notrigger_float[:,1], label=categories[1])
#plt.plot(t, array_data_in_notrigger_float[:,2], label=categories[2])
plt.legend(handles=[data1],bbox_to_anchor=(0,0.6,1, 0.4) )
#plt.xlim([26, 46])
#plt.ylim([-50, 50])
plt.ylabel('Features')
plt.xlabel('Time (s)')
plt.title('Locomotion Mode Prediction Using Logistic Regression')
plt.grid()
plt.show()


# ## Plot-3rd Model-Regression

# In[20]:


#%%
preds = lr_model2.predict(array_data_in_notrigger_float)
labelcategories = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent','Standing']
#print(preds)
plt.figure(figsize=(15, 9))
cmap = plt.get_cmap('jet_r')
labelpreds0, = plt.plot(t, preds[:, 0]*450, label=labelcategories[0], color = cmap(0/len(labelcategories)))
labelpreds1, = plt.plot(t, preds[:, 1]*450, label=labelcategories[1], color = cmap(1/len(labelcategories)))
labelpreds2, = plt.plot(t, preds[:, 2]*450, label=labelcategories[2], color = cmap(2/len(labelcategories)))
labelpreds3, = plt.plot(t, preds[:, 3]*450, label=labelcategories[3], color = cmap(3/len(labelcategories)))
labelpreds4, = plt.plot(t, preds[:, 4]*450, label=labelcategories[4], color = cmap(4/len(labelcategories)))
labelpreds5, = plt.plot(t, preds[:, 5]*450, label=labelcategories[5], color = cmap(5/len(labelcategories)))
labelpreds6, = plt.plot(t, preds[:, 6]*450, label=labelcategories[6], color = cmap(6/len(labelcategories)))
labelgt0, = plt.plot(t, label_prep[:, 0]*(-450),marker='.', label=[labelcategories[0],'-gt'], color = cmap(0/len(labelcategories)))
labelgt1, = plt.plot(t, label_prep[:, 1]*(-450),marker='.', label=[labelcategories[1],'-gt'], color = cmap(1/len(labelcategories)))
labelgt2, = plt.plot(t, label_prep[:, 2]*(-450),marker='.', label=[labelcategories[2],'-gt'], color = cmap(2/len(labelcategories)))
labelgt3, = plt.plot(t, label_prep[:, 3]*(-450),marker='.', label=[labelcategories[3],'-gt'], color = cmap(3/len(labelcategories)))
labelgt4, = plt.plot(t, label_prep[:, 4]*(-450),marker='.', label=[labelcategories[4],'-gt'], color = cmap(4/len(labelcategories)))
labelgt5, = plt.plot(t, label_prep[:, 5]*(-450),marker='.', label=[labelcategories[5],'-gt'], color = cmap(5/len(labelcategories)))
labelgt6, = plt.plot(t, label_prep[:, 6]*(-450),marker='.', label=[labelcategories[6],'-gt'], color = cmap(6/len(labelcategories)))

firstlegend = plt.legend(handles=[labelpreds0,labelpreds1,labelpreds2,labelpreds3,labelpreds4,labelpreds5,labelpreds6],
                         bbox_to_anchor=(0.5,0.7),ncol=3)
ax = plt.gca().add_artist(firstlegend)
secondlegend = plt.legend(handles=[labelgt0,labelgt1, labelgt2, labelgt3, labelgt4, labelgt5, labelgt6],
                          bbox_to_anchor=(0.65,0.2),ncol=3)
ax = plt.gca().add_artist(secondlegend)
#labelpreds, = plt.plot(t, label_prep*4)
data1, = plt.plot(t, array_data_in_notrigger_float[:,-2-1], label=categories[-2-1-8])
#plt.plot(t, array_data_in_notrigger_float[:,1], label=categories[1])
#plt.plot(t, array_data_in_notrigger_float[:,2], label=categories[2])
plt.legend(handles=[data1],bbox_to_anchor=(0,0.6,1, 0.4) )
#plt.xlim([26, 46])
#plt.ylim([-50, 50])
plt.ylabel('Features')
plt.xlabel('Time (s)')
plt.title('Locomotion Mode Prediction Using Logistic Regression')
plt.grid()
plt.show()


# In[29]:


import os
os.getcwd()

