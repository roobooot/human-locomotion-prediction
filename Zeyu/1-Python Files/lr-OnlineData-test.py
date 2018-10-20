
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
import os
#from imu_preprocess_utils import *
#import tsne as tsne


#%% Data input
def readdata (datapath):#read data from csv and store in array with type of float
    data_in = list()
    with open(datapath) as csvfile:
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
    label = dict_data_in['Mode']
    label_prep = to_categorical(label)
    return array_data_in_notrigger_float, dict_data_in, rowcount, colcount, categories, label_prep
# In[2]:


#%% Import data and save as dict and list
DatasetPath = r'C:\Users\Zed_Luz\OneDrive - 南方科技大学\BigScaleFiles\1-datasets for bilateral lower limb neuromechanical signals\2-Data'
#OnlineDatapath = r'C:\Users\Zed_Luz\OneDrive - 南方科技大学\BigScaleFiles\1-datasets for bilateral lower limb neuromechanical signals\2-Data\AB185\Processed\AB185_Circuit_001_post.csv'
OnlineDatapath = dict()
Subjects = []# ['AB185', 'AB186',...]
for (dirpath, dirnames, filenames) in os.walk(DatasetPath):
    Subjects.extend(dirnames)
    break
DataType = [] # ['Features', 'MVC', 'Processed', 'Raw']
for (dirpath, dirnames, filenames) in os.walk(os.path.join(DatasetPath,Subjects[0])):
    DataType.extend(dirnames)
    break
DataFileName = [] # ['Features', 'MVC', 'Processed', 'Raw']
for (dirpath, dirnames, filenames) in os.walk(os.path.join(DatasetPath,Subjects[0],DataType[2])):
    DataFileName.extend(filenames)
    break
data1 = os.path.join(DatasetPath,Subjects[0],DataType[2], DataFileName[0])
data2 = os.path.join(DatasetPath,Subjects[0],DataType[2], DataFileName[2])
array_data1, dict_data1, rowcount1, colcount1, categories1, label_prep1 = readdata(data1)
array_data2, dict_data2, rowcount2, colcount2, categories2, label_prep2 = readdata(data2)
Train_data = array_data1
Val_data = array_data2
#%% Select specific channels of data 

print(len(categories1),'All of Data types:',categories1)
selectedchannels = ['Right_Shank_Ax', 'Right_Shank_Ay', 'Right_Shank_Az', 'Right_Shank_Gy', 'Right_Shank_Gz', 'Right_Shank_Gx', 'Right_Thigh_Ax', 'Right_Thigh_Ay', 'Right_Thigh_Az', 'Right_Thigh_Gy', 'Right_Thigh_Gz', 'Right_Thigh_Gx', 'Left_Shank_Ax', 'Left_Shank_Ay', 'Left_Shank_Az', 'Left_Shank_Gy', 'Left_Shank_Gz', 'Left_Shank_Gx', 'Left_Thigh_Ax', 'Left_Thigh_Ay', 'Left_Thigh_Az', 'Left_Thigh_Gy', 'Left_Thigh_Gz', 'Left_Thigh_Gx', 'Waist_Ax', 'Waist_Ay', 'Waist_Az', 'Waist_Gy', 'Waist_Gz', 'Waist_Gx', 'Right_TA', 'Right_MG', 'Right_SOL', 'Right_BF', 'Right_ST', 'Right_VL', 'Right_RF', 'Left_TA', 'Left_MG', 'Left_SOL', 'Left_BF', 'Left_ST', 'Left_VL', 'Left_RF', 'Right_Ankle', 'Right_Knee', 'Left_Ankle', 'Left_Knee', 'Right_Ankle_Velocity', 'Right_Knee_Velocity', 'Left_Ankle_Velocity', 'Left_Knee_Velocity']
selectedindex = [categories1.index(selectedchannels[i]) 
                    for i in range(len(selectedchannels))]
selectedchannelsNum = len(selectedchannels)
selecteddatain_train = Train_data[:,selectedindex]
selecteddatain_val = Val_data[:,selectedindex]
label_train = dict_data1['Mode']
label_train_prep = to_categorical(label_train)
label_val = dict_data2['Mode']
label_val_prep = to_categorical(label_val)
#%% Display train data
exp_dur_train = rowcount1/500 # sample rate: 500Hz
exp_dur_val = rowcount2/500 # sample rate: 500Hz
nfeat = selectedchannelsNum
t_train = np.linspace(0, exp_dur_train, rowcount1)
t_val = np.linspace(0, exp_dur_val, rowcount2)
plt.figure(figsize=(10, 6))
plt.plot(t_train, label_train_prep*300)
plt.plot(t_train, selecteddatain_train)
plt.show()
# ## First Model-Regression

# In[35]:


lr_model = Sequential()
lr_model.add(Dense(32, input_dim=nfeat, activation='relu'))
lr_model.add(Dense(label_train_prep.shape[1]))
lr_model.add(Activation('softmax'))
lr_model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
lr_model.fit(selecteddatain_train, label_train_prep, epochs=30, batch_size=64)
score = lr_model.evaluate(selecteddatain_val, label_val_prep, batch_size=64)

# # Second Model-Regression

# In[27]:


# =============================================================================
# lr_model1 = Sequential()
# lr_model1.add(Dense(32, input_dim=nfeat, activation='relu'))
# lr_model1.add(Dense(label_prep.shape[1]))
# lr_model1.add(Activation('softmax'))
# lr_model1.compile(optimizer='rmsprop',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
# lr_model1.fit(array_data_in_notrigger_float, label_prep, validation_split=0.33, epochs=30, batch_size=64)
# 
# =============================================================================

# ## Third Model-Regression

# In[28]:


# =============================================================================
# lr_model2 = Sequential()
# lr_model2.add(Dense(32, input_dim=nfeat, activation='sigm`oid'))
# lr_model2.add(Dense(label_prep.shape[1]))
# lr_model2.add(Activation('softmax'))
# lr_model2.compile(optimizer='adam',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
# lr_model2.fit(array_data_in_notrigger_float, label_prep, validation_split=0.33, epochs=30, batch_size=64)
# 
# =============================================================================

# ## Plot-1st Model-Regression

# In[31]:


#%%
preds = lr_model.predict(selecteddatain_val)
labelcategories = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent','Standing']
#print(preds)
plt.figure(figsize=(15, 9))
cmap = plt.get_cmap('jet_r')
labelpreds0, = plt.plot(t_val, preds[:, 0]*450, label=labelcategories[0], color = cmap(0/len(labelcategories)))
labelpreds1, = plt.plot(t_val, preds[:, 1]*450, label=labelcategories[1], color = cmap(1/len(labelcategories)))
labelpreds2, = plt.plot(t_val, preds[:, 2]*450, label=labelcategories[2], color = cmap(2/len(labelcategories)))
labelpreds3, = plt.plot(t_val, preds[:, 3]*450, label=labelcategories[3], color = cmap(3/len(labelcategories)))
labelpreds4, = plt.plot(t_val, preds[:, 4]*450, label=labelcategories[4], color = cmap(4/len(labelcategories)))
labelpreds5, = plt.plot(t_val, preds[:, 5]*450, label=labelcategories[5], color = cmap(5/len(labelcategories)))
labelpreds6, = plt.plot(t_val, preds[:, 6]*450, label=labelcategories[6], color = cmap(6/len(labelcategories)))
labelgt0, = plt.plot(t_val, label_val_prep[:, 0]*(-450),marker='.', label=[labelcategories[0],'-gt'], color = cmap(0/len(labelcategories)))
labelgt1, = plt.plot(t_val, label_val_prep[:, 1]*(-450),marker='.', label=[labelcategories[1],'-gt'], color = cmap(1/len(labelcategories)))
labelgt2, = plt.plot(t_val, label_val_prep[:, 2]*(-450),marker='.', label=[labelcategories[2],'-gt'], color = cmap(2/len(labelcategories)))
labelgt3, = plt.plot(t_val, label_val_prep[:, 3]*(-450),marker='.', label=[labelcategories[3],'-gt'], color = cmap(3/len(labelcategories)))
labelgt4, = plt.plot(t_val, label_val_prep[:, 4]*(-450),marker='.', label=[labelcategories[4],'-gt'], color = cmap(4/len(labelcategories)))
labelgt5, = plt.plot(t_val, label_val_prep[:, 5]*(-450),marker='.', label=[labelcategories[5],'-gt'], color = cmap(5/len(labelcategories)))
labelgt6, = plt.plot(t_val, label_val_prep[:, 6]*(-450),marker='.', label=[labelcategories[6],'-gt'], color = cmap(6/len(labelcategories)))

firstlegend = plt.legend(handles=[labelpreds0,labelpreds1,labelpreds2,labelpreds3,labelpreds4,labelpreds5,labelpreds6],
                         bbox_to_anchor=(0.5,0.7),ncol=3)
ax = plt.gca().add_artist(firstlegend)
secondlegend = plt.legend(handles=[labelgt0,labelgt1, labelgt2, labelgt3, labelgt4, labelgt5, labelgt6],
                          bbox_to_anchor=(0.65,0.2),ncol=3)
ax = plt.gca().add_artist(secondlegend)
#labelpreds, = plt.plot(t, label_prep*4)
data1, = plt.plot(t_val, selecteddatain_val[:,-2-1], label=categories1[-2-1-8])
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
# =============================================================================
# preds = lr_model1.predict(array_data_in_notrigger_float)
# labelcategories = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent','Standing']
# #print(preds)-Regression
# plt.figure(figsize=(15, 9))
# cmap = plt.get_cmap('jet_r')
# labelpreds0, = plt.plot(t, preds[:, 0]*450, label=labelcategories[0], color = cmap(0/len(labelcategories)))
# labelpreds1, = plt.plot(t, preds[:, 1]*450, label=labelcategories[1], color = cmap(1/len(labelcategories)))
# labelpreds2, = plt.plot(t, preds[:, 2]*450, label=labelcategories[2], color = cmap(2/len(labelcategories)))
# labelpreds3, = plt.plot(t, preds[:, 3]*450, label=labelcategories[3], color = cmap(3/len(labelcategories)))
# labelpreds4, = plt.plot(t, preds[:, 4]*450, label=labelcategories[4], color = cmap(4/len(labelcategories)))
# labelpreds5, = plt.plot(t, preds[:, 5]*450, label=labelcategories[5], color = cmap(5/len(labelcategories)))
# labelpreds6, = plt.plot(t, preds[:, 6]*450, label=labelcategories[6], color = cmap(6/len(labelcategories)))
# labelgt0, = plt.plot(t, label_prep[:, 0]*(-450),marker='.', label=[labelcategories[0],'-gt'], color = cmap(0/len(labelcategories)))
# labelgt1, = plt.plot(t, label_prep[:, 1]*(-450),marker='.', label=[labelcategories[1],'-gt'], color = cmap(1/len(labelcategories)))
# labelgt2, = plt.plot(t, label_prep[:, 2]*(-450),marker='.', label=[labelcategories[2],'-gt'], color = cmap(2/len(labelcategories)))
# labelgt3, = plt.plot(t, label_prep[:, 3]*(-450),marker='.', label=[labelcategories[3],'-gt'], color = cmap(3/len(labelcategories)))
# labelgt4, = plt.plot(t, label_prep[:, 4]*(-450),marker='.', label=[labelcategories[4],'-gt'], color = cmap(4/len(labelcategories)))
# labelgt5, = plt.plot(t, label_prep[:, 5]*(-450),marker='.', label=[labelcategories[5],'-gt'], color = cmap(5/len(labelcategories)))
# labelgt6, = plt.plot(t, label_prep[:, 6]*(-450),marker='.', label=[labelcategories[6],'-gt'], color = cmap(6/len(labelcategories)))
# 
# firstlegend = plt.legend(handles=[labelpreds0,labelpreds1,labelpreds2,labelpreds3,labelpreds4,labelpreds5,labelpreds6],
#                          bbox_to_anchor=(0.5,0.7),ncol=3)
# ax = plt.gca().add_artist(firstlegend)
# secondlegend = plt.legend(handles=[labelgt0,labelgt1, labelgt2, labelgt3, labelgt4, labelgt5, labelgt6],
#                           bbox_to_anchor=(0.65,0.2),ncol=3)
# ax = plt.gca().add_artist(secondlegend)
# #labelpreds, = plt.plot(t, label_prep*4)
# data1, = plt.plot(t, array_data_in_notrigger_float[:,-2-1], label=categories[-2-1-8])
# #plt.plot(t, array_data_in_notrigger_float[:,1], label=categories[1])
# #plt.plot(t, array_data_in_notrigger_float[:,2], label=categories[2])
# plt.legend(handles=[data1],bbox_to_anchor=(0,0.6,1, 0.4) )
# #plt.xlim([26, 46])
# #plt.ylim([-50, 50])
# plt.ylabel('Features')
# plt.xlabel('Time (s)')
# plt.title('Locomotion Mode Prediction Using Logistic Regression')
# plt.grid()
# plt.show()
# 
# =============================================================================

# ## Plot-3rd Model-Regression

# In[20]:


#%%
# =============================================================================
# preds = lr_model2.predict(array_data_in_notrigger_float)
# labelcategories = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent','Standing']
# #print(preds)
# plt.figure(figsize=(15, 9))
# cmap = plt.get_cmap('jet_r')
# labelpreds0, = plt.plot(t, preds[:, 0]*450, label=labelcategories[0], color = cmap(0/len(labelcategories)))
# labelpreds1, = plt.plot(t, preds[:, 1]*450, label=labelcategories[1], color = cmap(1/len(labelcategories)))
# labelpreds2, = plt.plot(t, preds[:, 2]*450, label=labelcategories[2], color = cmap(2/len(labelcategories)))
# labelpreds3, = plt.plot(t, preds[:, 3]*450, label=labelcategories[3], color = cmap(3/len(labelcategories)))
# labelpreds4, = plt.plot(t, preds[:, 4]*450, label=labelcategories[4], color = cmap(4/len(labelcategories)))
# labelpreds5, = plt.plot(t, preds[:, 5]*450, label=labelcategories[5], color = cmap(5/len(labelcategories)))
# labelpreds6, = plt.plot(t, preds[:, 6]*450, label=labelcategories[6], color = cmap(6/len(labelcategories)))
# labelgt0, = plt.plot(t, label_prep[:, 0]*(-450),marker='.', label=[labelcategories[0],'-gt'], color = cmap(0/len(labelcategories)))
# labelgt1, = plt.plot(t, label_prep[:, 1]*(-450),marker='.', label=[labelcategories[1],'-gt'], color = cmap(1/len(labelcategories)))
# labelgt2, = plt.plot(t, label_prep[:, 2]*(-450),marker='.', label=[labelcategories[2],'-gt'], color = cmap(2/len(labelcategories)))
# labelgt3, = plt.plot(t, label_prep[:, 3]*(-450),marker='.', label=[labelcategories[3],'-gt'], color = cmap(3/len(labelcategories)))
# labelgt4, = plt.plot(t, label_prep[:, 4]*(-450),marker='.', label=[labelcategories[4],'-gt'], color = cmap(4/len(labelcategories)))
# labelgt5, = plt.plot(t, label_prep[:, 5]*(-450),marker='.', label=[labelcategories[5],'-gt'], color = cmap(5/len(labelcategories)))
# labelgt6, = plt.plot(t, label_prep[:, 6]*(-450),marker='.', label=[labelcategories[6],'-gt'], color = cmap(6/len(labelcategories)))
# 
# firstlegend = plt.legend(handles=[labelpreds0,labelpreds1,labelpreds2,labelpreds3,labelpreds4,labelpreds5,labelpreds6],
#                          bbox_to_anchor=(0.5,0.7),ncol=3)
# ax = plt.gca().add_artist(firstlegend)
# secondlegend = plt.legend(handles=[labelgt0,labelgt1, labelgt2, labelgt3, labelgt4, labelgt5, labelgt6],
#                           bbox_to_anchor=(0.65,0.2),ncol=3)
# ax = plt.gca().add_artist(secondlegend)
# #labelpreds, = plt.plot(t, label_prep*4)
# data1, = plt.plot(t, array_data_in_notrigger_float[:,-2-1], label=categories[-2-1-8])
# #plt.plot(t, array_data_in_notrigger_float[:,1], label=categories[1])
# #plt.plot(t, array_data_in_notrigger_float[:,2], label=categories[2])
# plt.legend(handles=[data1],bbox_to_anchor=(0,0.6,1, 0.4) )
# #plt.xlim([26, 46])
# #plt.ylim([-50, 50])
# plt.ylabel('Features')
# plt.xlabel('Time (s)')
# plt.title('Locomotion Mode Prediction Using Logistic Regression')
# plt.grid()
# plt.show()
# 
# =============================================================================

#%% CNNs
def get_sub_sequences(data_array, y_array, window_size=120, step_size=90, dims=None, seq_out=False, causal=True):
    rows = data_array.shape[0]
    cols = data_array.shape[1]

    if dims == None:
        outdims = [i for i in range(cols)]
    else:
        outdims = dims

    sequences = rows//step_size
    out_x = np.zeros((sequences, window_size, len(outdims)))
    if seq_out:
        out_y = np.zeros((sequences, window_size, y_array.shape[1]))
    else:
        out_y = np.zeros((sequences, y_array.shape[1]))

    idxs = range(window_size, rows, step_size)    

    for i, j in enumerate(idxs):
        out_x[i, :, :] = data_array[j-window_size:j, outdims]
        if seq_out:
            out_y[i, :, :] = y_array[j-window_size:j, :]
        else:
            out_y[i, :] = y_array[j, :]

    return out_x, out_y
data_seq_train, label_seq_train = get_sub_sequences(selecteddatain_train, label_train_prep, window_size=100, step_size=2)
data_seq_train = np.reshape(data_seq_train, newshape=(data_seq_train.shape[0], data_seq_train.shape[1], data_seq_train.shape[2], 1))
data_seq_val, label_seq_val = get_sub_sequences(selecteddatain_val, label_val_prep, window_size=100, step_size=2)
data_seq_val = np.reshape(data_seq_val, newshape=(data_seq_val.shape[0], data_seq_val.shape[1], data_seq_val.shape[2], 1))
cnn_model1 = Sequential()
cnn_model1.add(Conv2D(filters=16, kernel_size=9, input_shape=(100, data_seq_train.shape[2], 1),
                      data_format='channels_last', activation='relu', padding='valid'))
cnn_model1.add(MaxPooling2D(pool_size=(10, 3), padding='valid'))
cnn_model1.add(Flatten())
cnn_model1.add(Dense(label_train_prep.shape[1], activation='softmax'))
#cnn_model1.add(Dense(3, activation='softmax'))
cnn_model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model1.summary()
cnn_model1.fit(data_seq_train, label_seq_train, epochs= 10, batch_size=128)
#%% Plot- CNNs Model result
preds = cnn_model1.predict(data_seq_val)
labelcategories = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent','Standing']
#print(preds)
plt.figure(figsize=(15, 9))
cmap = plt.get_cmap('jet_r')
labelpreds0, = plt.plot(t_val, preds[:, 0]*450, label=labelcategories[0], color = cmap(0/len(labelcategories)))
labelpreds1, = plt.plot(t_val, preds[:, 1]*450, label=labelcategories[1], color = cmap(1/len(labelcategories)))
labelpreds2, = plt.plot(t_val, preds[:, 2]*450, label=labelcategories[2], color = cmap(2/len(labelcategories)))
labelpreds3, = plt.plot(t_val, preds[:, 3]*450, label=labelcategories[3], color = cmap(3/len(labelcategories)))
labelpreds4, = plt.plot(t_val, preds[:, 4]*450, label=labelcategories[4], color = cmap(4/len(labelcategories)))
labelpreds5, = plt.plot(t_val, preds[:, 5]*450, label=labelcategories[5], color = cmap(5/len(labelcategories)))
labelpreds6, = plt.plot(t_val, preds[:, 6]*450, label=labelcategories[6], color = cmap(6/len(labelcategories)))
labelgt0, = plt.plot(t_val, label_val_prep[:, 0]*(-450),marker='.', label=[labelcategories[0],'-gt'], color = cmap(0/len(labelcategories)))
labelgt1, = plt.plot(t_val, label_val_prep[:, 1]*(-450),marker='.', label=[labelcategories[1],'-gt'], color = cmap(1/len(labelcategories)))
labelgt2, = plt.plot(t_val, label_val_prep[:, 2]*(-450),marker='.', label=[labelcategories[2],'-gt'], color = cmap(2/len(labelcategories)))
labelgt3, = plt.plot(t_val, label_val_prep[:, 3]*(-450),marker='.', label=[labelcategories[3],'-gt'], color = cmap(3/len(labelcategories)))
labelgt4, = plt.plot(t_val, label_val_prep[:, 4]*(-450),marker='.', label=[labelcategories[4],'-gt'], color = cmap(4/len(labelcategories)))
labelgt5, = plt.plot(t_val, label_val_prep[:, 5]*(-450),marker='.', label=[labelcategories[5],'-gt'], color = cmap(5/len(labelcategories)))
labelgt6, = plt.plot(t_val, label_val_prep[:, 6]*(-450),marker='.', label=[labelcategories[6],'-gt'], color = cmap(6/len(labelcategories)))

firstlegend = plt.legend(handles=[labelpreds0,labelpreds1,labelpreds2,labelpreds3,labelpreds4,labelpreds5,labelpreds6],
                         bbox_to_anchor=(0.5,0.7),ncol=3)
ax = plt.gca().add_artist(firstlegend)
secondlegend = plt.legend(handles=[labelgt0,labelgt1, labelgt2, labelgt3, labelgt4, labelgt5, labelgt6],
                          bbox_to_anchor=(0.65,0.2),ncol=3)
ax = plt.gca().add_artist(secondlegend)
#labelpreds, = plt.plot(t, label_prep*4)
data1, = plt.plot(t_val, selecteddatain_val[:,-2-1], label=categories1[-2-1-8])
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
