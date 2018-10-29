
# coding: utf-8

# In[1]:


#-*- coding:utf-8 -*-
import numpy as np
from scipy.io import loadmat, savemat
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Reshape
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
import os
from lzy_utils import readdata, get_sub_sequences
from sklearn import preprocessing
#from imu_preprocess_utils import *
#import tsne as tsne

#%% Import data and save as dict and list
DatasetPath = r'C:\Users\Zed_Luz\OneDrive - 南方科技大学\BigScaleFiles\1-datasets for bilateral lower limb neuromechanical signals\2-Data'
#OnlineDatapath = r'C:\Users\Zed_Luz\OneDrive - 南方科技大学\BigScaleFiles\1-datasets for bilateral lower limb neuromechanical signals\2-Data\AB185\Processed\AB185_Circuit_001_post.csv'
OnlineDatapath = dict()
IfMaxAbsPreprocess = True
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

#%%
if IfMaxAbsPreprocess:
    max_abs_scaler = preprocessing.MaxAbsScaler()
    Train_data = max_abs_scaler.fit_transform(Train_data)
    Val_data = max_abs_scaler.fit_transform(Val_data)
#%% Select specific channels of data 
labelcategories = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent',
                       'Standing']
print(len(categories1),'All of Data types:',categories1)
selectedchannels = ['Right_Shank_Ax', 'Right_Shank_Ay', 'Right_Shank_Az', 'Right_Shank_Gy', 
                    'Right_Shank_Gz', 'Right_Shank_Gx', 'Right_Thigh_Ax', 'Right_Thigh_Ay', 
                    'Right_Thigh_Az', 'Right_Thigh_Gy', 'Right_Thigh_Gz', 'Right_Thigh_Gx', 
                    'Left_Shank_Ax', 'Left_Shank_Ay', 'Left_Shank_Az', 'Left_Shank_Gy', 
                    'Left_Shank_Gz', 'Left_Shank_Gx', 'Left_Thigh_Ax', 'Left_Thigh_Ay', 
                    'Left_Thigh_Az', 'Left_Thigh_Gy', 'Left_Thigh_Gz', 'Left_Thigh_Gx', 
                    'Waist_Ax', 'Waist_Ay', 'Waist_Az', 'Waist_Gy', 'Waist_Gz', 'Waist_Gx',
                    'Right_Ankle', 'Right_Knee', 'Left_Ankle', 'Left_Knee', 
                    'Right_Ankle_Velocity', 'Right_Knee_Velocity', 
                    'Left_Ankle_Velocity', 'Left_Knee_Velocity']
# selectedchannels = ['Right_Ankle', 'Right_Knee', 'Left_Ankle', 'Left_Knee', 
#                     'Right_Ankle_Velocity', 'Right_Knee_Velocity', 
#                     'Left_Ankle_Velocity', 'Left_Knee_Velocity']
selectedindex = [categories1.index(selectedchannels[i]) 
                    for i in range(len(selectedchannels))]
selectedchannelsNum = len(selectedchannels)
selecteddatain_train = Train_data[:,selectedindex]
selecteddatain_val = Val_data[:,selectedindex]
label_train = dict_data1['Mode']
label_train_prep = to_categorical(label_train)
label_val = dict_data2['Mode']
label_val_prep = to_categorical(label_val)
exp_dur_train = rowcount1/500 # sample rate: 500Hz
exp_dur_val = rowcount2/500 # sample rate: 500Hz
nfeat = selectedchannelsNum
t_train = np.linspace(0, exp_dur_train, rowcount1)
t_val = np.linspace(0, exp_dur_val, rowcount2)
#%% Display train data
print('Illustate ', selectedchannelsNum, 'channels of data...')
plt.figure(figsize=(10, 6))
cm = plt.get_cmap(name='gist_rainbow')
for i in range(7):
        plt.plot(t_train, label_prep1[:,i]*selecteddatain_train.max(), label=labelcategories[i])
        plt.set_cmap(cm)
        plt.fill_between(t_train, label_prep1[:,i]*selecteddatain_train.min(), 
                         label_prep1[:,i]*selecteddatain_train.max(), alpha=0.3)
plt.plot(t_train, selecteddatain_train)
plt.legend()
plt.show()
#%% First Model-Regression
lr_model = Sequential()
lr_model.add(Dense(32, input_dim=nfeat, activation='relu'))
lr_model.add(Dense(label_train_prep.shape[1]))
lr_model.add(Activation('softmax'))
lr_model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
lr_model.fit(selecteddatain_train, label_train_prep, epochs=30, batch_size=64)
lr_score = lr_model.evaluate(selecteddatain_val, label_val_prep, batch_size=64)
# lr_score = lr_model.evaluate(selecteddatain_train,label_train_prep)
print(lr_score)
#%% Second Model-Regression



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

#%% Third Model-Regression


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

#%% CNNs config
win_size = 300
step_leng = 10
data_seq_train_Oshape, label_seq_train = get_sub_sequences(selecteddatain_train, 
                                                    label_train_prep, window_size=win_size, step_size=step_leng)
data_seq_train = np.reshape(data_seq_train_Oshape, newshape=(data_seq_train_Oshape.shape[0], data_seq_train_Oshape.shape[1], 
                                                      data_seq_train_Oshape.shape[2], 1))
data_seq_val_Oshape, label_seq_val = get_sub_sequences(selecteddatain_val, label_val_prep, window_size=win_size,
                                                step_size=step_leng)
data_seq_val = np.reshape(data_seq_val_Oshape, newshape=(data_seq_val_Oshape.shape[0], data_seq_val_Oshape.shape[1], 
                                                  data_seq_val_Oshape.shape[2], 1))
#%%Train
cnn_model1 = Sequential()
cnn_model1.add(Conv2D(filters=128, kernel_size=(15,data_seq_train.shape[2]), input_shape=(win_size, data_seq_train.shape[2], 1),
                      data_format='channels_last', activation='relu', padding='valid'))
cnn_model1.add(MaxPooling2D(pool_size=(2, 1), padding='valid', data_format='channels_last'))
cnn_model1.add(Reshape((cnn_model1.output_shape[1],cnn_model1.output_shape[3],1), input_shape=cnn_model1.output_shape[1:4]))
cnn_model1.add(Conv2D(filters=256, kernel_size=(24,cnn_model1.output_shape[2]), input_shape=cnn_model1.output_shape[1:4],
                      data_format='channels_last', activation='relu', padding='valid'))
cnn_model1.add(MaxPooling2D(pool_size=(2,1), padding='valid', data_format='channels_last'))
cnn_model1.add(Flatten())
cnn_model1.add(Dense(label_train_prep.shape[1], activation='softmax'))
#cnn_model1.add(Dense(3, activation='softmax'))
cnn_model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model1.summary()
cnn_model1.fit(data_seq_train, label_seq_train, epochs= 10, batch_size=128)
score = cnn_model1.evaluate(data_seq_val, label_seq_val)
print('Evaluation Loss:', score[0],'Evaluation Accuracy:', score[1])
#%%Model2
cnn_model2 = Sequential()
cnn_model2.add(Conv2D(filters=128, kernel_size=(15,1), input_shape=(win_size, data_seq_train.shape[2], 1),
                      data_format='channels_last', activation='relu', padding='valid'))
cnn_model2.add(MaxPooling2D(pool_size=(2, 1), padding='valid', data_format='channels_last'))
#cnn_model2.add(Reshape((43,128,38), input_shape=(43,38,128)))
cnn_model2.add(Conv2D(filters=256, kernel_size=(12,1), input_shape=(43,38,128),
                      data_format='channels_last', activation='relu', padding='valid'))
cnn_model2.add(MaxPooling2D(pool_size=(2,1), padding='valid', data_format='channels_last'))
cnn_model2.add(Conv2D(filters=256, kernel_size=(15,1), input_shape=(16,38,256),
                      data_format='channels_last', activation='relu', padding='valid'))
cnn_model2.add(MaxPooling2D(pool_size=(2,1), padding='valid', data_format='channels_last'))
cnn_model2.add(Flatten())
cnn_model2.add(Dense(label_train_prep.shape[1], activation='softmax'))
#cnn_model1.add(Dense(3, activation='softmax'))
cnn_model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model2.summary()
cnn_model2.fit(data_seq_train, label_seq_train, epochs= 10, batch_size=128)
score = cnn_model1.evaluate(data_seq_val, label_seq_val)
print('Evaluation Loss:', score[0],'Evaluation Accuracy:', score[1])
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

#%%Evaluation for different sets
data2 = os.path.join(DatasetPath,Subjects[0],DataType[2], DataFileName[4])
array_data2, dict_data2, rowcount2, colcount2, categories2, label_prep2 = readdata(data2)
Val_data = array_data2
selecteddatain_val = Val_data[:,selectedindex]
label_val = dict_data2['Mode']
label_val_prep = to_categorical(label_val)
data_seq_val, label_seq_val = get_sub_sequences(selecteddatain_val, label_val_prep, window_size=100, step_size=2)
data_seq_val = np.reshape(data_seq_val, newshape=(data_seq_val.shape[0], data_seq_val.shape[1], data_seq_val.shape[2], 1))
score = cnn_model1.evaluate(data_seq_val, label_seq_val)
print('Evaluation Loss:', score[0],'Evaluation Accuracy:', score[1])
