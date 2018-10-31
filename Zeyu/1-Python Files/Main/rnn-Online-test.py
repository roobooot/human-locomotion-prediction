
# coding: utf-8

# In[2]:


#-*- coding:utf-8 -*-
import numpy as np
from scipy.io import loadmat, savemat
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers import Flatten, Reshape, Dropout, SimpleRNN, LSTM
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


# In[3]:


#%% Import data and save as dict and list
DatasetPath = r'C:\Users\Zed_Luz\OneDrive - 南方科技大学\BigScaleFiles\1-datasets for bilateral lower limb neuromechanical signals\2-Data'
#OnlineDatapath = r'C:\Users\Zed_Luz\OneDrive - 南方科技大学\BigScaleFiles\1-datasets for bilateral lower limb neuromechanical signals\2-Data\AB185\Processed\AB185_Circuit_001_post.csv'
OnlineDatapath = dict()
IfMaxAbsPreprocess = False
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


# In[4]:


#%%
if IfMaxAbsPreprocess:
    max_abs_scaler = preprocessing.MaxAbsScaler()
    Train_data = max_abs_scaler.fit_transform(Train_data)
    Val_data = max_abs_scaler.fit_transform(Val_data)
#Minmax
Min_max_scaller = preprocessing.MinMaxScaler()
Train_data = Min_max_scaller.fit_transform(Train_data)
Val_data = Min_max_scaller.fit_transform(Val_data)


# In[5]:


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


# In[6]:


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


# ## Win_size=200, Step_size=10

# In[14]:


win_size = 200
step_leng = 10
data_seq_train_Oshape, label_seq_train = get_sub_sequences(selecteddatain_train, 
                                                    label_train_prep, window_size=win_size, step_size=step_leng)
data_seq_train = np.reshape(data_seq_train_Oshape, newshape=(data_seq_train_Oshape.shape[0], data_seq_train_Oshape.shape[1], 
                                                      data_seq_train_Oshape.shape[2], 1))
data_seq_val_Oshape, label_seq_val = get_sub_sequences(selecteddatain_val, label_val_prep, window_size=win_size,
                                                step_size=step_leng)
data_seq_val = np.reshape(data_seq_val_Oshape, newshape=(data_seq_val_Oshape.shape[0], data_seq_val_Oshape.shape[1], 
                                                  data_seq_val_Oshape.shape[2], 1))


# In[15]:


data_seq_train_Oshape.shape


# In[16]:


label_seq_train.shape


# In[17]:


data_seq_train_Oshape.shape[-2:]


# In[18]:


model_rnn = Sequential()
model_rnn.add(LSTM(24, input_shape=(None,data_seq_train_Oshape.shape[2]),return_sequences=True))
model_rnn.add(LSTM(12,return_sequences=False))
model_rnn.add(Dense(label_seq_train.shape[1],activation='softmax'))
model_rnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model_rnn.summary()


# In[19]:


history = model_rnn.fit(data_seq_train_Oshape,label_seq_train,epochs=10, validation_data=(data_seq_val_Oshape,label_seq_val))


# In[20]:


results = model_rnn.predict(data_seq_val_Oshape)
results.shape


# In[13]:


plt.scatter(range(len(data_seq_val)),np.argmax(results,axis=-1),c='r',label='Prediction')
plt.scatter(range(len(data_seq_val)),np.argmax(label_seq_val,axis=-1),c='g',label='GroundTruth')
plt.legend()
plt.ylabel('Modes')
plt.show()


# In[14]:


plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('loss')
plt.legend()
plt.xlabel('epoch')
plt.show()


# ## Win_size=400, Step_size=10

# In[15]:


win_size = 200
step_leng = 10
data_seq_train_Oshape, label_seq_train = get_sub_sequences(selecteddatain_train, 
                                                    label_train_prep, window_size=win_size, step_size=step_leng)
data_seq_train = np.reshape(data_seq_train_Oshape, newshape=(data_seq_train_Oshape.shape[0], data_seq_train_Oshape.shape[1], 
                                                      data_seq_train_Oshape.shape[2], 1))
data_seq_val_Oshape, label_seq_val = get_sub_sequences(selecteddatain_val, label_val_prep, window_size=win_size,
                                                step_size=step_leng)
data_seq_val = np.reshape(data_seq_val_Oshape, newshape=(data_seq_val_Oshape.shape[0], data_seq_val_Oshape.shape[1], 
                                                  data_seq_val_Oshape.shape[2], 1))


# In[16]:


model_rnn = Sequential()
model_rnn.add(LSTM(24, input_shape=(None,data_seq_train_Oshape.shape[2]),
                   return_sequences=True))
model_rnn.add(LSTM(12,return_sequences=False))
model_rnn.add(Dense(label_seq_train.shape[1],activation='softmax'))
model_rnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model_rnn.summary()


# In[17]:


history = model_rnn.fit(data_seq_train_Oshape,label_seq_train,epochs=10, validation_data=(data_seq_val_Oshape,label_seq_val))


# In[18]:


results = model_rnn.predict(data_seq_val_Oshape)
results.shape


# In[19]:


plt.scatter(range(len(data_seq_val)),np.argmax(results,axis=-1),c='r',label='Prediction')
plt.scatter(range(len(data_seq_val)),np.argmax(label_seq_val,axis=-1),c='g',label='GroundTruth')
plt.legend()
plt.ylabel('Modes')
plt.show()


# In[23]:


plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('loss')
plt.legend()
plt.xlabel('epoch')
plt.show()


# ## Win_size=50, Step_size=5

# In[7]:


win_size = 50
step_leng = 5
data_seq_train_Oshape, label_seq_train = get_sub_sequences(selecteddatain_train, 
                                                    label_train_prep, window_size=win_size, step_size=step_leng)
data_seq_train = np.reshape(data_seq_train_Oshape, newshape=(data_seq_train_Oshape.shape[0], data_seq_train_Oshape.shape[1], 
                                                      data_seq_train_Oshape.shape[2], 1))
data_seq_val_Oshape, label_seq_val = get_sub_sequences(selecteddatain_val, label_val_prep, window_size=win_size,
                                                step_size=step_leng)
data_seq_val = np.reshape(data_seq_val_Oshape, newshape=(data_seq_val_Oshape.shape[0], data_seq_val_Oshape.shape[1], 
                                                  data_seq_val_Oshape.shape[2], 1))


# In[8]:


model_rnn = Sequential()
model_rnn.add(LSTM(24, input_shape=(None,data_seq_train_Oshape.shape[2]),
                   return_sequences=True))
model_rnn.add(LSTM(12,return_sequences=False))
model_rnn.add(Dense(label_seq_train.shape[1],activation='softmax'))
model_rnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model_rnn.summary()


# In[9]:


history = model_rnn.fit(data_seq_train_Oshape,label_seq_train,epochs=100, validation_data=(data_seq_val_Oshape,label_seq_val))


# In[10]:


results = model_rnn.predict(data_seq_val_Oshape)
results.shape


# In[11]:


plt.scatter(range(len(data_seq_val)),np.argmax(results,axis=-1),c='r',label='Prediction')
plt.scatter(range(len(data_seq_val)),np.argmax(label_seq_val,axis=-1),c='g',label='GroundTruth')
plt.legend()
plt.ylabel('Modes')
plt.show()


# In[13]:


plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('loss')
plt.legend()
plt.xlabel('epoch')
plt.show()

