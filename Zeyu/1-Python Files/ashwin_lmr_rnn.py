
# coding: utf-8

# # Locomotion Mode Classification Using RNN Architectures

# ## Imports

# In[1]:

import os
import numpy as np
from scipy.io import loadmat
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, LSTM, SimpleRNN, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten
import keras.backend as K
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence, to_categorical


# ## Read dataset and preprocess

# In[17]:


# Read sit-stand data from file
sit_stand_data = loadmat('../../Datasets/ashwin_testdata/sit_stand_data_labeled.mat')
walk_data = loadmat('../../Datasets/ashwin_testdata/walk_data_ashwin.mat')


# Take one 90 seconds of data from the sit-stand and 2 minutes from the walking data for Testing.
tmp_x = sit_stand_data['imu_features']
tmp_y = sit_stand_data['lm_label']
# Sit-Stand test set
X_test = tmp_x[tmp_x.shape[0]-8950:, :]
Y_test = tmp_y[tmp_x.shape[0]-8950:, :]/100.
# Sit-Stand training set
X_train = tmp_x[:tmp_x.shape[0]-8950, :]
Y_train = tmp_y[:tmp_x.shape[0]-8950, :]/100

tmp_x = walk_data['imu_features']
tmp_y = walk_data['lm_label']/300.

# Total test set
X_test = np.vstack((X_test, tmp_x[tmp_x.shape[0]-12000:, :]))
Y_test = np.vstack((Y_test, tmp_y[tmp_y.shape[0]-12000:, :]))

# Total training set
X_train = np.vstack((X_train, tmp_x[:tmp_x.shape[0]-12000, :]))
Y_train = np.vstack((Y_train, tmp_y[:tmp_x.shape[0]-12000, :]))

print(X_test.shape)
print(X_train.shape)

plt.figure(figsize=(15, 9))
plt.title("Test Set")
plt.plot(X_test[:, 56:])
plt.plot(Y_test*100)
plt.ylim([-120, 120])
plt.show()

# Function to preprocess the data into sequences for the RNN
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

