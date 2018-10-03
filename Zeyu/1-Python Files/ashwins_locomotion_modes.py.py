
# coding: utf-8

# In[15]:


# Imports
import numpy as np
from scipy.io import loadmat
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.utils import plot_model
from keras.layers import Input, Dense, Activation, LSTM, SimpleRNN, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten
import keras.backend as K
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence, to_categorical
from keras.callbacks import TensorBoard

# ## Import the IMU Data

# In[2]:

IfReadModel = True
# Import the data from the mat file

data_in = loadmat('../../Datasets/ashwin_testdata/walk_data_ashwin.mat')
X = data_in['imu_features']
Y = data_in['lm_label']/300
Ytemp = np.argwhere(Y)[:, 1]
Yint = np.zeros(Ytemp.shape)
for i in range(Ytemp.shape[0]):
    if Ytemp[i] == 1:
        Yint[i] = 0
    elif Ytemp[i] == 4:
        Yint[i] = 1
    elif Ytemp[i] == 5:
        Yint[i] = 2
            
Y_prep = to_categorical(Yint)


# Get some statistics from the data
exp_dur = X.shape[0]/100.0
nfeat = X.shape[1]
t = np.linspace(0, exp_dur, X.shape[0])

# Plot parts of the data
plt.figure(figsize=(10, 6))
plt.plot(t, X[:, 56])
plt.plot(t, Y*300)
plt.xlim([6, 12])
plt.ylabel('Features')
plt.xlabel('Time (s)')
plt.show()


# ## Preprocess the data

# In[3]:

# Normalization
scale_factors = np.std(X, axis=0)
mean_values = np.mean(X, axis=0)
X_prep = (X - mean_values)/scale_factors
#print("After normalizing: ")
#print("Mean is: ", np.mean(X_prep, axis=0))
#print("Stdev. is: ", np.std(X_prep, axis=0))

# Plot all of the normalized data
plt.figure(figsize=(10, 6))
plt.plot(t, X_prep)
plt.plot(t, Y*10)
plt.xlim([6, 12])
plt.ylabel('Features')
plt.xlabel('Time (s)')
plt.show()


# ## Trying Logistic Regression

# In[4]:


lr_model = Sequential()
lr_model.add(Dense(32, input_dim=nfeat, activation='relu'))
lr_model.add(Dense(3))
lr_model.add(Activation('softmax'))
lr_model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
lr_model.fit(X_prep, Y_prep, epochs=30, batch_size=64)


# ### Plot the labels
# 

# In[5]:


preds = lr_model.predict(X_prep)
print(preds)
plt.figure(figsize=(15, 9))
plt.plot(t, X[:, 56], label='Right Foot Tilt')
plt.plot(t, X[:, 57], label='Right Shank Tilt')
plt.plot(t, X[:, 58], label='Right Thigh Tilt')
plt.plot(t, preds[:, 0]*40, label='Standing')
plt.plot(t, preds[:, 1]*40, label='Walking')
plt.plot(t, preds[:, 2]*40, label='Unknown')
plt.xlim([26, 46])
plt.ylim([-50, 50])
plt.ylabel('Features')
plt.xlabel('Time (s)')
plt.title('Locomotion Mode Prediction Using Logistic Regression')
plt.grid()
plt.legend()
plt.show()


# ## Using CNNs

# ### CNN Specific Pre-Processing of the data

# In[23]:


# Function to split the data and labels into windows that are causally aligned.
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

# # Split into test data and train data
# X_test = X_prep[X_prep.shape[0]-8950:, :]
# Y_test = Y_prep[Y_prep.shape[0]-8950:, :]
# X_train = X_prep[:X_prep.shape[0]-8950, :]
# Y_train = Y_prep[:Y_prep.shape[0]-8950, :]

# Generate dataset of sub-sequences
X_seq, Y_seq = get_sub_sequences(X_prep, Y_prep, window_size=120, step_size=1)
X_seq = np.reshape(X_seq, newshape=(X_seq.shape[0], X_seq.shape[1], X_seq.shape[2], 1))
# X_seq_test, Y_seq_test = get_sub_sequences(X_test, Y_test, window_size=120, step_size=1)
# X_seq_test = np.reshape(X_seq_test, newshape=(X_seq_test.shape[0], X_seq_test.shape[1], X_seq_test.shape[2], 1))
# X_seq_train, Y_seq_train = get_sub_sequences(X_train, Y_train, window_size=120, step_size=1)
# X_seq_train = np.reshape(X_seq_train, newshape=(X_seq_train.shape[0], X_seq_train.shape[1], X_seq_train.shape[2], 1))
# Plot some subsequences to make sure that we're doing it right
# plt.figure(figsize=(15, 9))
# for i in range(1, 10, 1):
#     plt.subplot(3, 3, i)
#     plt.plot(X_seq[i*2+590, :, 56:59])
#     plt.title('Subsequence from {}, LM={}'.format(i*2+590, Y_seq[i]))
#     plt.xlabel('Sample')
#     plt.ylabel('Value')
#     plt.grid()

# plt.show()


# ### Model 1

# In[25]:

if IfReadModel:
    cnn_model1 = load_model('CNN_Model.h5')
else:
    cnn_model1 = Sequential()
    cnn_model1.add(Conv2D(filters=16, kernel_size=9, input_shape=(120, 63, 1),
                          data_format='channels_last', activation='relu', padding='valid'))
    cnn_model1.add(MaxPooling2D(pool_size=(10, 3), padding='valid'))
    cnn_model1.add(Flatten())
    cnn_model1.add(Dense(3, activation='softmax'))
    #cnn_model1.add(Dense(3, activation='softmax'))
    cnn_model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model1.summary()
    print("X_seq_train: ", X_seq.shape)
    print("Y_seq_train: ", Y_seq.shape)
    dlcallback = keras.callbacks.TensorBoard(log_dir='./TensorboardLogs', histogram_freq=10,
                                             write_graph=True, write_images=True, write_grads=True, batch_size=128,
                                             update_freq='batch')
    cnn_model1.fit(X_seq, Y_seq, validation_split=0.33, batch_size=128, callbacks=[dlcallback])
    plot_model(cnn_model1, to_file='model.png') # Visualization
# score = cnn_model1.evaluate(X_seq_test, Y_seq_test, batch_size=128)
cnn_preds = cnn_model1.predict(X_seq) # Results of the prediction from the trained model

# ### Plot the results
plt.figure(figsize=(15, 9))
plt.plot(t, X[:, 56], label='Right Foot Tilt')
plt.plot(t, X[:, 57], label='Right Shank Tilt')
plt.plot(t, X[:, 58], label='Right Thigh Tilt')
plt.plot(t, cnn_preds[:, 0]*40, label='Standing')
plt.plot(t, cnn_preds[:, 1]*40, label='Walking')
plt.plot(t, cnn_preds[:, 2]*40, label='Unknown')
plt.xlim([26, 46])
plt.ylim([-50, 50])
plt.ylabel('Features')
plt.xlabel('Time (s)')
plt.title('Locomotion Mode Prediction Using CNN')
plt.grid()
plt.legend()
plt.show()
