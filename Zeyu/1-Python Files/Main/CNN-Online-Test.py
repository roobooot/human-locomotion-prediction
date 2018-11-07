
# coding: utf-8

# In[47]:
import model
import matplotlib.pyplot as plt


# In[45]:
OnlineData = model.DATA(ReadALLSubjs=False,SplitData=True)


# In[46]:


OnlineData.scandata(Subj=[1])
OnlineData.stackdata(ChooseSensors=['IMU'])


# In[14]:
OnlineData.PreprocessData(Preprocess='AbsMax')


# In[6]:
OnlineData.displayConfig()

# In[15]:


#OnlineData.displaydata()


# In[17]:


OnlineData.PrepareCNNdataset()


# ### CNNs model1

# In[43]:

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
cnn_model1 = Sequential()
cnn_model1.add(Conv2D(filters=50,
                      kernel_size=(5,1),
                      input_shape=(OnlineData.WINDOW_SIZE, OnlineData.TRAIN_DATA_all.shape[2], 1),
                      data_format='channels_last',
                      activation='relu',
                      padding='valid'
                     ))
cnn_model1.add(MaxPooling2D(pool_size=(2, 1),
                            padding='valid',
                            data_format='channels_last'
                           ))
cnn_model1.add(Dropout(0.5))
cnn_model1.add(Conv2D(filters=50,
                      kernel_size=(5,1),
                      input_shape=cnn_model1.output_shape[1:4],
                      data_format='channels_last',
                      activation='relu',
                      padding='valid'
                      ))
cnn_model1.add(MaxPooling2D(pool_size=(2,1),
                            padding='valid',
                            data_format='channels_last'
                           ))
cnn_model1.add(Dropout(0.5))
cnn_model1.add(Conv2D(filters=40,
                     kernel_size=(5,1),
                     input_shape=cnn_model1.output_shape[1:4],
                     data_format='channels_last',
                     activation='relu',
                     padding='valid'
                     ))
cnn_model1.add(MaxPooling2D(pool_size=(2,1),
                            padding='valid',
                            data_format='channels_last'
                           ))
cnn_model1.add(Dropout(0.5))
cnn_model1.add(Conv2D(filters=40,
                     kernel_size=(3,1),
                     input_shape=cnn_model1.output_shape[1:4],
                     data_format='channels_last',
                     activation='relu',
                     padding='valid'
                     ))
cnn_model1.add(MaxPooling2D(pool_size=(2,1),
                            padding='valid',
                            data_format='channels_last'
                           ))
cnn_model1.add(Conv2D(filters=40,
                     kernel_size=(3,1),
                     input_shape=cnn_model1.output_shape[1:4],
                     data_format='channels_last',
                     activation='relu',
                     padding='valid'
                     ))
cnn_model1.add(Conv2D(filters=400,
                     kernel_size=(1,cnn_model1.output_shape[2]),
                     input_shape=cnn_model1.output_shape[1:4],
                     data_format='channels_last',
                     activation='relu',
                     padding='valid'
                     ))
cnn_model1.add(Flatten())
cnn_model1.add(Dense(OnlineData.TRAIN_LABEL_all.shape[1],
                     activation='softmax'
                     ))
cnn_model1.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
cnn_model1.summary()


# In[41]:
CNN1281525612=cnn_model1.fit(OnlineData.TRAIN_DATA_all,
                             OnlineData.TRAIN_LABEL_all,
                             epochs=10,
                             batch_size=128,
                             validation_data=(OnlineData.VAL_DATA_all,OnlineData.VAL_LABEL_all))

# In[23]:
Figloss = plt.figure(1)
plt.plot(CNN1281525612.history['loss'], label='train_loss')
plt.plot(CNN1281525612.history['val_loss'], label='val_loss')
plt.title('CNN-10Epochs--Dropout(0.5)')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show(Figloss)
FigAcc = plt.figure(2)
plt.plot(CNN1281525612.history['acc'], label='train_acc')
plt.plot(CNN1281525612.history['val_acc'],label='val_acc')
plt.title('CNN-10epoches--Dropout(0.5)')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show(FigAcc)
