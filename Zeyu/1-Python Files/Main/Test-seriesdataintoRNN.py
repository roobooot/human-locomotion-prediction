import model
import matplotlib.pyplot as plt

Onlinedata = model.DATA(ReadALLSubjs=True, SplitData=True)
Onlinedata.scandata()
Onlinedata.stackdata(ChooseSensors=['IMU'])
Onlinedata.PreprocessData()
Onlinedata.PrepareRNNdataset()
Onlinedata.TRAIN_DATA_all.shape
from keras.models import Sequential
from keras.layers import LSTM, Dense

model_rnn5 = Sequential()
model_rnn5.add(LSTM(512, input_shape=(None, Onlinedata.TRAIN_DATA_all.shape[1]),
                    dropout=0.5))
model_rnn5.add(Dense(Onlinedata.TRAIN_LABEL_all.shape[1], activation='softmax'))
model_rnn5.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model_rnn5.summary()
TenEpochesWithDropout = model_rnn5.fit(
    Onlinedata.TRAIN_DATA_all.reshape((Onlinedata.TRAIN_DATA_all.shape[0], 1, Onlinedata.TRAIN_DATA_all.shape[1])),
    Onlinedata.TRAIN_LABEL_all, epochs=10,
    validation_data=(
        Onlinedata.VAL_DATA_all.reshape((Onlinedata.VAL_DATA_all.shape[0], 1, Onlinedata.VAL_DATA_all.shape[1])),
        Onlinedata.VAL_LABEL_all))
model_rnn5.save('../ModelLogs/RNN-10Epochs-LSTM(512)-dropout(0.5)-AbsMax.h5')

plt.figure(1)
plt.plot(TenEpochesWithDropout.history['loss'], label='train_loss')
plt.plot(TenEpochesWithDropout.history['val_loss'], label='val_loss')
plt.title('RNN-10Epochs-LSTM(512)-Dropout(0.5)')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
plt.figure(2)
plt.plot(TenEpochesWithDropout.history['acc'], label='train_acc')
plt.plot(TenEpochesWithDropout.history['val_acc'], label='val_acc')
plt.title('RNN-10epoches-LSTM(512)-Dropout(0.5)')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
