# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:20:46 2018

@author: Zed_Luz
"""
import lzy_utils
import config
import os
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn import preprocessing

class DATA(config.Config):
    DataType = ['Features', 'MVC', 'Processed', 'Raw']

    def __init__(self, SplitData = False, ReadALLSubjs = False):
        # self.path = config.Config.DATAPATH
        self.Subjects = []# ['AB185', 'AB186',...]
        self.SCANEDSubjects = []
        self.__DATAFileName = [] # [Subj1[Trails],Subj2[Trails],[...],[...]]
        self.IfReadAll = ReadALLSubjs
        self.IfSplitData = SplitData
        self.CHOOSEDCHANNELS=[]
    def scandata(self, Subj = None):
        self.__DATAFileName = []#Refresh the variable if case it overlap with multi times calling this methods.
        self.SCANEDSubjects = Subj
        if self.IfReadAll:
            for (dirpath, dirnames, filenames) in os.walk(self.DATAPATH):
                self.Subjects.extend(dirnames)
                break
            for i in range(len(self.Subjects)):
                DataFileName=[]
                for (dirpath, dirnames, filenames) in os.walk(os.path.join(self.DATAPATH,self.Subjects[i],self.DataType[2])):
                    DataFileName.extend(filenames)
                    for j in range(len(DataFileName)):
                        DataFileName[j]=os.path.join(self.DATAPATH,self.Subjects[i],self.DataType[2], str(DataFileName[j]))
                    break
                self.__DATAFileName.append(DataFileName)
        else:
            try:
                for (dirpath, dirnames, filenames) in os.walk(self.DATAPATH):
                    self.Subjects.extend(dirnames)
                    break
                for i in range(len(self.SCANEDSubjects)):
                    DataFileName=[]
                    for (dirpath, dirnames, filenames) in os.walk(os.path.join(self.DATAPATH,self.Subjects[self.SCANEDSubjects[i]-1],self.DataType[2])):
                        DataFileName.extend(filenames)
                        for j in range(len(DataFileName)):
                            DataFileName[j]=os.path.join(self.DATAPATH,self.Subjects[self.SCANEDSubjects[i]-1],self.DataType[2], str(DataFileName[j]))
                        break
                    print('Scaning the',self.SCANEDSubjects[i],'subject is done.')
                    self.__DATAFileName.append(DataFileName)
            except TypeError:
                print('WARNING: please type in which subjects data you want read in, eg. [1] or [1,2,4,5,6]')
                    
        
    def stackdata(self, ChooseSensors=['IMU','GONIO']):
        self.CHOOSEDCHANNELS=[] # refresh
        for sensor in range(len(ChooseSensors)):
            if ChooseSensors[sensor] == 'IMU':
                self.CHOOSEDCHANNELS.extend(self.INDEX_IMU)
            if ChooseSensors[sensor] == 'EMG':
                self.CHOOSEDCHANNELS.extend(self.INDEX_EMG)
            if ChooseSensors[sensor] == 'GONIO':
                self.CHOOSEDCHANNELS.extend(self.INDEX_GONIO)
        print( len(self.CHOOSEDCHANNELS),'channels of',ChooseSensors,'data are selected:')
        if self.IfSplitData:
            print('Start stacking data...(Split data into train set and val set), and split ratio is ', config.Config.SPILTRATIO)
            for i in range(len(self.__DATAFileName)):
                for j in range(len(self.__DATAFileName[i])):
                    array_data1, dict_data1, rowcount1, colcount1, categories1, label_prep1 = lzy_utils.readdata(self.__DATAFileName[i][j])
                    array_data1 = array_data1[:,self.CHOOSEDCHANNELS]
                    if i==0 and j ==0:
                        CUTPoint = int(len(array_data1)*(1-config.Config.SPILTRATIO))#split the data, cutpoint is a integer.
                        Train_data_all = array_data1[:CUTPoint,:]
                        Train_label_all = label_prep1[:CUTPoint,:]
                        Val_data_all = array_data1[CUTPoint+1:,:]
                        Val_label_all = label_prep1[CUTPoint+1:,:]
                    else:
                        Train_data_all = np.vstack((Train_data_all,array_data1[:CUTPoint,:]))
                        Train_label_all = np.vstack((Train_label_all,label_prep1[:CUTPoint,:]))
                        Val_data_all = np.vstack((Val_data_all,array_data1[CUTPoint+1:,:]))
                        Val_label_all = np.vstack((Val_label_all,label_prep1[CUTPoint+1:,:]))
                print('Finish stacking data of the',i+1,'subjects:',j+1,'trails')
            self.TRAIN_DATA_all = Train_data_all
            self.TRAIN_LABEL_all = Train_label_all
            self.VAL_DATA_all = Val_data_all
            self.VAL_LABEL_all = Val_label_all
            print('Done stacking data!', '\nTrain dataset:', self.TRAIN_DATA_all.shape, 
                  '\nVal dataset:', self.VAL_DATA_all.shape)
        else:
            print('Start stacking data...(all data are stacked together)')
            for i in range(len(self.__DATAFileName)):
                for j in range(len(self.__DATAFileName[i])):
                    array_data1, dict_data1, rowcount1, colcount1, categories1, label_prep1 = lzy_utils.readdata(self.__DATAFileName[i][j])
                    array_data1 = array_data1[:,self.CHOOSEDCHANNELS]
                    if i==0 & j ==0:
                        data_all = array_data1
                        label_all = label_prep1
                    else:
                        data_all = np.vstack((data_all,array_data1))
                        label_all = np.vstack((label_all,label_prep1))
                print('Finish stacking data of ',i+1,'subjects')
            self.DATA_all = data_all
            self.LABEL_all = label_all
            print('Done stacking data!\n', 'ALL Dataset:', self.DATA_all.shape)
        
    def PreprocessData(self, Preprocess='AbsMax'):# default scaller is MaxAbs
        try:
            print('The preprocess method is',Preprocess)
        except NameError:
            print('Please enter in the preprocess method, ')
        if Preprocess=='AbsMax':
            scaler = preprocessing.MaxAbsScaler()
        if Preprocess=='MinMax':
            scaler = preprocessing.MinMaxScaler()
        if self.IfSplitData:
            try:
                self.TRAIN_DATA_all = scaler.fit_transform(self.TRAIN_DATA_all)
                self.VAL_DATA_all = scaler.fit_transform(self.VAL_DATA_all)
            except NameError:
                print('The scaller haven''t been defined, the data hasn''t been processed')
        else:
            try:
                self.DATA_all = scaler.fit_transform(self.DATA_all)
                self.LABEL_all = scaler.fit_transform(self.LABEL_all)
            except NameError:
                print('The scaller haven''t been defined, the data hasn''t been processed')
        print('.\n..\n...\nPreprocess is done')
    def DisplayData(self):
                # Display train data
        print('Illustate ',len(self.CHOOSEDCHANNELS) , 'channels of data...')
        exp_dur_train = self.TRAIN_DATA_all.shape[0]/500 # sample rate: 500Hz
        exp_dur_val = self.VAL_DATA_all.shape[0]/500 # sample rate: 500Hz
        t_train = np.linspace(0, exp_dur_train, self.TRAIN_DATA_all.shape[0])
        t_val = np.linspace(0, exp_dur_val, self.VAL_DATA_all.shape[0])
        plot1 = plt.figure(figsize=(10, 6),constrained_layout=True)
        spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=plot1)
        cm = plt.get_cmap(name='gist_rainbow')
        
        TRAINPLOT = plot1.add_subplot(spec1[0, 0],title='Train data set')
        for i in range(self.TRAIN_LABEL_all.shape[1]):
                plt.plot(t_train, self.TRAIN_LABEL_all[:,i]*self.TRAIN_DATA_all.max(), label=self.LABELCATEGORIES[i])
                plt.set_cmap(cm)
                plt.fill_between(t_train, self.TRAIN_LABEL_all[:,i]*self.TRAIN_DATA_all.min(), 
                                 self.TRAIN_LABEL_all[:,i]*self.TRAIN_DATA_all.max(), alpha=0.3)
        plt.plot(t_train, self.TRAIN_DATA_all)
        TRAINPLOT.set_xlabel('Time(s)')
        
        VALPLOT = plot1.add_subplot(spec1[0, 1],title='Val data set')
        for i in range(self.VAL_LABEL_all.shape[1]):
                plt.plot(t_val, self.VAL_LABEL_all[:,i]*self.VAL_DATA_all.max(), label=self.LABELCATEGORIES[i])
                plt.set_cmap(cm)
                plt.fill_between(t_val, self.VAL_LABEL_all[:,i]*self.VAL_DATA_all.min(), 
                                 self.VAL_LABEL_all[:,i]*self.VAL_DATA_all.max(), alpha=0.3)
        plt.plot(t_val, self.VAL_DATA_all)
        VALPLOT.set_xlabel('Time(s)')
    def PrepareCNNdataset(self):
        data_seq_train_Oshape, self.TRAIN_LABEL_all = lzy_utils.get_sub_sequences(self.TRAIN_DATA_all, 
                                                            self.TRAIN_LABEL_all,
                                                            window_size=self.WINDOW_SIZE,
                                                            step_size=self.STEP_SIZE)
        self.TRAIN_DATA_all = np.reshape(data_seq_train_Oshape, newshape=(data_seq_train_Oshape.shape[0],
                                                                     data_seq_train_Oshape.shape[1],
                                                                     data_seq_train_Oshape.shape[2], 1))
        data_seq_val_Oshape, self.VAL_LABEL_all = lzy_utils.get_sub_sequences(self.VAL_DATA_all,
                                                                         self.VAL_LABEL_all,
                                                                         window_size=self.WINDOW_SIZE,
                                                                         step_size=self.STEP_SIZE)
        self.VAL_DATA_all = np.reshape(data_seq_val_Oshape, newshape=(data_seq_val_Oshape.shape[0],
                                                                      data_seq_val_Oshape.shape[1], 
                                                                      data_seq_val_Oshape.shape[2], 1))
        print('The dataset for CNN is prepared',
              '\nwhose shape of train set is',self.TRAIN_DATA_all.shape,
              '\nwhose shape of val set is',self.VAL_DATA_all.shape)
    def PrepareRNNdataset(self):
        self.TRAIN_DATA_all, self.TRAIN_LABEL_all = lzy_utils.get_sub_sequences(self.TRAIN_DATA_all, 
                                                            self.TRAIN_LABEL_all,
                                                            window_size=self.WINDOW_SIZE,
                                                            step_size=self.STEP_SIZE)
        self.VAL_DATA_all, self.VAL_LABEL_all = lzy_utils.get_sub_sequences(self.VAL_DATA_all,
                                                                         self.VAL_LABEL_all,
                                                                         window_size=self.WINDOW_SIZE,
                                                                         step_size=self.STEP_SIZE)
        print('The dataset for CNN is prepared',
              '\nwhose shape of train set is',self.TRAIN_DATA_all.shape,
              '\nwhose shape of val set is',self.VAL_DATA_all.shape)


            
    # def splitData(self):
        

        # datapath = os.path.join(self.DATAPATH,self.Subjects[0],self.DataType[2], self.DataFileName[0])
        # array_data_in_notrigger_float, dict_data_in, rowcount, colcount, categories, label_prep = lzy_utils.readdata(self.DATAPATH)
        # return array_data_in_notrigger_float, dict_data_in, rowcount, colcount, categories, label_prep
    