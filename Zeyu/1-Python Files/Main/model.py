# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:20:46 2018

@author: Zed_Luz
"""
import lzy_utils
import config
import os
import numpy as np
from sklearn import preprocessing

class DATA(config.Config):
    DataType = ['Features', 'MVC', 'Processed', 'Raw']

    def __init__(self, SplitData = False, ReadALL = False):
        self.path = config.Config.DATAPATH
        self.Subjects = []# ['AB185', 'AB186',...]
        self.DATAFileName = [] # [Subj1[Trails],Subj2[Trails],[...],[...]]
        self.IfReadAll = ReadALL
        self.IfSplitData = SplitData
    def scandata(self, Subj = None):
        if self.IfReadAll:
            DataFileName=[]
            for (dirpath, dirnames, filenames) in os.walk(self.DATAPATH):
                self.Subjects.extend(dirnames)
                break
            for i in range(len(self.Subjects)):
                for (dirpath, dirnames, filenames) in os.walk(os.path.join(self.DATAPATH,self.Subjects[i],self.DataType[2])):
                    DataFileName.extend(filenames)
                    for j in range(len(DataFileName)):
                        DataFileName[j]=os.path.join(self.DATAPATH,self.Subjects[i],self.DataType[2], str(DataFileName[j]))
                    break
                self.DATAFileName.append(DataFileName)
                DataFileName=[]
        else:
            try:
                DataFileName=[]
                for (dirpath, dirnames, filenames) in os.walk(self.DATAPATH):
                    self.Subjects.extend(dirnames)
                    break
                for i in range(len(Subj)):
                    for (dirpath, dirnames, filenames) in os.walk(os.path.join(self.DATAPATH,self.Subjects[Subj[i]-1],self.DataType[2])):
                        DataFileName.extend(filenames)
                        for j in range(len(DataFileName)):
                            DataFileName[j]=os.path.join(self.DATAPATH,self.Subjects[i],self.DataType[2], str(DataFileName[j]))
                        break
                    self.DATAFileName.append(DataFileName)
                    DataFileName=[]
            except TypeError:
                print('WARNING: please type in which subjects data you want read in, eg. [1] or [1,2,4,5,6]')
                    
        
    def stackdata(self):
        if self.IfSplitData:
            print('Start stacking data...(Split data into train set and val set), and split ratio is ', config.Config.SPILTRATIO)
            for i in range(len(self.DATAFileName)):
                for j in range(len(self.DATAFileName[i])):
                    array_data1, dict_data1, rowcount1, colcount1, categories1, label_prep1 = lzy_utils.readdata(self.DATAFileName[i][j])
                    if i==0 & j ==0:
                        CUTPoint = int(len(array_data1)*(1-config.Config.SPILTRATIO))
                        Train_data_all = array_data1[:CUTPoint,:]
                        Train_label_all = label_prep1[:CUTPoint,:]
                        Val_data_all = array_data1[CUTPoint+1:,:]
                        Val_label_all = label_prep1[CUTPoint+1:,:]
    
                    else:
                        Train_data_all = np.vstack((Train_data_all,array_data1[:CUTPoint,:]))
                        Train_label_all = np.vstack((Train_label_all,label_prep1[:CUTPoint,:]))
                        Val_data_all = np.vstack((Val_data_all,array_data1[CUTPoint+1:,:]))
                        Val_label_all = np.vstack((Val_label_all,label_prep1[CUTPoint+1:,:]))
                print('Finish stacking data of ',i+1,'subjects')
            self.TRAIN_DATA_all = Train_data_all
            self.TRAIN_LABEL_all = Train_label_all
            self.VAL_DATA_all = Val_data_all
            self.VAL_LABEL_all = Val_label_all
        else:
            print('Start stacking data...(all data are stacked)')
            for i in range(len(self.DATAFileName)):
                for j in range(len(self.DATAFileName[i])):
                    array_data1, dict_data1, rowcount1, colcount1, categories1, label_prep1 = lzy_utils.readdata(self.DATAFileName[i][j])
                    if i==0 & j ==0:
                        data_all = array_data1
                        label_all = label_prep1
                    else:
                        data_all = np.vstack((data_all,array_data1))
                        label_all = np.vstack((label_all,label_prep1))
                print('Finish stacking data of ',i+1,'subjects')
            self.DATA_all = data_all
            self.LABEL_all = label_all
        print('Done stacking data!')
        
    def PreprocessData(self, Preprocess='AbsMax'):# default scaller is MaxAbs
        if self.Preprocess is not None:
            if self.Preprocess=='AbsMax':
                scaler = preprocessing.MaxAbsScaler()
            if self.Preprocess=='MinMax':
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
    


            
    # def splitData(self):
        

        # datapath = os.path.join(self.DATAPATH,self.Subjects[0],self.DataType[2], self.DataFileName[0])
        # array_data_in_notrigger_float, dict_data_in, rowcount, colcount, categories, label_prep = lzy_utils.readdata(self.DATAPATH)
        # return array_data_in_notrigger_float, dict_data_in, rowcount, colcount, categories, label_prep
    