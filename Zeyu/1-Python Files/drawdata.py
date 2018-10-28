# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:30:12 2018

@author: Zed_Luz
"""

#%% Data reader
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.utils import to_categorical
import os
from sklearn import preprocessing
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
#%% Main
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
data = os.path.join(DatasetPath,Subjects[0],DataType[2], DataFileName[0])
array_data1, dict_data1, rowcount1, colcount1, categories1, label_prep1 = readdata(data)
exp_dur = rowcount1/500
t_seq = np.linspace(0, exp_dur, rowcount1)
labelcategories = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent','Standing']

#%%Plot
plt.figure(num=0)
plt.title = 'Right_Shank_IMU'
for i in range(3):
    plt.plot(t_seq, array_data1[:,i]+i, label=categories1[i])
for i in range(7):
    plt.plot(t_seq, label_prep1[:,i]*3, label=labelcategories[i])
plt.legend()
plt.figure(num=1)
for i in range(3,6):
    plt.plot(t_seq, array_data1[:,i]+500*i, label=categories1[i])
for i in range(7):
    plt.plot(t_seq, label_prep1[:,i]*3000, label=labelcategories[i])
plt.legend()
IMUdataIndex = list(range(0,23+1,1))
EMGdataIndex = list(range(24,43+1,1))
GoniodataIndex = list(range(44,51+1,1))
ModedataIndex = 52
TriggerdataIndex = list(range(53,60+1,1))
plt.plot(t_seq, array_data1[:,IMUdataIndex])

enc = preprocessing.OrdinalEncoder()
enc.fit(np.array(categories1).reshape(-1,1))
IMUdataIndex = enc.transform([['Right_Shank_Ax'],['Left_Shank_Ax']])
X_scaled = preprocessing.scale(array_data1)
plt.plot(t_seq, X_scaled[:,5],label = 'scaled')
plt.plot(t_seq, array_data1[:,3],label = 'raw')
plt.legend()
#%% SubPlot
def PlotIMUs(IMU_Loc,Array_data, Label_prep, Channel_catagories, fig_size = (25,15)):
    labelcategories = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent','Standing']
    array_data1 = Array_data
    label_prep1 = Label_prep
    FirstIndex = Channel_catagories.index(IMU_Loc+'_Ax')
    fig2 = plt.figure(figsize=fig_size,constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig2)
    f2_ax1 = fig2.add_subplot(spec2[0, 0],title='IMU_Ax_raw')
    plt.plot(t_seq, array_data1[:,0+FirstIndex])
    for i in range(7):
        plt.plot(t_seq, label_prep1[:,i]*3, label=labelcategories[i])
    f2_ax1.set_xlabel('Time(s)')
    f2_ax1.set_ylabel('Acceleration(m/s2)')
    f2_ax2 = fig2.add_subplot(spec2[0, 1],title='IMU_Ay_raw')
    plt.plot(t_seq, array_data1[:,1+FirstIndex])
    for i in range(7):
        plt.plot(t_seq, label_prep1[:,i]*3)
    f2_ax2.set_xlabel('Time(s)')
    f2_ax2.set_ylabel('Acceleration(m/s2)')
    
    f2_ax3 = fig2.add_subplot(spec2[0, 2],title='IMU_Az_raw')
    plt.plot(t_seq, array_data1[:,2+FirstIndex])
    for i in range(7):
        plt.plot(t_seq, label_prep1[:,i]*3)
    f2_ax3.set_xlabel('Time(s)')
    f2_ax3.set_ylabel('Acceleration(m/s2)')
    
    f2_ax4 = fig2.add_subplot(spec2[1, 0],title='IMU_Gy_raw')
    plt.plot(t_seq, array_data1[:,3+FirstIndex])
    for i in range(7):
        plt.plot(t_seq, label_prep1[:,i]*200)
    f2_ax4.set_xlabel('Time(s)')
    f2_ax4.set_ylabel('Angular Velocity(degree/s)')
    
    f2_ax5 = fig2.add_subplot(spec2[1, 1],title='IMU_Gz_raw')
    plt.plot(t_seq, array_data1[:,4+FirstIndex])
    for i in range(7):
        plt.plot(t_seq, label_prep1[:,i]*200)
    f2_ax5.set_xlabel('Time(s)')
    f2_ax5.set_ylabel('Angular Velocity(degree/s)')
    
    f2_ax6 = fig2.add_subplot(spec2[1, 2],title='IMU_Gx_raw')
    plt.plot(t_seq, array_data1[:,5+FirstIndex])
    for i in range(7):
        plt.plot(t_seq, label_prep1[:,i]*200)
    f2_ax6.set_xlabel('Time(s)')
    f2_ax6.set_ylabel('Angular Velocity(degree/s)')
    
    fig2.legend(loc=4)
    fig2.suptitle(IMU_Loc,fontsize=16, x=0.4)
    savepath = r'C:\Users\Zed_Luz\OneDrive\3-MEE\21-NUS Lab Intern\Work\3-IMU-DeepLearning\Zeyu\1-Python Files\DataGraph\1-IMUs'
    fig2.savefig(os.path.join(savepath,IMU_Loc)+'.png')
#%%
def PlotGONIOs(GONIO_Loc,Array_data, Label_prep, Channel_catagories, fig_size = (25,15)):
    labelcategories = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent','Standing']
    array_data1 = Array_data
    label_prep1 = Label_prep
    cm = plt.get_cmap('gist_rainbow')
    FirstIndex = Channel_catagories.index(GONIO_Loc+'_Ankle')
    fig2 = plt.figure(figsize=fig_size,constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
    f2_ax1 = fig2.add_subplot(spec2[0, 0],title=GONIO_Loc+'_Knee_Angle_raw')
    f2_ax1.title.set_fontsize(16)
    plt.plot(t_seq, array_data1[:,1+FirstIndex],'k')
    for i in range(7):
        plt.plot(t_seq, label_prep1[:,i]*-100, label=labelcategories[i])
        plt.set_cmap(cm)
        plt.fill_between(t_seq, 0, label_prep1[:,i]*-100, alpha=0.3) 
    f2_ax1.set_xlabel('Time(s)')
    f2_ax1.set_ylabel('Acceleration(m/s2)')
    
    f2_ax2 = fig2.add_subplot(spec2[0, 1],title=GONIO_Loc+'_Ankle_Angle_raw')
    f2_ax2.title.set_fontsize(16)
    plt.plot(t_seq, array_data1[:,0+FirstIndex],'k')
    for i in range(7):
        plt.plot(t_seq, label_prep1[:,i]*20)
        plt.set_cmap(cm)
        plt.fill_between(t_seq, 0, label_prep1[:,i]*20, alpha=0.3)
    f2_ax2.set_xlabel('Time(s)')
    f2_ax2.set_ylabel('Acceleration(m/s2)')
    
    f2_ax4 = fig2.add_subplot(spec2[1, 0],title=GONIO_Loc+'_Knee_Velocity_raw')
    f2_ax4.title.set_fontsize(16)
    plt.plot(t_seq, array_data1[:,5+FirstIndex],'k')
    for i in range(7):
        plt.plot(t_seq, label_prep1[:,i]*450)
        plt.set_cmap(cm)
        plt.fill_between(t_seq, 0, label_prep1[:,i]*450, alpha=0.3)
    f2_ax4.set_xlabel('Time(s)')
    f2_ax4.set_ylabel('Angular Velocity(degree/s)')
    
    f2_ax5 = fig2.add_subplot(spec2[1, 1],title=GONIO_Loc+'_Ankle_Velocity_raw')
    f2_ax5.title.set_fontsize(16)
    plt.plot(t_seq, array_data1[:,4+FirstIndex],'k')
    for i in range(7):
        plt.plot(t_seq, label_prep1[:,i]*200)
        plt.set_cmap(cm)
        plt.fill_between(t_seq, 0, label_prep1[:,i]*200, alpha=0.3)
    f2_ax5.set_xlabel('Time(s)')
    f2_ax5.set_ylabel('Angular Velocity(degree/s)')
    
    fig2.legend(loc=4,ncol=2,title='Modes',fontsize=15)
    fig2.suptitle(GONIO_Loc+' Leg',fontsize=20)
    savepath = r'C:\Users\Zed_Luz\OneDrive\3-MEE\21-NUS Lab Intern\Work\3-IMU-DeepLearning\Zeyu\1-Python Files\DataGraph\2-GONIO'
    fig2.savefig(os.path.join(savepath,GONIO_Loc)+'.png')