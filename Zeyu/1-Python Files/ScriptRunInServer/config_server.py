# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:55:49 2018

@author: Zed_Luz
"""
import numpy as np

class Config(object):
    WINDOW_SIZE = 100
    STEP_SIZE = 50
    STEPS_PER_EPOCH = 1000
    LEARNING_RATE = 0.001
    MAXABS_PREPROCESS = False
    SPILTRATIO = 0.33
    LABELCATEGORIES = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent',
                       'Standing']
    INDEX_IMU = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29]
    INDEX_EMG = [30,31,32,33,34,35,36,37,38,39,40,41,42,43]
    INDEX_GONIO = [44, 45, 46, 47, 48, 49, 50, 51]
    
    DATAPATH = '../../../../../zhangkuangen/Zeyu/Dataset/'
    def __init__(self):
        self.MAXABS_PREPROCESS = False
    def displayConfig(self):
        print('\nConfiguration:')
        for a in dir(self):
            if not a.startswith("_") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
