# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:55:49 2018

@author: Zed_Luz
"""
import numpy as np

class Config(object):
    WINDOW_SIZE = 100
    STEP_SIZE = 10
    STEPS_PER_EPOCH = 1000
    LEARNING_RATE = 0.001
    MAXABS_PREPROCESS = False
    SPILTRATIO = 0.33
    LABELCATEGORIES = ['Sitting','Level Ground Walking','Ramp Ascent','Ramp Descent','Stair Ascent','Stair Descent',
                       'Standing']
    DATAPATH = r'C:\Users\Zed_Luz\OneDrive - 南方科技大学\BigScaleFiles\1-datasets for bilateral lower limb neuromechanical signals\2-Data'
    def __init__(self):
        self.MAXABS_PREPROCESS = False
    def display(self):
        print('\nConfiguration:')
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
