# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 08:55:21 2018

@author: Zed_Luz
"""

import model

a= model.DATA(SplitData=0.33, ReadALL=False)
a.scandata(Subj=[1,2])
a.displayConfig()
a.stackdata()
a.PreprocessData()
a.DisplayData()
