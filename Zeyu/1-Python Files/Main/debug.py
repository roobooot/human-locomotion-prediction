# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 08:55:21 2018

@author: Zed_Luz
"""

import model

a= model.DATA(SplitData=True, ReadALLSubjs=False)
a.scandata(Subj=[1])
a.stackdata()
a.PreprocessData()
a.DisplayData()
