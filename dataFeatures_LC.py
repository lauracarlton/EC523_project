#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 20:25:07 2023

@author: lauracarlton
"""

import preprocessing_func as PPF
import pandas as pd
from datetime import datetime 
from tqdm import tqdm 

file_root = '/Users/lauracarlton/Documents/child-mind-institute-detect-sleep-states/'

dataMatrix = PPF.dataPreProcess(file_root)


#%% other features to add

'''
want to add time features
want to add rolling avg, max and std 
- add these to the data matrix 
'''
data_matrix['timestamp'] = pd.to_datetime(data_matrix['timestamp'])




#%%
(.col('timestamp').str.to_datetime().dt.year()-2000).cast(pd.UInt8).alias('year'), 
pd.col('timestamp').str.to_datetime().dt.month().cast(pd.UInt8).alias('month'),
pd.col('timestamp').str.to_datetime().dt.day().cast(pd.UInt8).alias('day'), 
pd.col('timestamp').str.to_datetime().dt.hour().cast(pd.UInt8).alias('hour')



#%% window the data 
windowSize = 12

overlap = 6
nWindows = int(np.floor(data_matrix.shape[0]/windowSize))
idx1 = 0

WINDOWS = []
for w in tqdm(range(nWindows)):
    
    
    window = data_matrix.loc[w:w+windowSize]
    WINDOWS.append(window)
    
    
    
    
    






