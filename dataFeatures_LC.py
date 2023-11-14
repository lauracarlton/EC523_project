#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 20:25:07 2023

@author: lauracarlton
"""

import preprocessing_func as PPF
import pandas as pd
import numpy as np 
from tqdm import tqdm 
import torch 
# file_root = '/Users/lauracarlton/Documents/child-mind-institute-detect-sleep-states/'

data_matrix = PPF.dataLoadClean()


#%% other features to add

'''
- add in first difference 
'''

data_matrix['diff_enmo'] = np.hstack([0, np.diff(data_matrix['enmo_lpf'].to_numpy())])
data_matrix['diff_anglez'] = np.hstack([0, np.diff(data_matrix['anglez_lpf'].to_numpy())])

#%%

data2Window = data_matrix[['timestamp', 'enmo_lpf', 'anglez_lpf', 'diff_enmo', 'diff_anglez', 'event', 'series_id']]

#%% window the data 
windowSize = 12

overlap = 6
nWindows = int(np.floor(data_matrix.shape[0]/windowSize))
labels = np.zeros(nWindows)
idx1 = 0
asleep = 0
WINDOWS = []
for w in tqdm(range(nWindows)):
    
    
    window = data2Window.loc[w:w+windowSize-1]
    ids = window.series_id.unique()
    if len(ids) > 1:
        continue 
    
    events = window.event.unique()
    if 'onest' in events:
        labels[w] = 1
        alseep = 1
        
    elif 'wakeup' in events:
        labels[w] = 3
        asleep = 0
        
    elif asleep == 0:
        labels[w] = 2

    elif asleep == 1:
        labels[w] = 0

    window.drop(columns=['event', 'series_id'], inplace=True)    
    
    window['timestamp'] = pd.to_datetime(window['timestamp'])
    window = window.to_numpy()
    WINDOWS.append(window)
    

    
#%% convert list of dataframes to tensor

data_tensor = torch.tensor(WINDOWS)


### NEED TO SAVE ###
# data_tensor.save()


