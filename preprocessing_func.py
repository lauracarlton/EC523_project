#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:45:11 2023

@author: lauracarlton
"""

import pandas as pd
import numpy as np 

def dataLoad(path_root):
        
    csv_path = path_root + 'train_events.csv'
    parquet_path = path_root + 'train_series.parquet'
    
    df = pd.read_parquet(parquet_path, engine = 'pyarrow')
    
    
    csv_df = pd.read_csv(csv_path)
    
    return csv_df, df

def findBadData(csv_df):
    
    #TODO - are we removing these bad cases right now?
    
    #finding cases where there is an 'onset' but no 'wakeup'
    bad_cases = []

    for i in range(1, len(csv_df)):
        prev_row = csv_df.iloc[i - 1]
        current_row = csv_df.iloc[i]

        if prev_row['event'] == 'onset' and current_row['event'] == 'wakeup' and str(prev_row['step']) != 'nan' and str(current_row['step']) == 'nan':
            bad_cases.append((prev_row['series_id'], prev_row['step'], current_row['step']))


    #finding cases where there is a 'wakeup' but no prior 'onset'
    bad_cases2 = []
    
    for i in range(1, len(csv_df)):
        prev_row = csv_df.iloc[i - 1]
        current_row = csv_df.iloc[i]
    
        if prev_row['event'] == 'onset' and current_row['event'] == 'wakeup' and str(prev_row['step']) == 'nan' and str(current_row['step']) != 'nan':
            bad_cases2.append((prev_row['series_id'], prev_row['step'], current_row['step']))

    pass
        

def filterData():
    pass
    
def dataPreProcess(path_root, ):

    csv_df, df = dataLoad(path_root)
    merged_data = pd.merge(df, csv_df, on=['series_id', 'step'], how='left')
    
    #formating the data, idk this takes forever to run but theoretically this should work

    # merged_data = merged_data.drop('timestamp_y')
    merged_data['timestamp'] = merged_data['timestamp_x']
    merged_data.loc[merged_data['event']=='onset','event'] = -1
    merged_data.loc[merged_data['event']=='wakeup','event'] = 1
    
    findBadData(csv_df)
    
    merged_data['enmo_clipped'] = np.clip(merged_data['enmo'], a_min=None, a_max=1)

    #Formatting the data better to all the same size and into a matrix ??

    # specific_ids = ['038441c925bb', 'f8a8da8bdd00']  #list of specific series IDs (for testing), need to change to those you want
    # data = merged_data[merged_data['series_id'].isin(specific_ids)]

    # data_enmo = merged_data.pivot(index='series_id', columns='step', values='enmo')
    # data_enmo.fillna(0, inplace=True)

    # data_anglez = merged_data.pivot(index='series_id', columns='step', values='anglez')
    # data_anglez.fillna(0, inplace=True)

    # data_time = merged_data.pivot(index='series_id', columns='step', values='anglez')
    # data_anglez.fillna(0, inplace=True)
        
    data_matrix = pd.DataFrame(index=merged_data.index)
    data_matrix['enmo'] = merged_data['enmo']
    data_matrix['anglez'] = merged_data['anglez']
    data_matrix['timestamp'] = merged_data['timestamp']

    data_matrix.reset_index(inplace=True)
    data_matrix.columns.name = None
    
    
    return data_matrix
    
    



