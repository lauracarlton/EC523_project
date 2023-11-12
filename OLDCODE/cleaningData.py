#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:27:21 2023

@author: erynd
"""
import pandas as pd

file_path = '/Users/lauracarlton/Documents/child-mind-institute-detect-sleep-states/train_series.parquet'
df = pd.read_parquet(file_path)

# csv_file_path = '/Users/erynd/Library/CloudStorage/OneDrive-Personal/Grad School/Deep Learning/Project/train_events.csv'
csv_file_path = '/Users/lauracarlton/Documents/child-mind-institute-detect-sleep-states/train_events.csv'

csv_df = pd.read_csv(csv_file_path)

merged_data = pd.merge(df, csv_df, on=['series_id', 'step'], how='left')



merged_data = merged_data.drop('timestamp_y')
merged_data['timestamp'] = merged_data['timestamp_x']
merged_data.loc[merged_data['event']=='onset','event'] = -1
merged_data.loc[merged_data['event']=='wakeup','event'] = 1

bad_data = ['fb223ed2278c',
 '062cae666e2a',
 'c7b1283bb7eb',
 'efbfc4526d58',
 '2654a87be968',
 '390b487231ce',
 '137771d19ca2',
 'ce85771a714c',
 '89c7daa72eee',
 'e11b9d69f856',
 'a2b0a64ec9cf',
 'f564985ab692',
 '8898e6db816d',
 '0f9e60a8e56d',
 'a3e59c2ce3f6',
 'ba8083a2c3b8',
 '703b5efa9bc1',
 '44a41bba1ee7',
 'd5be621fd9aa',
 'f56824b503a0',
 'f8a8da8bdd00',
 'f981a0805fd0',
 '1e6717d93c1d',
 '9277be28a1cf',
 '03d92c9f6f8a',
 'cfeb11428dd7',
 '7476c0bd18d2',
 '3be2f86c3e45',
 'c5d08fc3e040',
 '0ce74d6d2106',
 '854206f602d0',
 'aed3850f65f0',
 '5aad18e7ce64',
 '154fe824ed87',
 'e4500e7e19e1',
 '73fb772e50fb',
 'c5365a55ebb7',
 '78569a801a38',
 'e30cb792a2bc',
 '2fc653ca75c7',
 'cf13ed7e457a',
 'b7fc34995d0f',
 '405df1b41f9f',
 'c107b5789660']

cleaned_data = merged_data[~merged_data['series_id'].isin(bad_data)]

cleaned_data.to_csv('training_data_cleaned.csv', index=False)

#127946340
