#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 13:45:26 2023

@author: erynd
"""
#Euclidean Norm Minus One
import pandas as pd

file_path = '/Users/erynd/Library/CloudStorage/OneDrive-Personal/Grad School/Deep Learning/Project/train_series.parquet'

df = pd.read_parquet(file_path)

#%%

print("First 5 rows of train_series.parquet:")
print(df.head())

#%%

csv_file_path = '/Users/erynd/Library/CloudStorage/OneDrive-Personal/Grad School/Deep Learning/Project/train_events.csv'

csv_df = pd.read_csv(csv_file_path)

print("First 5 rows of train_events.csv:")
print(csv_df.head())

#%%

merged_data = pd.merge(df, csv_df, on=['series_id', 'step'], how='left')

#%%
#plot based on specific id

import matplotlib.pyplot as plt

specific_id = '038441c925bb'
filtered_series_data = merged_data[merged_data['series_id'] == specific_id]

filtered_series_data['timestamp_x'] = pd.to_datetime(filtered_series_data['timestamp_x'])

# start_date = '2018-06-01'
# end_date = '2018-06-03'
# filtered_date_range = filtered_series_data[(filtered_series_data['timestamp_x'] >= start_date) & (filtered_series_data['timestamp_x'] <= end_date)]

plt.figure(figsize=(12, 6))

plt.plot(filtered_series_data['timestamp_x'], filtered_series_data['enmo'], label='Raw data', color='blue')

onset_data = filtered_series_data[filtered_series_data['event'] == 'onset']
for _, row in onset_data.iterrows():
    plt.axvline(row['timestamp_x'], color='red', linestyle='--')

wakeup_data = filtered_series_data[filtered_series_data['event'] == 'wakeup']
for _, row in wakeup_data.iterrows():
    plt.axvline(row['timestamp_x'], color='green', linestyle='--')

custom_legend = [plt.Line2D([0], [0], color ='blue', label='Raw data'),
                 plt.Line2D([0], [0], color='red', linestyle='--', label='Onset'),
                 plt.Line2D([0], [0], color='green', linestyle='--', label='Wakeup')]


plt.xlabel('Timestamp')
plt.ylabel('ENMO')
plt.title(f'Series_id: {specific_id}')
plt.grid(True)
plt.legend(handles=custom_legend)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


