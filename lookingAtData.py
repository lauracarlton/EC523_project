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

specific_id = '03d92c9f6f8a'
filtered_series_data = merged_data[merged_data['series_id'] == specific_id]

filtered_series_data['timestamp_x'] = pd.to_datetime(filtered_series_data['timestamp_x'])

start_date = '2018-06-01'
end_date = '2018-06-03'
filtered_date_range = filtered_series_data[(filtered_series_data['timestamp_x'] >= start_date) & (filtered_series_data['timestamp_x'] <= end_date)]

plt.figure(figsize=(12, 6))
plt.plot(filtered_date_range['timestamp_x'], filtered_date_range['enmo'], label='ENMO')
plt.xlabel('Timestamp')
plt.ylabel('ENMO')
plt.title(f'ENMO vs. Timestamp for series_id: {specific_id} (June 1st to June 3rd)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



