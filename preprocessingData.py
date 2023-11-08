#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:39:44 2023

@author: erynd
"""
#uploading data, test
import pandas as pd

file_path = '/Users/erynd/Library/CloudStorage/OneDrive-Personal/Grad School/Deep Learning/Project/train_series.parquet'
df = pd.read_parquet(file_path)

csv_file_path = '/Users/erynd/Library/CloudStorage/OneDrive-Personal/Grad School/Deep Learning/Project/train_events.csv'
csv_df = pd.read_csv(csv_file_path)

merged_data = pd.merge(df, csv_df, on=['series_id', 'step'], how='left')


#%%
#formating the data, idk this takes forever to run but theoretically this should work

merged_data = merged_data.drop('timestamp_y')
merged_data['timestamp'] = merged_data['timestamp_x']
merged_data.loc[merged_data['event']=='onset','event'] = -1
merged_data.loc[merged_data['event']=='wakeup','event'] = 1

#%%

#finding what the data looks like, and how much is actually labeled vs 'nan'

nan_count = csv_df[csv_df['step'].astype(str) == 'nan'].groupby('series_id').size()
total_count = csv_df['series_id'].value_counts()
nan_percentage = (nan_count / total_count) * 100

import matplotlib.pyplot as plt

plt.hist(nan_percentage, bins=100)

plt.xlabel('Percentage of NaN')
plt.ylabel('Frequency')
plt.title('Distribution of Percentage of NaN Values per Series ID')

plt.show()

over_percentage = nan_percentage[nan_percentage > 75].index.tolist()
#Potentially we should get rid of a certain percentage of series_ids that contain more than 75% 'nan'
#there are 40 out of 277 series_ids that have more than 75% just none-labeled data

#I think it can learn from time points of no labels (ie 'not in use'), but if there 
#are too many periods of this I think it'll mess with the model


#%%

#finding cases where there is an 'onset' but no 'wakeup'
bad_cases = []

for i in range(1, len(csv_df)):
    prev_row = csv_df.iloc[i - 1]
    current_row = csv_df.iloc[i]

    if prev_row['event'] == 'onset' and current_row['event'] == 'wakeup' and str(prev_row['step']) != 'nan' and str(current_row['step']) == 'nan':
        bad_cases.append((prev_row['series_id'], prev_row['step'], current_row['step']))

for series_id, onset_step, wakeup_step in bad_cases:
    print("Series ID:", series_id)
    print("Onset Step:", onset_step)
    print("Wakeup Step:", wakeup_step)
    print()

# Series ID: 0ce74d6d2106
# Onset Step: 332376.0
# Wakeup Step: nan

#%%
#finding cases where there is a 'wakeup' but no prior 'onset'
bad_cases2 = []

for i in range(1, len(csv_df)):
    prev_row = csv_df.iloc[i - 1]
    current_row = csv_df.iloc[i]

    if prev_row['event'] == 'onset' and current_row['event'] == 'wakeup' and str(prev_row['step']) == 'nan' and str(current_row['step']) != 'nan':
        bad_cases2.append((prev_row['series_id'], prev_row['step'], current_row['step']))

for series_id, onset_step, wakeup_step in bad_cases2:
    print("Series ID:", series_id)
    print("Onset Step:", onset_step)
    print("Wakeup Step:", wakeup_step)
    print()

# Series ID: 154fe824ed87
# Onset Step: nan
# Wakeup Step: 514980.0

# Series ID: 44a41bba1ee7
# Onset Step: nan
# Wakeup Step: 165684.0

# Series ID: efbfc4526d58
# Onset Step: nan
# Wakeup Step: 114864.0

# Series ID: f8a8da8bdd00
# Onset Step: nan
# Wakeup Step: 291384.0

#%%
#making sure that each night has only 2 events (and not 1 or 3/more events)

event_counts = csv_df.groupby(['series_id', 'night'])['event'].nunique()
event_counts = event_counts.reset_index()
event_counts.columns = ['series_id', 'night', 'event_count']
event_bad = event_counts[event_counts['event_count'] != 2]

print("Rows where 'event_count' is not 2:")
print(event_bad)

#all seem good

#%%
#scatter plot of enmo values
#see if we want to clip the values, someone mentioned that it helped slightly

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(range(len(merged_data['enmo'])), merged_data['enmo'])
plt.title('Scatter Plot for ENMO Outlier Detection')
plt.xlabel('Data Point')
plt.ylabel('ENMO')
plt.show()

#%%

#seeing how much of the data is above the threshold

threshold = 1
above_threshold_count = (merged_data['enmo'] > threshold).sum()
total_count = len(merged_data)
percentage_above_threshold = (above_threshold_count / total_count) * 100

print(f"Percentage of 'enmo' values above {threshold}: {percentage_above_threshold:.2f}%")

#%%

#if we want to clip the values to a certain threshold
import numpy as np

merged_data['enmo_clipped'] = np.clip(merged_data['enmo'], a_min=None, a_max=threshold)

#%%

#Formatting the data better to all the same size and into a matrix ??

specific_ids = ['038441c925bb', 'f8a8da8bdd00']  #list of specific series IDs (for testing), need to change to those you want
data = merged_data[merged_data['series_id'].isin(specific_ids)]

data_enmo = data.pivot(index='series_id', columns='step', values='enmo')
data_enmo.fillna(0, inplace=True)

data_anglez = data.pivot(index='series_id', columns='step', values='anglez')
data_anglez.fillna(0, inplace=True)

data_matrix = pd.DataFrame(index=data_enmo.index)
data_matrix['enmo'] = [data_enmo.iloc[i].values for i in range(data_enmo.shape[0])]
data_matrix['anglez'] = [data_anglez.iloc[i].values for i in range(data_anglez.shape[0])]

data_matrix.reset_index(inplace=True)
data_matrix.columns.name = None

#%%

#low pass filtering of the data

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, savgol_filter

specific_id = '038441c925bb'
data = merged_data[merged_data['series_id'] == specific_id]
data = data['enmo'][:2000] #only want to look at a small amount, or else it'll take forever

cutoff_frequency = 0.1

def low_pass_filter(data, cutoff_freq, sampling_rate):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)


plt.plot(data, label='Original Data', alpha=0.5)
plt.plot(low_pass_filter(data, cutoff_frequency, sampling_rate=1.0), label='Low-Pass Filter', alpha=0.8)
plt.legend()
plt.title('Low-Pass Filter')

#I think low pass filtering works great, should apply it to all the data (enmo and anglez)

