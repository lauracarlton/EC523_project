#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaning up the data and saving it to another variable. 
"""
import pandas as pd

file_path = '/Users/erynd/Library/CloudStorage/OneDrive-Personal/Grad School/Deep Learning/Project/train_series.parquet'
df = pd.read_parquet(file_path)

csv_file_path = '/Users/erynd/Library/CloudStorage/OneDrive-Personal/Grad School/Deep Learning/Project/train_events.csv'

csv_df = pd.read_csv(csv_file_path)

merged_data = pd.merge(df, csv_df, on=['series_id', 'step'], how='left')

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
#there are 40 out of 277 series_ids that have more than 75% just none-labeled data
#therefore, we are trimming these

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

#total of 43 bad cases, trimming them below. These are saved as "clean data"

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

#%%

#to be able to load onto the git the cleaned data was split into four files

split_size = len(df) // 4

for i in range(4):
    start = i * split_size
    end = None if i == 3 else start + split_size
    split_df = df.iloc[start:end]
    split_df.to_parquet(f'training_data_cleaned_part_{i+1}.parquet')


