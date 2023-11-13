#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:45:11 2023

@author: lauracarlton
"""
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import os

def dataLoadClean():
    """
    This function loads and concatenates data from multiple parquet files and 
    returns a single merged dataframe.
    
    Returns:
        pd.DataFrame: Merged dataframe containing data from parquet files.
    """
    #directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    path1 = os.path.join(script_dir, 'Data', 'training_data_cleaned_part_1.parquet')
    path2 = os.path.join(script_dir, 'Data', 'training_data_cleaned_part_2.parquet')
    path3 = os.path.join(script_dir, 'Data', 'training_data_cleaned_part_3.parquet')
    path4 = os.path.join(script_dir, 'Data', 'training_data_cleaned_part_4.parquet')
    
    df1 = pd.read_parquet(path1, engine='pyarrow')
    df2 = pd.read_parquet(path2, engine='pyarrow')
    df3 = pd.read_parquet(path3, engine='pyarrow')    
    df4 = pd.read_parquet(path4, engine='pyarrow')

    df = pd.concat([df1, df2, df3, df4], ignore_index=True)

    return df
    


def plot(merged_data, column_name, series_id, start_date=None, end_date=None):
    """
    Plot a specific column data for a specific series_id with event markers.
    
    Parameters:
        merged_data (DataFrame): The merged DataFrame containing series data and events.
        column_name (str): The name of the column to plot.
        series_id (str): The series_id to filter the data by.
        start_date (str, optional): The start date for filtering the data. Format: 'YYYY-MM-DD'.
        end_date (str, optional): The end date for filtering the data. Format: 'YYYY-MM-DD'.
    """
    filtered_series_data = merged_data[merged_data['series_id'] == series_id]
    filtered_series_data['timestamp'] = pd.to_datetime(filtered_series_data['timestamp'])

    if start_date and end_date:
        filtered_series_data = filtered_series_data[(filtered_series_data['timestamp'] >= start_date) &
                                                    (filtered_series_data['timestamp'] <= end_date)]

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_series_data['timestamp'], filtered_series_data[column_name], label=f'{column_name}', color='blue')

    onset_data = filtered_series_data[filtered_series_data['event'] == 'onset']
    for _, row in onset_data.iterrows():
        plt.axvline(row['timestamp'], color='red', linestyle='--')

    wakeup_data = filtered_series_data[filtered_series_data['event'] == 'wakeup']
    for _, row in wakeup_data.iterrows():
        plt.axvline(row['timestamp'], color='green', linestyle='--')

    custom_legend = [plt.Line2D([0], [0], color='blue', label=f'{column_name}'),
                     plt.Line2D([0], [0], color='red', linestyle='--', label='Onset'),
                     plt.Line2D([0], [0], color='green', linestyle='--', label='Wakeup')]

    plt.xlabel('Timestamp')
    plt.ylabel(column_name)
    plt.title(f'Series_id: {series_id}')
    plt.grid(True)
    plt.legend(handles=custom_legend)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def low_pass_filter(data, column_name, cutoff_freq, filter_order, sampling_rate=0.2, series_id=None):
    """
    Apply a low-pass filter to a specific column of data and plot the results.

    Parameters:
        data (pandas.DataFrame): Input DataFrame containing the data.
        column_name (str): Name of the column to filter.
        cutoff_freq (float): Cutoff frequency for the low-pass filter.
        filter_order (int): Order of the Butterworth filter.
        sampling_rate (float, optional): Sampling rate of the data. Default is 0.2.
        series_id (str, optional): Series ID for filtering a specific subset of data. 
        Default is None.

    Returns:
        numpy.ndarray: Filtered data.
    """
    if series_id:
        data = data[data['series_id'] == series_id]

    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    signal_data = data[column_name].to_numpy()

    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, signal_data)
    
    return filtered_data


def fft(data, column_name, sampling_rate=0.2, series_id=None):
    """
    Perform Fast Fourier Transform (FFT) on a time-domain signal.

    Parameters:
        data (pandas.DataFrame): Input dataframe containing time-domain signals.
        column_name (str): Name of the column to perform FFT on.
        sampling_rate (float): Sampling rate of the input signals.
        series_id (str, optional): Series ID for filtering a specific subset of data. 
        Default is None.

    Returns:
        numpy.ndarray: Frequency values.
        numpy.ndarray: Magnitude spectrum.
    """
    if series_id:
        data = data[data['series_id'] == series_id]

    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    signal_data = data[column_name].to_numpy()

    n = len(signal_data)

    fft_result = np.fft.fft(signal_data)
    freq = np.fft.fftfreq(n, 1.0 / sampling_rate)
    
    magnitude_spectrum = np.abs(fft_result) / n

    positive_freq_mask = freq >= 0
    freq = freq[positive_freq_mask]
    magnitude_spectrum = magnitude_spectrum[positive_freq_mask]

    return freq, magnitude_spectrum


def rolling_average(data, window_size, column_name, series_id=None):
    """
    Apply a rolling average filter to a specific column of data.

    Parameters:
        data (pandas.DataFrame): Input DataFrame containing the data.
        window_size (int): Size of the moving average window.
        column_name (str): Name of the column to filter.
        series_id (str, optional): Series ID for filtering a specific subset of data. 
        Default is None.

    Returns:
        pandas.Series: Filtered data.
    """
    if series_id:
        data = data[data['series_id'] == series_id]

    if window_size <= 0 or window_size > len(data):
        raise ValueError("Invalid window size")

    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    filtered_data = data[column_name].rolling(window=window_size, min_periods=1).mean()

    return filtered_data

#%%
#Low pass filtering

df = dataLoadClean()

#%%
import os
import pandas as pd


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'Data')
file_paths = [
    os.path.join(data_dir, 'training_data_cleaned_part_1.parquet'),
    os.path.join(data_dir, 'training_data_cleaned_part_2.parquet'),
    os.path.join(data_dir, 'training_data_cleaned_part_3.parquet'),
    os.path.join(data_dir, 'training_data_cleaned_part_4.parquet')
]

for file_path in file_paths:
    df = pd.read_parquet(file_path, engine='pyarrow')
    
    df['enmo_lpf'] = low_pass_filter(df, column_name='enmo', cutoff_freq=0.01, filter_order=2)
    df['anglez_lpf'] = low_pass_filter(df, column_name='anglez', cutoff_freq=0.01, filter_order=2)
    
    df.to_parquet(file_path, engine='pyarrow', index=False)
               
                      
