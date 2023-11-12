#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 13:45:26 2023

@author: erynd
"""
import pandas as pd
import matplotlib.pyplot as plt

def plot_series_data(merged_data, series_id, start_date=None, end_date=None):
    """
    Plot ENMO data for a specific series_id with event markers.
    
    Parameters:
    merged_data (DataFrame): The merged DataFrame containing series data and events.
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
    plt.plot(filtered_series_data['timestamp'], filtered_series_data['enmo'], label='Raw data', color='blue')

    onset_data = filtered_series_data[filtered_series_data['event'] == 'onset']
    for _, row in onset_data.iterrows():
        plt.axvline(row['timestamp'], color='red', linestyle='--')

    wakeup_data = filtered_series_data[filtered_series_data['event'] == 'wakeup']
    for _, row in wakeup_data.iterrows():
        plt.axvline(row['timestamp'], color='green', linestyle='--')

    custom_legend = [plt.Line2D([0], [0], color='blue', label='Raw data'),
                     plt.Line2D([0], [0], color='red', linestyle='--', label='Onset'),
                     plt.Line2D([0], [0], color='green', linestyle='--', label='Wakeup')]

    plt.xlabel('Timestamp')
    plt.ylabel('ENMO')
    plt.title(f'Series_id: {series_id}')
    plt.grid(True)
    plt.legend(handles=custom_legend)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_series_data(cleaned_data, '03d92c9f6f8a', '2018-06-01', '2018-06-03')


