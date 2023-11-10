#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:53:58 2023

@author: lauracarlton
"""

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

# from metric import score 

column_names = {
    'series_id_column_name': 'series_id',
    'time_column_name': 'step',
    'event_column_name': 'event',
    'score_column_name': 'score',
}

tolerances = {
    'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360], 
    'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
}

# Process complete.
print("Setup Complete.")

#%%

data_transforms = [
    pl.col('timestamp').str.to_datetime(), 
    (pl.col('timestamp').str.to_datetime().dt.year()-2000).cast(pl.UInt8).alias('year'), 
    pl.col('timestamp').str.to_datetime().dt.month().cast(pl.UInt8).alias('month'),
    pl.col('timestamp').str.to_datetime().dt.day().cast(pl.UInt8).alias('day'), 
    pl.col('timestamp').str.to_datetime().dt.hour().cast(pl.UInt8).alias('hour')
]

additional_transforms = [
    pl.col('anglez').cast(pl.Int16), # Casting anglez to 16 bit integer
    (pl.col('enmo')*1000).cast(pl.UInt16), # Convert enmo to 16 bit uint
]

train_series = pl.scan_parquet('/Users/lauracarlton/Documents/child-mind-institute-detect-sleep-states/train_series.parquet').with_columns(
    data_transforms + additional_transforms
    )

train_events = pl.read_csv('/Users/lauracarlton/Documents/child-mind-institute-detect-sleep-states/train_events.csv').with_columns(
    data_transforms
    ).drop_nulls()

test_series = pl.scan_parquet('/Users/lauracarlton/Documents/child-mind-institute-detect-sleep-states/test_series.parquet').with_columns(
    data_transforms + additional_transforms
    )

# Removing null events and nights with mismatched counts from series_events
mismatches = train_events.drop_nulls().group_by(['series_id', 'night']).agg([
    ((pl.col('event') == 'onset').sum() == (pl.col('event') == 'wakeup').sum()).alias('balanced')
    ]).sort(by=['series_id', 'night']).filter(~pl.col('balanced'))

for mm in mismatches.to_numpy(): 
    train_events = train_events.filter(~((pl.col('series_id') == mm[0]) & (pl.col('night') == mm[1])))

# Getting series ids as a list for convenience
series_ids = train_events['series_id'].unique(maintain_order=True).to_list()

# Updating train_series to only keep these series ids
train_series = train_series.filter(pl.col('series_id').is_in(series_ids))

# Process complete.
print("Process Complete.")

#%%

features = [pl.col('hour')]
feature_cols = ['hour']

for minute in [5, 30, 120, 480]:
    for col in ['enmo', 'anglez']:
        features += [
            pl.col(col).rolling_mean(12 * minute, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'{col}_{minute}-mean'),
            pl.col(col).rolling_max(12 * minute, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'{col}_{minute}-max'),
            pl.col(col).rolling_std(12 * minute, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'{col}_{minute}-std')
        ]
        feature_cols += [
            f'{col}_{minute}-mean',
            f'{col}_{minute}-max',
            f'{col}_{minute}-std'
        ]
        features += [
            (pl.col(col).diff().abs().rolling_mean(12 * minute, center=True, min_periods=1) * 10).abs().cast(pl.UInt32).alias(f'{col}_v1_{minute}-mean'),
            (pl.col(col).diff().abs().rolling_max(12 * minute, center=True, min_periods=1) * 10).abs().cast(pl.UInt32).alias(f'{col}_v1_{minute}-max'),
            (pl.col(col).diff().abs().rolling_std(12 * minute, center=True, min_periods=1) * 10).abs().cast(pl.UInt32).alias(f'{col}_v1_{minute}-std')
        ]
        feature_cols += [
            f'{col}_v1_{minute}-mean',
            f'{col}_v1_{minute}-max',
            f'{col}_v1_{minute}-std'
        ]
cols = ['series_id', 'step', 'timestamp']
train_series = train_series.with_columns(features).select(cols + feature_cols)
test_series = test_series.with_columns(features).select(cols + feature_cols)

# Process complete.
print("Process Complete.")

#%%

def make_train_dataset(train_data, train_events, drop_nulls=False) :
    
    series_ids = train_data['series_id'].unique(maintain_order=True).to_list()
    X, y = pl.DataFrame(), pl.DataFrame()
    for idx in tqdm(series_ids) : 
        
        # Normalizing sample features
        sample = train_data.filter(pl.col('series_id')==idx).with_columns(
            [(pl.col(col) / pl.col(col).std()).cast(pl.Float32) for col in feature_cols if col != 'hour']
        )
        
        events = train_events.filter(pl.col('series_id')==idx)
        
        if drop_nulls : 
            # Removing datapoints on dates where no data was recorded
            sample = sample.filter(
                pl.col('timestamp').dt.date().is_in(events['timestamp'].dt.date())
            )
        
        X = X.vstack(sample[cols + feature_cols])

        onsets = events.filter((pl.col('event') == 'onset') & (pl.col('step') != None))['step'].to_list()
        wakeups = events.filter((pl.col('event') == 'wakeup') & (pl.col('step') != None))['step'].to_list()

        # NOTE: This will break if there are event series without any recorded onsets or wakeups
        y = y.vstack(sample.with_columns(
            sum([(onset <= pl.col('step')) & (pl.col('step') <= wakeup) for onset, wakeup in zip(onsets, wakeups)]).cast(pl.Boolean).alias('asleep')
            ).select('asleep')
            )
    
    y = y.to_numpy().ravel()
    
    return X, y

# Process complete.
print("Process Complete.")

#%%

def get_events(series, classifier):
    series_ids = series['series_id'].unique(maintain_order=True).to_list()
    events = pl.DataFrame(schema={'series_id': str, 'step': int, 'event': str, 'score': float})

    for idx in tqdm(series_ids):

        # Collecting sample and normalizing features
        scale_cols = [col for col in feature_cols if (col != 'hour') and (series[col].std() != 0)]
        X = series.filter(pl.col('series_id') == idx).select(cols + feature_cols).with_columns(
            [(pl.col(col) / series[col].std()).cast(pl.Float32) for col in scale_cols]
        )

        # Applying classifier to get predictions and scores
        preds, probs = classifier.predict(X[feature_cols]), classifier.predict_proba(X[feature_cols])[:, 1]

        #NOTE: Considered using rolling max to get sleep periods excluding <30 min interruptions, but ended up decreasing performance
        X = X.with_columns(
            pl.lit(preds).cast(pl.Int8).alias('prediction'), 
            pl.lit(probs).alias('probability')
                        )
        
        # Getting predicted onset and wakeup time steps
        pred_onsets = X.filter(X['prediction'].diff() > 0)['step'].to_list()
        pred_wakeups = X.filter(X['prediction'].diff() < 0)['step'].to_list()

        if len(pred_onsets) > 0:

            # Ensuring all predicted sleep periods begin and end
            if min(pred_wakeups) < min(pred_onsets):
                pred_wakeups = pred_wakeups[1:]

            if max(pred_onsets) > max(pred_wakeups):
                pred_onsets = pred_onsets[:-1]

            # Keeping sleep periods longer than 30 minutes
            sleep_periods = [(onset, wakeup) for onset, wakeup in zip(pred_onsets, pred_wakeups) if
                             wakeup - onset >= 12 * 30]

            for onset, wakeup in sleep_periods:
                # Scoring using mean probability over the period
                score = X.filter((pl.col('step') >= onset) & (pl.col('step') <= wakeup))['probability'].mean()

                # Adding sleep event to the dataframe
                events = events.vstack(pl.DataFrame().with_columns(
                    pl.Series([idx, idx]).alias('series_id'),
                    pl.Series([onset, wakeup]).alias('step'),
                    pl.Series(['onset', 'wakeup']).alias('event'),
                    pl.Series([score, score]).alias('score')
                ))

    # Adding row id column
    events = events.to_pandas().reset_index().rename(columns={'index': 'row_id'})

    return events

# Process complete.
print("Process Complete.")

#%%

train_data = train_series.filter(pl.col('series_id').is_in(series_ids)).take_every(12 * 5).collect()

# Process complete.
print("Process Complete.")


X_train, y_train = make_train_dataset(train_data, train_events)




