#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:03:37 2023

@author: erynd
"""

import pandas as pd

file_path = '/Users/erynd/Library/CloudStorage/OneDrive-Personal/Grad School/Deep Learning/Project/train_series.parquet'

df = pd.read_parquet(file_path)

#%%

print("First 5 rows of train_series.parquet:")
print(df.head())

#%%

csv_file_path = '/Users/erynd/Library/CloudStorage/OneDrive-Personal/Grad School/Deep Learning/Project/train_events.csv'

csv_df = pd.read_csv(csv_file_path)

print("Top 5 rows of train_events.csv:")
print(csv_df.head())


#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#do we need to preprocess the data? 

#binary classification (we want onset and wakeup)
label_mapping = {'wakeup': 0, 'onset': 1}
csv_df['event'] = csv_df['event'].map(label_mapping)

#common key merge via series_id
merged_data = pd.merge(df, csv_df, on=['series_id', 'step'], how='inner')

X = merged_data[['anglez', 'enmo']]
y = merged_data['event']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1, activation='sigmoid'))  #binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)

accuracy = model.evaluate(X_test, y_test)[1]
print(f'Test Accuracy: {accuracy}')
#Test Accuracy: 0.6025038957595825

#predictions = model.predict(test_data)

#%%

from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import tensorflow as tf

def create_lstm_model(units=50, learning_rate=0.001, batch_size=32, epochs=10):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=units, activation='relu', input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_lstm_model, verbose=0)

#gridsearch!
param_grid = {
    'units': [50, 100, 150],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 30]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

best_params = grid_result.best_params_
best_model = grid_result.best_estimator_

#train the best model on the full training data will give best results I believe
best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'])
test_accuracy = best_model.score(X_test, y_test)

results = grid_result.cv_results_
param_combinations = results['params']
mean_test_scores = results['mean_test_score']

plt.figure(figsize=(12, 6))
plt.scatter(range(len(param_combinations)), mean_test_scores, marker='o')
plt.xticks(range(len(param_combinations)), [str(param) for param in param_combinations], rotation=90)
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Mean Test Score (Accuracy)')
plt.title('Grid Search Results')
plt.grid()
plt.show()

print(f'Best Hyperparameters: {best_params}')
print(f'Test Accuracy of the Best Model: {test_accuracy}')

