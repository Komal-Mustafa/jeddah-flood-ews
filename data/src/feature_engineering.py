# File: data/src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def create_features(df):
    print('Creating features...')
    df = df.copy()

    df['hour']        = df['date'].dt.hour
    df['month']       = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['hour_sin']    = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']    = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12)

    df['precip_1hr']    = df['precipitation'].rolling(1,  min_periods=1).sum()
    df['precip_3hr']    = df['precipitation'].rolling(3,  min_periods=1).sum()
    df['precip_6hr']    = df['precipitation'].rolling(6,  min_periods=1).sum()
    df['precip_12hr']   = df['precipitation'].rolling(12, min_periods=1).sum()
    df['precip_24hr']   = df['precipitation'].rolling(24, min_periods=1).sum()
    df['precip_48hr']   = df['precipitation'].rolling(48, min_periods=1).sum()
    df['precip_rate']   = df['precipitation'].diff().fillna(0).clip(lower=0)
    df['precip_max3hr'] = df['precipitation'].rolling(3).max()

    rain_occurred = (df['precipitation'] > 0.1).astype(int)
    consecutive_dry = []
    count = 0
    for val in rain_occurred:
        if val == 0:
            count += 1
        else:
            count = 0
        consecutive_dry.append(count)
    df['dry_hours_before'] = consecutive_dry

    df['cape_category'] = pd.cut(df['cape'],
        bins=[-1, 500, 1500, 3000, 99999],
        labels=[0, 1, 2, 3]).astype(float)

    df['soil_moisture_avg'] = df[[
        'soil_moisture_0_1cm', 'soil_moisture_1_3cm', 'soil_moisture_3_9cm'
    ]].mean(axis=1)

    df['high_risk_month'] = df['month'].isin([11, 12, 1, 2]).astype(int)

    print(f'Created {len(df.columns)} total features')
    return df

def prepare_sequences(df, seq_length=24):
    print(f'Creating sequences of length {seq_length}...')

    feature_cols = [
        'precip_1hr', 'precip_3hr', 'precip_6hr', 'precip_12hr',
        'precip_24hr', 'precip_rate', 'precip_max3hr',
        'cape', 'cape_category', 'soil_moisture_avg',
        'relativehumidity_2m', 'surface_pressure', 'windspeed_10m',
        'temperature_2m', 'dewpoint_2m',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'dry_hours_before', 'high_risk_month'
    ]

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print('Scaler saved to models/scaler.pkl')

    data = df[feature_cols].values
    labels = df['flood_label'].values

    n = len(df) - seq_length
    X = np.lib.stride_tricks.sliding_window_view(data, (seq_length, data.shape[1])).squeeze(1).astype(np.float32)
    y = labels[seq_length:].astype(np.float32)
    print(f'X shape: {X.shape}  (samples, time_steps, features)')
    print(f'y shape: {y.shape}')
    return X, y, feature_cols

if __name__ == '__main__':
    df = pd.read_csv('data/processed/labeled_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    os.makedirs('models', exist_ok=True)

    df_feat = create_features(df)
    X, y, cols = prepare_sequences(df_feat, seq_length=24)

    np.save('data/processed/X.npy', X)
    np.save('data/processed/y.npy', y)
    print('Features saved to data/processed/')