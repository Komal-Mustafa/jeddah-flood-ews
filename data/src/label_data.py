# File: data/src/label_data.py
# PURPOSE: Combines climate data with real flood event labels
# RUN: python data/src/label_data.py

import pandas as pd
import numpy as np

def load_and_label():
    print('Loading raw climate data...')
    df = pd.read_csv('data/raw/jeddah_climate_raw.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Load flood events
    floods = pd.read_csv('data/raw/flood_events.csv', comment='#')
    floods['date'] = pd.to_datetime(floods['date'])

    # Add flood label columns
    df['flood_label'] = 0
    df['flood_severity'] = 0

    # For each known flood event, label the surrounding 12 hours
    for _, event in floods.iterrows():
        start = event['date'] - pd.Timedelta(hours=6)
        end   = event['date'] + pd.Timedelta(hours=6)
        mask  = (df['date'] >= start.tz_localize('Asia/Riyadh')) & (df['date'] <= end.tz_localize('Asia/Riyadh'))
        df.loc[mask, 'flood_label']    = 1
        df.loc[mask, 'flood_severity'] = event['severity']

    print(f'Total flood hours labeled: {df.flood_label.sum()}')
    print(f'Total non-flood hours:     {(df.flood_label==0).sum()}')
    return df

def clean_data(df):
    print('Cleaning data...')
    df = df.ffill(limit=3)
    df = df.fillna(df.median(numeric_only=True))
    df['precipitation'] = df['precipitation'].clip(lower=0)
    df['rain']          = df['rain'].clip(lower=0)
    print(f'Final dataset: {len(df):,} rows, {len(df.columns)} columns')
    return df

if __name__ == '__main__':
    df = load_and_label()
    df = clean_data(df)
    df.to_csv('data/processed/labeled_data.csv', index=False)
    print('Saved to data/processed/labeled_data.csv')