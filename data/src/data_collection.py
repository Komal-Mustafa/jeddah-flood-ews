# File: src/data_collection.py
# PURPOSE: Downloads historical climate data for Jeddah
# File: src/data_collection.py
# PURPOSE: Downloads historical climate data for Jeddah
# RUN: python src/data_collection.py


import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
import os
from datetime import datetime


# ── Setup cache (saves data locally so we don't re-download) ──
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


# ── Jeddah coordinates ──────────────────────────────────────────
JEDDAH_LAT = 21.5433
JEDDAH_LON = 39.1728


def download_historical_data(start_year=1985, end_year=2024):
    """
    Downloads hourly ERA5 climate data for Jeddah.
    Returns a pandas DataFrame ready for ML training.
    """
    print(f'Downloading data from {start_year} to {end_year}...')
    print('This may take 2-5 minutes for large date ranges...')
    
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude':  JEDDAH_LAT,
        'longitude': JEDDAH_LON,
        'start_date': f'{start_year}-01-01',
        'end_date':   f'{end_year}-12-31',
        'hourly': [
            'precipitation',           # mm — main flood trigger
            'rain',                    # mm — liquid rainfall
            'temperature_2m',          # C  — temperature
            'dewpoint_2m',             # C  — humidity proxy
            'relativehumidity_2m',     # %  — humidity
            'windspeed_10m',           # km/h
            'winddirection_10m',       # degrees
            'surface_pressure',        # hPa — atmospheric pressure
            'soil_moisture_0_1cm',     # m3/m3 — very top soil saturation
            'soil_moisture_1_3cm',     # m3/m3 — shallow soil saturation
            'soil_moisture_3_9cm',     # m3/m3 — deeper soil saturation
            'cape',                    # J/kg — convective instability (key!)
            'et0_fao_evapotranspiration',  # mm — evaporation
        ],
        'timezone': 'Asia/Riyadh',
        'models': 'era5',              # Use ERA5 reanalysis for accuracy
    }
    
    # Make the API call
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Extract the hourly data
    hourly = response.Hourly()
    
    # Build a dictionary with all variables
    hourly_data = {
        'date': pd.date_range(
            start = pd.Timestamp(hourly.Time(), unit='s', tz='Asia/Riyadh'),
            end   = pd.Timestamp(hourly.TimeEnd(), unit='s', tz='Asia/Riyadh'),
            freq  = pd.Timedelta(seconds=hourly.Interval()),
            inclusive='left'
        ),
        'precipitation':        hourly.Variables(0).ValuesAsNumpy(),
        'rain':                 hourly.Variables(1).ValuesAsNumpy(),
        'temperature_2m':       hourly.Variables(2).ValuesAsNumpy(),
        'dewpoint_2m':          hourly.Variables(3).ValuesAsNumpy(),
        'relativehumidity_2m':  hourly.Variables(4).ValuesAsNumpy(),
        'windspeed_10m':        hourly.Variables(5).ValuesAsNumpy(),
        'winddirection_10m':    hourly.Variables(6).ValuesAsNumpy(),
        'surface_pressure':     hourly.Variables(7).ValuesAsNumpy(),
        'soil_moisture_0_1cm':  hourly.Variables(8).ValuesAsNumpy(),
        'soil_moisture_1_3cm':  hourly.Variables(9).ValuesAsNumpy(),
        'soil_moisture_3_9cm':  hourly.Variables(10).ValuesAsNumpy(),
        'cape':                 hourly.Variables(11).ValuesAsNumpy(),
        'et0_evapotranspiration': hourly.Variables(12).ValuesAsNumpy(),
    }
    
    df = pd.DataFrame(data=hourly_data)
    print(f'Downloaded {len(df):,} hourly records')
    return df


def save_data(df, filename='jeddah_climate_raw.csv'):
    path = os.path.join('data', 'raw', filename)
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv(path, index=False)
    print(f'Saved to {path}')
    print(f'Columns: {list(df.columns)}')
    print(f'Date range: {df.date.min()} to {df.date.max()}')


if __name__ == '__main__':
    df = download_historical_data(start_year=1985, end_year=2024)
    save_data(df)
    print('Done! Check data/raw/jeddah_climate_raw.csv')
