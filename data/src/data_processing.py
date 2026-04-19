import pandas as pd
import numpy as np
from pathlib import Path

# Paths
RAW_PATH = Path("data/raw/jeddah_climate_raw.csv")
PROCESSED_PATH = Path("data/processed/jeddah_climate_processed.csv")
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading raw data...")
df = pd.read_csv(RAW_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print("Engineering features...")
df['hour'] = df['date'].dt.hour
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear

# Rolling rainfall features (min_periods=1 prevents dropping rows)
df['rain_6h'] = df['rain'].rolling(6, min_periods=1).sum()
df['rain_24h'] = df['rain'].rolling(24, min_periods=1).sum()
df['rain_72h'] = df['rain'].rolling(72, min_periods=1).sum()

# Flood label: 24h rainfall > 20mm
df['flood_risk'] = (df['rain_24h'] > 20).astype(int)

print(f"Processed {len(df)} records")
print(f"Flood risk events: {df['flood_risk'].sum()} ({df['flood_risk'].mean()*100:.2f}%)")

df.to_csv(PROCESSED_PATH, index=False)
print(f"Saved to {PROCESSED_PATH}")