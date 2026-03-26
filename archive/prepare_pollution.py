import pandas as pd
import numpy as np
import os

os.makedirs('C:/Users/XJH/DeepPredict/data', exist_ok=True)

# Load
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv'
df = pd.read_csv(url)
print(f'Raw shape: {df.shape}')

# Use pm2.5 as target, drop rows with missing pm2.5
df = df.dropna(subset=['pm2.5'])
print(f'After dropping NaN: {df.shape}')

# Numeric columns only (drop 'No' and 'cbwd' categorical)
num_cols = ['year', 'month', 'day', 'hour', 'pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
df_num = df[num_cols].copy()

# Sort by time
df_num = df_num.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)

# Fill any remaining NaN
df_num = df_num.ffill().bfill()

print(f'Final shape: {df_num.shape}')
print(df_num.head())

# Save to CSV
df_num.to_csv(r'C:\Users\XJH\DeepPredict\data\pollution.csv', index=False)
print('Saved to pollution.csv')

# Show stats
target = 'pm2.5'
print(f'\nTarget ({target}) stats:')
print(df_num[target].describe())
