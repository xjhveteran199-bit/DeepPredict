import pandas as pd
import numpy as np

# Download Mauna Loa CO2 data
url = 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt'
cols = ['year','month','dec_date','monthly_avg','deseasonalized','ndays','stdev','unc']
# skiprows: first 49 lines are header/comments, data starts at line 50
df = pd.read_csv(url, skiprows=50, names=cols, sep=r'\s+', comment='#')
print('Shape:', df.shape)
print(df.head())
print(df.tail())

# Use deseasonalized values as target, drop NaN
df = df.dropna(subset=['deseasonalized'])
y = df['deseasonalized'].values
print(f'\n有效样本数: {len(y)}')
print(f'范围: {y.min():.2f} - {y.max():.2f}')

# Save as CSV
df.to_csv('co2_data.csv', index=False)
print('Saved co2_data.csv')
