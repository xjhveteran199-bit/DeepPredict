"""
Inspect the data to understand its pattern before training.
"""
import numpy as np
import pandas as pd
import sys

data_path = "C:/Users/XJH/Desktop/Raw_Data.csv"
df = pd.read_csv(data_path)
print(f"Shape: {df.shape}")
print(df.describe())

# Check K-with epifluidics specifically
k_col = 'K-with epifluidics'
k_vals = df[k_col].values
time_vals = df.iloc[:, 0].values

print(f"\nK-with epifluidics stats:")
print(f"  Mean: {k_vals.mean():.4f}")
print(f"  Std: {k_vals.std():.4f}")
print(f"  Min: {k_vals.min():.4f}")
print(f"  Max: {k_vals.max():.4f}")
print(f"  First 10: {k_vals[:10]}")
print(f"  Last 10: {k_vals[-10:]}")

# Check for trend
diff = np.diff(k_vals)
print(f"\nFirst differences - mean: {diff.mean():.6f}, std: {diff.std():.6f}")
print(f"First 10 diffs: {diff[:10]}")

# Check for periodic patterns using autocorrelation
from numpy.fft import fft as np_fft
n = len(k_vals)
fft_vals = np_fft(k_vals - k_vals.mean())
power = np.abs(fft_vals)**2
freqs = np.fft.fftfreq(n)
pos_mask = freqs > 0
print(f"\nTop frequencies: {sorted(zip(freqs[pos_mask], power[pos_mask]), key=lambda x: -x[1])[:5]}")
