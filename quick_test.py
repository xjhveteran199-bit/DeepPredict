# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np

# Quick test of data loader v2
from core.data_loader import DataLoader

dl = DataLoader()
ok, msg = dl.load_csv("data/pollution.csv")
print("OK" if ok else "FAIL", msg)

summary = dl.get_summary()
print("Shape:", summary['shape'])
print("Numeric:", summary['numeric_cols'])
print("Date cols:", summary['date_cols'])

# Sampling regularity
si = dl.detect_irregular_sampling()
print("Regular:", si.get('is_regular'))
print("CV:", round(si.get('cv', 0), 4))

# Feature matrix
X = dl.get_feature_matrix(exclude_cols=['No'])
print("Feature matrix:", X.shape)
print("Features:", list(X.columns)[:10])
