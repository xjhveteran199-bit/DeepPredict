import sys
sys.path.insert(0, '.')
import numpy as np
from src.models.cnn1d_complex import EnhancedCNN1DPredictor
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load ETTh1
df = pd.read_csv('test_data/ETTh1.csv')
date_cols = [c for c in df.columns if 'date' in c.lower()]
df = df.drop(columns=date_cols, errors='ignore').dropna()

print(f'Data: {df.shape[0]} samples, {df.shape[1]-1} features')

results = []

for name, target_col, normalize_X in [
    ('ETTh1 Multi-var (X norm)', 'OT', True),
    ('ETTh1 Multi-var (X raw)', 'OT', False),
    ('ETTh1 Single-var (HUFL)', 'HUFL', False),
]:
    y = df[target_col].values.astype(np.float32)
    X = df.drop(columns=[target_col]).values.astype(np.float32)

    # Normalize X per-feature (Critical for multivariate!)
    if normalize_X:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        print(f'\n{name}: X normalized')

    print(f'\n=== {name} ===')
    for run in range(3):
        model = EnhancedCNN1DPredictor()
        ok, _ = model.train(
            X, y,
            seq_len=96, pred_len=48,
            hidden_channels=64, num_scales=3, kernel_sizes=(3,5,7),
            num_res_blocks=2, epochs=50, batch_size=32,
            learning_rate=0.001, dropout=0.1, use_attention=True,
            test_size=0.2
        )
        if ok:
            m = model.metrics
            r2 = m['R2']
            rmse = m['RMSE']
            print(f'  Run{run+1}: R2={r2:.4f} RMSE={rmse:.4f}')
            results.append({'name': name, 'run': run+1, 'R2': r2, 'RMSE': rmse})
        else:
            print(f'  Run{run+1}: FAILED')

# Summary
print('\n' + '='*50)
print('SUMMARY')
print('='*50)
from collections import defaultdict
by_name = defaultdict(list)
for r in results:
    by_name[r['name']].append(r['R2'])

for n, vals in by_name.items():
    avg = np.mean(vals)
    best = max(vals)
    print(f'{n}: avg_R2={avg:.4f} best_R2={best:.4f}')
