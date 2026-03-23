"""分段测试：跑完即保存，不丢结果"""
import sys, json, time, os
sys.path.insert(0, '.')
import numpy as np
from src.models.cnn1d_complex import EnhancedCNN1DPredictor
import pandas as pd
from sklearn.preprocessing import StandardScaler

RESULTS_FILE = 'enhanced_cnn_results.json'
RANGE_FILE = 'enhanced_cnn_range.json'

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []

def save_result(name, run, r2, rmse, mae):
    results = load_results()
    results.append({'name': name, 'run': run, 'R2': r2, 'RMSE': rmse, 'MAE': mae})
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[SAVED] {name} Run{run}: R2={r2:.4f} RMSE={rmse:.4f}')

# Load data once
df = pd.read_csv('test_data/ETTh1.csv')
date_cols = [c for c in df.columns if 'date' in c.lower()]
df = df.drop(columns=date_cols, errors='ignore').dropna()

# ===== Test 1: ETTh1 Multi-var X norm =====
print('\n=== Test 1: ETTh1 Multi-var (X normalized) ===')
y = df['OT'].values.astype(np.float32)
X = df.drop(columns=['OT']).values.astype(np.float32)
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)
print(f'X shape: {X.shape}, y shape: {y.shape}')

for run in range(1, 4):
    print(f'\nRun {run}/3...')
    t0 = time.time()
    model = EnhancedCNN1DPredictor()
    ok, _ = model.train(X, y,
        seq_len=96, pred_len=48,
        hidden_channels=64, num_scales=3, kernel_sizes=(3,5,7),
        num_res_blocks=2, epochs=50, batch_size=32,
        learning_rate=0.001, dropout=0.1, use_attention=True,
        test_size=0.2
    )
    elapsed = time.time() - t0
    print(f'  Time: {elapsed:.1f}s, OK: {ok}')
    if ok:
        m = model.metrics
        save_result('ETTh1 Multi-var X-norm OT', run, m['R2'], m['RMSE'], m['MAE'])
    else:
        print(f'  FAILED')
