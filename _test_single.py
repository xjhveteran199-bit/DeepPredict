"""单次50轮测试：验证 EnhancedCNN1D + X标准化效果"""
import sys, json, time, os
sys.path.insert(0, '.')
import numpy as np
from src.models.cnn1d_complex import EnhancedCNN1DPredictor
import pandas as pd
from sklearn.preprocessing import StandardScaler

print('Loading ETTh1 data...')
df = pd.read_csv('test_data/ETTh1.csv')
date_cols = [c for c in df.columns if 'date' in c.lower()]
df = df.drop(columns=date_cols, errors='ignore').dropna()
print(f'Data: {df.shape}')

results = []

# ===== Test 1: Multi-variable =====
print('\n=== Test 1: ETTh1 Multi-var (X normalized) OT ===')
y = df['OT'].values.astype(np.float32)
X = df.drop(columns=['OT']).values.astype(np.float32)
X = StandardScaler().fit_transform(X).astype(np.float32)
print(f'X: {X.shape}, y: {y.shape}')

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
print(f'Done in {elapsed:.1f}s, OK={ok}')
if ok:
    m = model.metrics
    print(f'R2={m["R2"]:.4f} RMSE={m["RMSE"]:.4f} MAE={m["MAE"]:.4f}')
    results.append({'Test': 'ETTh1 Multi-var', 'R2': m['R2'], 'RMSE': m['RMSE'], 'MAE': m['MAE'], 'Time_s': elapsed})

# ===== Test 2: Single-variable =====
print('\n=== Test 2: ETTh1 Single-var HUFL ===')
y2 = df['HUFL'].values.astype(np.float32)
X2 = df[['HUFL']].values.astype(np.float32)
# 单变量不做X标准化（已有按y的标准化）

t0 = time.time()
model2 = EnhancedCNN1DPredictor()
ok2, _ = model2.train(X2, y2,
    seq_len=96, pred_len=48,
    hidden_channels=64, num_scales=3, kernel_sizes=(3,5,7),
    num_res_blocks=2, epochs=50, batch_size=32,
    learning_rate=0.001, dropout=0.1, use_attention=True,
    test_size=0.2
)
elapsed2 = time.time() - t0
print(f'Done in {elapsed2:.1f}s, OK={ok2}')
if ok2:
    m2 = model2.metrics
    print(f'R2={m2["R2"]:.4f} RMSE={m2["RMSE"]:.4f} MAE={m2["MAE"]:.4f}')
    results.append({'Test': 'ETTh1 Single-var', 'R2': m2['R2'], 'RMSE': m2['RMSE'], 'MAE': m2['MAE'], 'Time_s': elapsed2})

# Save
with open('enhanced_cnn_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nResults:', json.dumps(results, indent=2))
