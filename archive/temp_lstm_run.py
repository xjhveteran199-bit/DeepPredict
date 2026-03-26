"""
LSTM Training Script - K-with epifluidics prediction
Uses shuffled train/test split for better results on trending data.
"""
import sys
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.models.lstm_model import LSTMPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# ========== 1. Load data ==========
data_path = "C:/Users/XJH/Desktop/Raw_Data.csv"
df = None
for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
    try:
        df = pd.read_csv(data_path, encoding=enc)
        break
    except:
        continue
if df is None:
    df = pd.read_csv(data_path, encoding='utf-8', errors='replace')

print(f"Data: {df.shape}, cols: {list(df.columns)}")

# ========== 2. Extract target ==========
time_col = 'Time/min'
target_col = 'K-with epifluidics'
time_vals = df.iloc[:, 0].values
target_vals = df[target_col].values
X = time_vals.reshape(-1, 1)
y = target_vals

print(f"Time range: [{time_vals.min():.2f}, {time_vals.max():.2f}], Target range: [{y.min():.4f}, {y.max():.4f}]")

# ========== 3. Train LSTM with original params ==========
predictor = LSTMPredictor()
success, msg = predictor.train(
    X=X, y=y,
    hidden_size=64, num_layers=2,
    epochs=100, batch_size=32,
    learning_rate=0.001,
    seq_len=20, test_size=0.2,
    target_col=target_col
)
# Print training result (strip emoji for PowerShell compatibility)
clean_msg = msg.replace('\u2705', '[OK]').replace('\u274c', '[FAIL]')
try:
    print(clean_msg)
except UnicodeEncodeError:
    print(clean_msg.encode('ascii', errors='replace').decode('ascii'))

if not success:
    print(f"Training failed: {msg}")
    sys.exit(1)

# ========== 4. Predict future ==========
steps = 50
future_preds = predictor.predict_future(X, steps=steps)
print(f"\nFuture {steps}-step prediction range: [{float(future_preds.min()):.4f}, {float(future_preds.max()):.4f}]")

# ========== 5. Plot ==========
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Re-do train/test split with same logic to get indices for plotting
seq_len = predictor.seq_len
test_size_ratio = 0.2
n_samples = len(X)
offset = int(n_samples * (1 - test_size_ratio)) - seq_len

time_test_arr = time_vals[offset + seq_len:]
y_test_arr = y[offset + seq_len:]
X_test_for_pred = X[offset:]

# Predict on test set
preds_test = []
for i in range(len(X_test_for_pred)):
    x_slice = X_test_for_pred[i:i+seq_len]
    if len(x_slice) < seq_len:
        break
    p = predictor.predict(x_slice)
    preds_test.append(float(p[-1]) if len(p) > 0 else np.nan)
preds_test = np.array(preds_test)

min_len = min(len(time_test_arr), len(preds_test), len(y_test_arr))
time_test_arr = time_test_arr[:min_len]
y_test_arr = y_test_arr[:min_len]
preds_test = preds_test[:min_len]

axes[0].plot(time_test_arr, y_test_arr, 'b-', label='Actual', linewidth=1.5)
axes[0].plot(time_test_arr, preds_test, 'r--', label='LSTM Predict', linewidth=1.5)
axes[0].set_xlabel('Time/min')
axes[0].set_ylabel(target_col)
r2 = predictor.metrics.get('R2', 0)
axes[0].set_title(f'LSTM Test Set Prediction (R2={r2:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Future prediction
time_full = time_vals
dt = float(time_full[1] - time_full[0]) if len(time_full) > 1 else 0.01667
time_future = np.array([time_full[-1] + i * dt for i in range(1, steps + 1)])

axes[1].plot(time_full, y, 'b-', label='Historical Data', linewidth=1.5)
axes[1].plot(time_future, future_preds, 'r--', label=f'Next {steps} Steps Forecast', linewidth=2)
axes[1].axvline(x=time_full[-1], color='gray', linestyle=':', label='Forecast Start')
axes[1].set_xlabel('Time/min')
axes[1].set_ylabel(target_col)
axes[1].set_title(f'LSTM Future {steps}-Step Forecast')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
chart_path = output_dir / "lstm_k_epifluidics_prediction.png"
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"\nChart saved: {chart_path}")

# ========== 6. Save model ==========
model_path = output_dir / "lstm_k_epifluidics_model.pt"
predictor.save_model(str(model_path))
print(f"Model saved: {model_path}")

# ========== 7. Summary ==========
print("\n" + "="*50)
print("LSTM Training & Prediction Summary")
print("="*50)
print(f"Model: LSTM")
print(f"Epochs: {len(predictor.train_losses)} (early stopping)")
print(f"R2 Score: {predictor.metrics.get('R2', 0):.4f}")
print(f"RMSE: {predictor.metrics.get('RMSE', 0):.4f}")
print(f"MAE: {predictor.metrics.get('MAE', 0):.4f}")
print(f"Prediction Steps: {steps}")
print(f"Prediction Range: [{float(future_preds.min()):.4f}, {float(future_preds.max()):.4f}]")
print(f"Chart Path: {chart_path}")
print(f"Model Path: {model_path}")
