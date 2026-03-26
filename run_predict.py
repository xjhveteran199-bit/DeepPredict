# -*- coding: utf-8 -*-
"""Run K-with epifluidics 10-minute forecast prediction"""
import os
import sys
import traceback
from pathlib import Path

# 强制 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np

# ============================================================
# Step 1: 加载数据
# ============================================================
print('=' * 60)
print('Step 1: Load data')
print('=' * 60)

df = pd.read_csv(r'C:\Users\XJH\Desktop\Raw_Data.csv')
target_col = 'K-with epifluidics'
time_col = 'Time/min'

y = df[target_col].values.astype(np.float32)
x = pd.to_numeric(df[time_col], errors='coerce').fillna(0).values.astype(np.float32)
sort_idx = np.argsort(x)
y = y[sort_idx]
x = x[sort_idx]

# 时间步长: (50-10)/(2401-1) = 40/2400 = 0.01667 min/step
step_interval = (x.max() - x.min()) / (len(x) - 1)
# 10分钟对应的步数
n_steps_10min = max(1, int(round(10.0 / step_interval)))

print(f'Target: {target_col}')
print(f'Time range: {x.min():.4f} ~ {x.max():.4f} min')
print(f'K range: {y.min():.4f} ~ {y.max():.4f}')
print(f'Samples: {len(y)}')
print(f'Step interval: {step_interval:.5f} min/step')
print(f'10 min = {n_steps_10min} steps')

# ============================================================
# Step 2: 训练 CNN1D
# ============================================================
print()
print('=' * 60)
print('Step 2: Train CNN1D model')
print('=' * 60)

sys.path.insert(0, '.')
from src.models.cnn1d_model import CNN1DPredictorV4

model = CNN1DPredictorV4()
ok, msg = model.train(
    X=y, y=y,
    seq_len=96, pred_len=48,
    epochs=50, batch_size=16, learning_rate=0.001,
    test_size=0.2, target_col=target_col
)
print(f'Train result: ok={ok}')
print(f'Message: {msg}')
print(f'is_fitted: {model.is_fitted}')

if not ok:
    print('ERROR: Training failed!')
    sys.exit(1)

# ============================================================
# Step 3: 预测未来 10 分钟 (600 steps)
# ============================================================
print()
print('=' * 60)
print(f'Step 3: Predict future {n_steps_10min} steps (~10 min)')
print('=' * 60)

try:
    future_preds = model.predict_future(y, steps=n_steps_10min)
    print(f'Prediction shape: {future_preds.shape}')
    print(f'Prediction range: {future_preds.min():.4f} ~ {future_preds.max():.4f}')
    print(f'First 5 values: {list(future_preds[:5])}')
    print(f'Last 5 values: {list(future_preds[-5:])}')
except Exception as e:
    print(f'ERROR during predict_future: {e}')
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# Step 4: 计算预测时间轴 & 保存 CSV
# ============================================================
print()
print('=' * 60)
print('Step 4: Build result dataframe and save CSV')
print('=' * 60)

# 历史时间
last_time = x[-1]
future_times = np.array([last_time + (i + 1) * step_interval for i in range(n_steps_10min)])

print(f'Last historical time: {last_time:.5f} min')
print(f'First predicted time: {future_times[0]:.5f} min')
print(f'Last predicted time: {future_times[-1]:.5f} min')

output_dir = Path('outputs/cnn1d_k_10min_forecast')
output_dir.mkdir(parents=True, exist_ok=True)

result_df = pd.DataFrame({
    'Time/min': future_times,
    'K-with epifluidics (predicted)': future_preds
})
csv_path = output_dir / 'forecast_data.csv'
result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'CSV saved: {csv_path}')
print(result_df.head(10).to_string())

# ============================================================
# Step 5: 绘图
# ============================================================
print()
print('=' * 60)
print('Step 5: Plot and save chart')
print('=' * 60)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# x轴标签：时间刻度
xtick_range = np.linspace(x.min(), future_times[-1], 12)
xtick_labels = [f'{t:.1f}' for t in xtick_range]

fig, ax = plt.subplots(figsize=(12, 5))

# 历史数据（最后200点）
n_show = 200
ax.plot(x[-n_show:], y[-n_show:], 'b-', linewidth=1.5, label='Historical', alpha=0.8)

# 预测数据
ax.plot(future_times, future_preds, 'r--', linewidth=1.5, label='Forecast (10 min)', alpha=0.8)

# 分界线
ax.axvline(x=last_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Now')

ax.set_xlabel('Time (min)', fontsize=11)
ax.set_ylabel('K-with epifluidics', fontsize=11)
ax.set_title(f'CNN1D Forecast: K-with epifluidics (Next 10 min, {n_steps_10min} steps)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.FixedLocator(xtick_range))
ax.set_xticklabels(xtick_labels, rotation=45)

plt.tight_layout()
fig_path = output_dir / 'forecast_chart.png'
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Chart saved: {fig_path}')

# ============================================================
# Done
# ============================================================
print()
print('=' * 60)
print('ALL SUCCESS')
print('=' * 60)
print(f'Model: CNN1D')
print(f'Forecast: {n_steps_10min} steps (~10 minutes)')
print(f'Prediction range: {future_preds.min():.4f} ~ {future_preds.max():.4f}')
print(f'Output dir: {output_dir}')
