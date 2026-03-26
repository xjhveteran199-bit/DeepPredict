# -*- coding: utf-8 -*-
"""验证 LSTM 和 PatchTST 模型 - 公平对比测试"""
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, r'C:\Users\XJH\DeepPredict')
os.chdir(r'C:\Users\XJH\DeepPredict')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 日志文件
LOG_FILE = r'C:\Users\XJH\DeepPredict\validate_log.txt'
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("[DATA] Load temperature.csv\n")
    f.write("=" * 60 + "\n")

df = pd.read_csv(r'C:\Users\XJH\DeepPredict\test_data\temperature.csv')
y = df['Temp'].values.astype(np.float32)
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(f"[DATA] shape={df.shape}, len={len(y)}, range=[{y.min():.1f}, {y.max():.1f}]\n\n")

# ====== 实验A: PatchTST 单步预测 (和 LSTM 公平对比) ======
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("[Exp A] PatchTST 单步预测 (pred_len=1)\n")
    f.write("=" * 60 + "\n")

try:
    from src.models.patchtst_model import PatchTSTPredictor

    patchtst_a = PatchTSTPredictor()
    success, msg = patchtst_a.train(
        y, y,
        seq_len=96, pred_len=1, patch_size=16,
        d_model=64, n_heads=2, n_layers=2, d_ff=128,
        epochs=30, batch_size=32
    )
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[PatchTST-A] Success: {success}\n")
        msg_clean = msg.replace('\u2705', '[OK]').replace('\u274c', '[FAIL]').replace('\U0001f4a1', '[IDEA]')
        f.write(f"[PatchTST-A] {msg_clean}\n")
        last_seq = y[-96:]
        pred_a = patchtst_a.predict(last_seq, pred_len=1)
        f.write(f"[PatchTST-A] Single-step pred: {pred_a[0]:.4f}\n")
        f.write(f"[PatchTST-A] Metrics: {patchtst_a.metrics}\n\n")
except Exception as e:
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[PatchTST-A] FAILED: {e}\n")
    import traceback; traceback.print_exc()

# ====== 实验A对照: LSTM 单步预测 ======
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write("-" * 60 + "\n")
    f.write("[Exp A Ctrl] LSTM 单步预测\n")
    f.write("-" * 60 + "\n")

try:
    from src.models.lstm_model import LSTMPredictor

    lstm = LSTMPredictor()
    success, msg = lstm.train(
        y, y,
        seq_len=30, hidden_size=32, num_layers=2,
        epochs=30, batch_size=64
    )
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[LSTM] Success: {success}\n")
        msg_clean = msg.replace('\u2705', '[OK]').replace('\u274c', '[FAIL]')
        f.write(f"[LSTM] {msg_clean}\n")
        f.write(f"[LSTM] Metrics: {lstm.metrics}\n\n")
except Exception as e:
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[LSTM] FAILED: {e}\n")
    import traceback; traceback.print_exc()

# ====== 实验B: PatchTST 多步预测 (大模型 + 多epoch) ======
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("[Exp B] PatchTST 多步预测 - 大模型\n")
    f.write("=" * 60 + "\n")

try:
    patchtst_b = PatchTSTPredictor()
    success, msg = patchtst_b.train(
        y, y,
        seq_len=96, pred_len=48, patch_size=16,
        d_model=256, n_heads=8, n_layers=4, d_ff=512,
        epochs=100, batch_size=32, learning_rate=0.0003
    )
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[PatchTST-B] Success: {success}\n")
        msg_clean = msg.replace('\u2705', '[OK]').replace('\u274c', '[FAIL]').replace('\U0001f4a1', '[IDEA]')
        f.write(f"[PatchTST-B] {msg_clean}\n")
        f.write(f"[PatchTST-B] Metrics: {patchtst_b.metrics}\n\n")
except Exception as e:
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[PatchTST-B] FAILED: {e}\n")
    import traceback; traceback.print_exc()

# ====== 实验B对照: LSTM 多步滚动预测 ======
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write("-" * 60 + "\n")
    f.write("[Exp B Ctrl] LSTM 多步滚动预测 (48步)\n")
    f.write("-" * 60 + "\n")

try:
    lstm_b = LSTMPredictor()
    success, msg = lstm_b.train(
        y, y,
        seq_len=96, hidden_size=64, num_layers=2,
        epochs=100, batch_size=64
    )
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[LSTM-B] Success: {success}\n")
        msg_clean = msg.replace('\u2705', '[OK]').replace('\u274c', '[FAIL]')
        f.write(f"[LSTM-B] {msg_clean}\n")

    # 滚动预测48步
    multistep_pred = []
    current_seq = list(y[-96:])
    for _ in range(48):
        pred_val = lstm_b.predict(np.array(current_seq[-96:]))[-1]
        multistep_pred.append(pred_val)
        current_seq.append(pred_val)
    multistep_pred = np.array(multistep_pred)

    y_true = y[-48:]
    mse = np.mean((multistep_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(multistep_pred - y_true))
    ss_res = np.sum((y_true - multistep_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[LSTM-B Multi-step] RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}\n")
        f.write(f"[LSTM-B] Metrics: {lstm_b.metrics}\n\n")
except Exception as e:
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[LSTM-B] FAILED: {e}\n")
    import traceback; traceback.print_exc()

with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("ALL EXPERIMENTS COMPLETE\n")
    f.write("=" * 60 + "\n")

print("DONE - see validate_log.txt")
