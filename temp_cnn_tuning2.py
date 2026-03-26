"""
CNN1D 参数快速调优 - 基于 CNN1DModelV4
快速测试关键参数组合，目标 R² ≥ 0.55
"""
import sys
sys.path.insert(0, r"C:\Users\XJH\DeepPredict")

import numpy as np
import pandas as pd
import torch
import json
from src.models.cnn1d_model import CNN1DPredictorV4

# 加载测试数据
df = pd.read_csv(r"C:\Users\XJH\DeepResearch\test_data\dp_temperature.csv")
print(f"数据集: {df.shape}")

# 单变量：用索引作为X，温度作为y
X = np.arange(len(df)).astype(np.float32)
y = df["Temp"].values.astype(np.float32)
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"y range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")

# 关键参数组合（精简测试）
# seq_len: 90, 180, 365
# epochs: 80, 100, 150  
# hidden_channels: 64, 128
# kernel_size: 5, 7

results = []

# 测试函数
def test_config(seq_len, epochs, hidden_channels, kernel_size):
    try:
        predictor = CNN1DPredictorV4()
        # pred_len 设为 seq_len 的一半（合理的多步预测）
        pred_len = max(1, seq_len // 2)
        
        success, msg = predictor.train(
            X=X, y=y,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_channels=hidden_channels,
            num_layers=3,
            kernel_size=kernel_size,
            epochs=epochs,
            batch_size=16,
            learning_rate=0.001,
            test_size=0.2,
            target_col="Temp"
        )
        
        if success:
            r2 = predictor.metrics.get('R2', 0)
            rmse = predictor.metrics.get('RMSE', 999)
            print(f"  seq={seq_len}, ep={epochs}, hid={hidden_channels}, ker={kernel_size} => R²={r2:.4f}, RMSE={rmse:.4f}")
            return {'seq_len': seq_len, 'epochs': epochs, 'hidden_channels': hidden_channels, 'kernel_size': kernel_size, 'pred_len': pred_len, 'R2': r2, 'RMSE': rmse, 'success': True}
        else:
            print(f"  seq={seq_len}, ep={epochs}, hid={hidden_channels}, ker={kernel_size} => FAILED: {msg[:80]}")
            return {'seq_len': seq_len, 'epochs': epochs, 'hidden_channels': hidden_channels, 'kernel_size': kernel_size, 'pred_len': pred_len, 'R2': 0, 'RMSE': 999, 'success': False, 'error': msg[:200]}
    except Exception as e:
        print(f"  seq={seq_len}, ep={epochs}, hid={hidden_channels}, ker={kernel_size} => ERROR: {str(e)[:80]}")
        return {'seq_len': seq_len, 'epochs': epochs, 'hidden_channels': hidden_channels, 'kernel_size': kernel_size, 'pred_len': pred_len, 'R2': 0, 'RMSE': 999, 'success': False, 'error': str(e)[:200]}

print("\n" + "="*70)
print("开始 CNN1D 参数调优 (CNN1DModelV4)")
print("="*70)

# 策略：先用 seq_len=365 + epochs=150 测试两个 hidden_channels 和 kernel_sizes
print("\n=== 阶段1: 最佳 seq_len 探索 (seq_len=365, epochs=150) ===")
for hidden in [64, 128]:
    for kernel in [5, 7]:
        r = test_config(seq_len=365, epochs=150, hidden_channels=hidden, kernel_size=kernel)
        results.append(r)

# 阶段2: 如果还没达标，测试 seq_len=180
print("\n=== 阶段2: seq_len=180 测试 ===")
best_so_far = max([r for r in results if r['success']], key=lambda x: x['R2'], default=None)
print(f"当前最佳: {best_so_far}")

if best_so_far is None or best_so_far['R2'] < 0.55:
    for hidden in [64, 128]:
        for kernel in [5, 7]:
            r = test_config(seq_len=180, epochs=150, hidden_channels=hidden, kernel_size=kernel)
            results.append(r)

# 阶段3: seq_len=90
print("\n=== 阶段3: seq_len=90 测试 ===")
best_so_far = max([r for r in results if r['success']], key=lambda x: x['R2'], default=None)
print(f"当前最佳: {best_so_far}")

if best_so_far is None or best_so_far['R2'] < 0.55:
    for hidden in [64, 128]:
        for kernel in [5, 7]:
            r = test_config(seq_len=90, epochs=150, hidden_channels=hidden, kernel_size=kernel)
            results.append(r)

# 找最优
successful = [r for r in results if r['success']]
if successful:
    best = max(successful, key=lambda x: x['R2'])
    print("\n" + "="*70)
    print("最优配置:")
    print(f"  seq_len={best['seq_len']}, epochs={best['epochs']}, hidden={best['hidden_channels']}, kernel={best['kernel_size']}")
    print(f"  R²={best['R2']:.4f}, RMSE={best['RMSE']:.4f}")
    print("="*70)
    
    # 保存结果
    with open(r"C:\Users\XJH\DeepPredict\cnn1d_tuning_results.json", 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'all_results': results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 cnn1d_tuning_results.json")
    
    print("\n所有成功配置（按R²排序）:")
    for r in sorted(successful, key=lambda x: x['R2'], reverse=True):
        print(f"  seq={r['seq_len']}, ep={r['epochs']}, hid={r['hidden_channels']}, ker={r['kernel_size']} => R²={r['R2']:.4f}")
else:
    print("\n所有配置均失败！")
    with open(r"C:\Users\XJH\DeepPredict\cnn1d_tuning_results.json", 'w', encoding='utf-8') as f:
        json.dump({'best': None, 'all_results': results}, f, ensure_ascii=False, indent=2)
