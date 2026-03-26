"""
CNN1D 参数调优脚本
测试 seq_len x epochs x hidden_channels x kernel_size 组合
目标: R² ≥ 0.55
"""
import sys
sys.path.insert(0, r"C:\Users\XJH\DeepPredict")

import numpy as np
import pandas as pd
import torch
import json
from src.models.cnn1d_complex import EnhancedCNN1DPredictor

# 加载测试数据
df = pd.read_csv(r"C:\Users\XJH\DeepResearch\test_data\dp_temperature.csv")
print(f"数据集: {df.shape}")
print(df.head(3))

# 使用 Date 和 Temp 列
if "Date" in df.columns and "Temp" in df.columns:
    X = np.arange(len(df)).astype(np.float32)
    y = df["Temp"].values.astype(np.float32)
elif df.shape[1] >= 2:
    X = np.arange(len(df)).astype(np.float32)
    y = df.iloc[:, 1].values.astype(np.float32)
else:
    X = np.arange(len(df)).astype(np.float32)
    y = df.iloc[:, 0].values.astype(np.float32)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"y range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")

# 参数网格
seq_lens = [90, 180, 365]
epochs_list = [80, 100, 150]
hidden_channels_list = [64, 128]
kernel_sizes_list = [(5,), (7,)]  # 用单kernel版本测试

results = []

print("\n" + "="*80)
print("开始 CNN1D 参数调优")
print("="*80)

total = len(seq_lens) * len(epochs_list) * len(hidden_channels_list) * len(kernel_sizes_list)
counter = 0

for seq_len in seq_lens:
    for epochs in epochs_list:
        for hidden in hidden_channels_list:
            for kernel_sizes in kernel_sizes_list:
                counter += 1
                print(f"\n[{counter}/{total}] Testing: seq_len={seq_len}, epochs={epochs}, hidden={hidden}, kernel={kernel_sizes}")
                
                try:
                    predictor = EnhancedCNN1DPredictor()
                    success, msg = predictor.train(
                        X=X, y=y,
                        seq_len=seq_len,
                        pred_len=max(1, seq_len // 4),  # 保持合理比例
                        hidden_channels=hidden,
                        kernel_sizes=kernel_sizes,
                        num_scales=1,  # 简化：用单kernel
                        num_res_blocks=2,
                        epochs=epochs,
                        batch_size=32,
                        learning_rate=0.001,
                        test_size=0.2,
                        dropout=0.15,
                        use_attention=True,
                    )
                    
                    if success:
                        r2 = predictor.metrics.get('R2', 0)
                        rmse = predictor.metrics.get('RMSE', 999)
                        print(f"  => R²={r2:.4f}, RMSE={rmse:.4f}")
                        results.append({
                            'seq_len': seq_len,
                            'epochs': epochs,
                            'hidden_channels': hidden,
                            'kernel_sizes': kernel_sizes,
                            'R2': r2,
                            'RMSE': rmse,
                            'success': True
                        })
                    else:
                        print(f"  => FAILED: {msg[:100]}")
                        results.append({
                            'seq_len': seq_len,
                            'epochs': epochs,
                            'hidden_channels': hidden,
                            'kernel_sizes': kernel_sizes,
                            'R2': 0,
                            'RMSE': 999,
                            'success': False,
                            'error': msg[:200]
                        })
                        
                except Exception as e:
                    print(f"  => ERROR: {str(e)[:100]}")
                    results.append({
                        'seq_len': seq_len,
                        'epochs': epochs,
                        'hidden_channels': hidden,
                        'kernel_sizes': kernel_sizes,
                        'R2': 0,
                        'RMSE': 999,
                        'success': False,
                        'error': str(e)[:200]
                    })

# 保存结果
with open(r"C:\Users\XJH\DeepPredict\cnn1d_tuning_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 找最优
successful_results = [r for r in results if r['success']]
if successful_results:
    best = max(successful_results, key=lambda x: x['R2'])
    print("\n" + "="*80)
    print("最优配置:")
    print(f"  seq_len={best['seq_len']}, epochs={best['epochs']}, hidden={best['hidden_channels']}, kernel={best['kernel_sizes']}")
    print(f"  R²={best['R2']:.4f}, RMSE={best['RMSE']:.4f}")
    print("="*80)
    
    # 所有成功结果排序
    print("\n所有成功配置（按R²排序）:")
    for r in sorted(successful_results, key=lambda x: x['R2'], reverse=True):
        print(f"  seq={r['seq_len']}, ep={r['epochs']}, hid={r['hidden_channels']}, ker={r['kernel_sizes']} => R²={r['R2']:.4f}")
else:
    print("\n所有配置均失败！")
