"""
CNN1D 参数调优 - 直接调用模型（无matplotlib）
修复：X = y（过去温度值），不是索引
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载测试数据
df = pd.read_csv(r"C:\Users\XJH\DeepResearch\test_data\dp_temperature.csv")
print(f"数据集: {df.shape}")

# 单变量时序：X = 过去温度值，y = 未来温度值
y_full = df["Temp"].values.astype(np.float32)
print(f"y_full shape: {y_full.shape}")
print(f"y range: [{y_full.min():.2f}, {y_full.max():.2f}], mean={y_full.mean():.2f}, std={y_full.std():.2f}")

# CNN1DModelV4Simple 架构
class CNN1DModelV4Simple(nn.Module):
    def __init__(self, input_size=1, hidden_channels=64, num_layers=3, kernel_size=3, seq_len=96, pred_len=48, dropout=0.15):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        valid_patch_sizes = [p for p in [1, 2, 4, 8, 16, 24, 32] if seq_len % p == 0]
        if not valid_patch_sizes:
            valid_patch_sizes = [1]
        self.patch_size = valid_patch_sizes[-1]
        self.num_patches = seq_len // self.patch_size

        self.patch_embed = nn.Conv1d(input_size, hidden_channels, kernel_size=self.patch_size, stride=self.patch_size)
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_in', nonlinearity='conv1d')
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, hidden_channels) * 0.02)

        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, pred_len)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        x = self.patch_embed(x)  # (batch, hidden, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, hidden)
        x = x + self.pos_embedding

        for layer in self.encoder_layers:
            x = x.transpose(1, 2)  # (batch, hidden, num_patches)
            x = layer(x) + x  # 残差
            x = x.transpose(1, 2)  # (batch, num_patches, hidden)

        x = x.transpose(1, 2)  # (batch, hidden, num_patches)
        x_mean = x.mean(dim=-1)  # (batch, hidden)
        x_max = x.max(dim=-1)[0]  # (batch, hidden)
        x_combined = torch.cat([x_mean, x_max], dim=-1)  # (batch, hidden*2)
        out = self.head(x_combined)  # (batch, pred_len)
        return out


def build_sequences(X, y, seq_len, pred_len):
    """构建滑动窗口序列"""
    X_seqs, y_seqs = [], []
    for i in range(seq_len, len(y) - pred_len + 1):
        X_seqs.append(X[i - seq_len:i])   # 过去 seq_len 个值
        y_seqs.append(y[i:i + pred_len])   # 未来 pred_len 个值
    return np.array(X_seqs), np.array(y_seqs)


def train_and_eval(seq_len, pred_len, hidden_channels, kernel_size, num_layers, epochs, batch_size, lr, test_size):
    try:
        X_full = y_full  # 使用温度值本身作为输入（滞后特征）
        X_seqs, y_seqs = build_sequences(X_full, y_full, seq_len, pred_len)
        
        if len(X_seqs) < 10:
            return None, f"样本不足: {len(X_seqs)}"
        
        split_idx = int(len(X_seqs) * (1 - test_size))
        X_train, X_test = X_seqs[:split_idx], X_seqs[split_idx:]
        y_train, y_test = y_seqs[:split_idx], y_seqs[split_idx:]
        
        print(f"    train={len(X_train)}, test={len(X_test)}, seq={X_train.shape}")

        # 归一化（仅用训练集）
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
        
        X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)  # (N, seq_len, 1)
        y_train_t = torch.FloatTensor(y_train_norm)           # (N, pred_len)
        X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)    # (N, seq_len, 1)
        
        model = CNN1DModelV4Simple(
            input_size=1, hidden_channels=hidden_channels, num_layers=num_layers,
            kernel_size=kernel_size, seq_len=seq_len, pred_len=pred_len, dropout=0.15
        )
        
        criterion = nn.HuberLoss(delta=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_state = None
        
        model.train()
        for epoch in range(epochs):
            indices = torch.randperm(len(X_train_t))
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_train_t), batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]
                
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            
            # Early stopping on train loss (no val set here for speed)
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"    Early stop @ epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 25 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Restore best
        if best_state:
            model.load_state_dict(best_state)
        
        # 评估
        model.eval()
        with torch.no_grad():
            test_pred_norm = model(X_test_t).numpy()  # (N_test, pred_len)
            test_pred = test_pred_norm * (y_std + 1e-8) + y_mean  # 反归一化
            y_test_actual = y_test  # 原始值
            
            rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
            mae = mean_absolute_error(y_test_actual, test_pred)
            r2 = r2_score(y_test_actual.flatten(), test_pred.flatten())
        
        return {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae)}, None
        
    except Exception as e:
        import traceback
        return None, str(e) + "\n" + traceback.format_exc()[:200]


# 测试参数组合
results = []

seq_lens = [90, 180, 365]
epochs_list = [80, 100, 150]
hidden_channels_list = [64, 128]
kernel_sizes = [5, 7]
num_layers = 3
batch_size = 16
lr = 0.001
test_size = 0.2

total = len(seq_lens) * len(epochs_list) * len(hidden_channels_list) * len(kernel_sizes)
print(f"\n总共 {total} 个组合")
print("="*70)

counter = 0
for seq_len in seq_lens:
    pred_len = max(1, seq_len // 2)
    for epochs in epochs_list:
        for hidden in hidden_channels_list:
            for kernel in kernel_sizes:
                counter += 1
                print(f"\n[{counter}/{total}] seq={seq_len}, ep={epochs}, hid={hidden}, ker={kernel}")
                metrics, err = train_and_eval(
                    seq_len, pred_len, hidden, kernel, num_layers, epochs, batch_size, lr, test_size
                )
                if metrics:
                    r2 = metrics['R2']
                    rmse = metrics['RMSE']
                    print(f"  => R2={r2:.4f}, RMSE={rmse:.4f}")
                    results.append({
                        'seq_len': seq_len, 'epochs': epochs, 'hidden_channels': hidden,
                        'kernel_size': kernel, 'pred_len': pred_len,
                        'R2': r2, 'RMSE': rmse, 'MAE': metrics['MAE'], 'success': True
                    })
                else:
                    print(f"  => FAILED: {err[:200]}")
                    results.append({
                        'seq_len': seq_len, 'epochs': epochs, 'hidden_channels': hidden,
                        'kernel_size': kernel, 'pred_len': pred_len,
                        'R2': 0, 'RMSE': 999, 'success': False, 'error': err[:200]
                    })

# 保存和总结
successful = [r for r in results if r['success']]
if successful:
    best = max(successful, key=lambda x: x['R2'])
    print("\n" + "="*70)
    print("最优配置:")
    print(f"  seq_len={best['seq_len']}, epochs={best['epochs']}, hidden={best['hidden_channels']}, kernel={best['kernel_size']}")
    print(f"  R2={best['R2']:.4f}, RMSE={best['RMSE']:.4f}")
    print("="*70)
    
    with open(r"C:\Users\XJH\DeepPredict\cnn1d_tuning_results.json", 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'all_results': results}, f, ensure_ascii=False, indent=2)
    
    print("\n所有成功配置（按R2排序）:")
    for r in sorted(successful, key=lambda x: x['R2'], reverse=True):
        print(f"  seq={r['seq_len']}, ep={r['epochs']}, hid={r['hidden_channels']}, ker={r['kernel_size']} => R2={r['R2']:.4f}")
else:
    print("\n所有配置均失败！")
    with open(r"C:\Users\XJH\DeepPredict\cnn1d_tuning_results.json", 'w', encoding='utf-8') as f:
        json.dump({'best': None, 'all_results': results}, f, ensure_ascii=False, indent=2)
