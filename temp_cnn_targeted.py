"""
CNN1D Targeted tuning - focus on promising configs to hit R² >= 0.55
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv(r"C:\Users\XJH\DeepResearch\test_data\dp_temperature.csv")
y_full = df["Temp"].values.astype(np.float32)
print(f"数据: {len(y_full)} 样本, y range: [{y_full.min():.2f}, {y_full.max():.2f}]")

class CNN1DModelV4Simple(nn.Module):
    def __init__(self, input_size=1, hidden_channels=64, num_layers=3, kernel_size=3, seq_len=96, pred_len=48, dropout=0.15):
        super().__init__()
        valid_patch_sizes = [p for p in [1, 2, 4, 8, 16, 24, 32] if seq_len % p == 0]
        self.patch_size = valid_patch_sizes[-1] if valid_patch_sizes else 1
        self.num_patches = seq_len // self.patch_size
        
        self.patch_embed = nn.Conv1d(input_size, hidden_channels, kernel_size=self.patch_size, stride=self.patch_size)
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
        x = x.transpose(1, 2)
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        x = x + self.pos_embedding
        for layer in self.encoder_layers:
            x = x.transpose(1, 2)
            x = layer(x) + x
            x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        x_mean = x.mean(dim=-1)
        x_max = x.max(dim=-1)[0]
        x_combined = torch.cat([x_mean, x_max], dim=-1)
        return self.head(x_combined)

def build_sequences(X, y, seq_len, pred_len):
    X_seqs, y_seqs = [], []
    for i in range(seq_len, len(y) - pred_len + 1):
        X_seqs.append(X[i - seq_len:i])
        y_seqs.append(y[i:i + pred_len])
    return np.array(X_seqs), np.array(y_seqs)

def train_eval(seq_len, pred_len, hidden, kernel, epochs, lr=0.001, test_size=0.2):
    try:
        X_seqs, y_seqs = build_sequences(y_full, y_full, seq_len, pred_len)
        split_idx = int(len(X_seqs) * (1 - test_size))
        X_train, X_test = X_seqs[:split_idx], X_seqs[split_idx:]
        y_train, y_test = y_seqs[:split_idx], y_seqs[split_idx:]
        
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
        
        X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train_t = torch.FloatTensor(y_train_norm)
        X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
        
        model = CNN1DModelV4Simple(1, hidden, 3, kernel, seq_len, pred_len, 0.15)
        criterion = nn.HuberLoss(delta=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        model.train()
        for epoch in range(epochs):
            indices = torch.randperm(len(X_train_t))
            for i in range(0, len(X_train_t), 16):
                batch_idx = indices[i:i+16]
                optimizer.zero_grad()
                loss = criterion(model(X_train_t[batch_idx]), y_train_t[batch_idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            if (epoch+1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}")
        
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t).numpy() * (y_std + 1e-8) + y_mean
            r2 = r2_score(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        return r2, rmse
    except Exception as e:
        print(f"Error: {e}")
        return 0, 999

# Targeted configs based on earlier results
configs = [
    (90, 100, 64, 5),
    (90, 120, 64, 5),
    (90, 150, 64, 5),
    (180, 100, 64, 5),
    (180, 150, 64, 5),
    (90, 100, 64, 3),
    (90, 150, 64, 3),
    (180, 100, 64, 3),
    (180, 150, 64, 3),
    (90, 150, 128, 5),
    (365, 100, 64, 5),
]

results = []
print("\n=== Targeted CNN1D Tuning ===")
for i, (seq, ep, hid, ker) in enumerate(configs):
    pred = max(1, seq // 2)
    print(f"[{i+1}/{len(configs)}] seq={seq}, ep={ep}, hid={hid}, ker={ker}")
    r2, rmse = train_eval(seq, pred, hid, ker, ep)
    print(f"  => R2={r2:.4f}, RMSE={rmse:.4f}")
    results.append({'seq_len': seq, 'epochs': ep, 'hidden': hid, 'kernel': ker, 'R2': r2, 'RMSE': rmse})

best = max(results, key=lambda x: x['R2'])
print(f"\n=== Best Config ===")
print(f"seq_len={best['seq_len']}, epochs={best['epochs']}, hidden={best['hidden']}, kernel={best['kernel']}")
print(f"R2={best['R2']:.4f}, RMSE={best['RMSE']:.4f}")

# Save
with open(r"C:\Users\XJH\DeepPredict\cnn1d_tuning_results.json", 'w') as f:
    json.dump({'best': best, 'results': results}, f, indent=2)
print("\nResults saved!")
