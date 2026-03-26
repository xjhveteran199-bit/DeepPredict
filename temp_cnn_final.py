"""
CNN1D Final Tuning - seed-controlled, focused on hitting R2 >= 0.55
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Fixed seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

df = pd.read_csv(r"C:\Users\XJH\DeepResearch\test_data\dp_temperature.csv")
y_full = df["Temp"].values.astype(np.float32)
print(f"Data: {len(y_full)} samples, y: [{y_full.min():.2f}, {y_full.max():.2f}], mean={y_full.mean():.2f}")

class CNN1D(nn.Module):
    def __init__(self, input_size=1, hidden=64, num_layers=3, kernel=3, seq_len=96, pred_len=48, dropout=0.15):
        super().__init__()
        psp = [p for p in [1,2,4,8,16,24,32] if seq_len % p == 0]
        ps = psp[-1] if psp else 1
        np_ = seq_len // ps
        self.pe = nn.Parameter(torch.randn(1, np_, hidden) * 0.02)
        self.enc = nn.ModuleList()
        for _ in range(num_layers):
            self.enc.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel, padding=kernel//2, bias=False),
                nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout)))
        self.head = nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, pred_len))
        self.patch = nn.Conv1d(input_size, hidden, ps, stride=ps)
    
    def forward(self, x):
        x = self.patch(x.transpose(1,2)).transpose(1,2) + self.pe
        for layer in self.enc:
            x = x.transpose(1,2)
            x = layer(x) + x
            x = x.transpose(1,2)
        x = x.transpose(1,2)
        m, mx = x.mean(dim=-1), x.max(dim=-1)[0]
        return self.head(torch.cat([m, mx], dim=-1))

def build_seq(X, y, sl, pl):
    xs, ys = [], []
    for i in range(sl, len(y)-pl+1):
        xs.append(X[i-sl:i]); ys.append(y[i:i+pl])
    return np.array(xs), np.array(ys)

def run(sl, ep, hid, ker, lr=0.001):
    pl = max(1, sl // 2)
    xs, ys = build_seq(y_full, y_full, sl, pl)
    si = int(len(xs) * 0.8)
    Xt, Xv = xs[:si], xs[si:]
    yt, yv = ys[:si], ys[si:]
    
    ym, ys2 = yt.mean(), yt.std()
    yt_n = (yt - ym) / (ys2 + 1e-8)
    
    Xtt = torch.FloatTensor(Xt).unsqueeze(-1)
    ytt = torch.FloatTensor(yt_n)
    Xvt = torch.FloatTensor(Xv).unsqueeze(-1)
    
    model = CNN1D(1, hid, 3, ker, sl, pl, 0.15)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.HuberLoss(delta=0.5)
    
    for e in range(ep):
        idx = torch.randperm(len(Xtt))
        for i in range(0, len(Xtt), 16):
            bi = idx[i:i+16]
            opt.zero_grad()
            loss = crit(model(Xtt[bi]), ytt[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    
    model.eval()
    with torch.no_grad():
        pred = model(Xvt).numpy() * (ys2+1e-8) + ym
        r2 = r2_score(yv, pred)
        rmse = np.sqrt(mean_squared_error(yv, pred))
    return r2, rmse

# Key configs to test - based on trend analysis
configs = [
    (90, 150, 64, 5),   # More epochs for seq=90
    (90, 200, 64, 5),   # Even more epochs
    (180, 100, 64, 5),  # seq=180
    (180, 150, 64, 5),  # seq=180 with more epochs
    (90, 150, 128, 5),  # larger hidden
    (90, 120, 64, 3),   # kernel=3 baseline
]

results = []
print(f"\n=== Final CNN1D Tuning (seed={SEED}) ===")
for i, (sl, ep, hid, ker) in enumerate(configs):
    print(f"[{i+1}/{len(configs)}] seq={sl}, ep={ep}, hid={hid}, ker={ker}")
    r2, rmse = run(sl, ep, hid, ker)
    print(f"  => R2={r2:.4f}, RMSE={rmse:.4f}")
    results.append({'seq_len':sl, 'epochs':ep, 'hidden':hid, 'kernel':ker, 'R2':r2, 'RMSE':rmse})

best = max(results, key=lambda x: x['R2'])
print(f"\n=== BEST ===")
print(f"seq_len={best['seq_len']}, epochs={best['epochs']}, hidden={best['hidden']}, kernel={best['kernel']}")
print(f"R2={best['R2']:.4f}, RMSE={best['RMSE']:.4f}")

with open(r"C:\Users\XJH\DeepPredict\cnn1d_tuning_results.json", 'w') as f:
    json.dump({'best': best, 'all': results}, f, indent=2)
print("\nDone! Results saved.")
