"""
PatchTST 训练脚本 - 针对桌面 Raw_Data.csv 数据
目标：用 Time/min 作为时间索引，K-with epifluidics 作为目标列
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import sys
sys.path.insert(0, r"C:\Users\XJH\DeepPredict\src")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# ========== 1. 加载数据 ==========
DATA_PATH = r"C:\Users\XJH\Desktop\Raw_Data.csv"
OUTPUT_DIR = Path(r"C:\Users\XJH\DeepPredict\outputs\patchtst_k_epifluidics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# 使用第一个 Time/min 列（原始列）
time_col = "Time/min"
target_col = "K-with epifluidics"

# 按时间排序
df_sorted = df.sort_values(time_col).reset_index(drop=True)
print(f"时间范围: {df_sorted[time_col].min():.4f} ~ {df_sorted[time_col].max():.4f}")
print(f"目标范围: {df_sorted[target_col].min():.4f} ~ {df_sorted[target_col].max():.4f}")

# 提取目标列（单变量时序）
y = df_sorted[target_col].values.astype(np.float32)
time_vals = df_sorted[time_col].values.astype(np.float32)

# 由于 PatchTST 单变量模式 X=y
X_for_model = y.copy()

print(f"\n样本数: {len(y)}")

# ========== 2. 导入并训练 PatchTST ==========
from src.models.patchtst_model import PatchTSTPredictor

# 针对2401条数据，设置合理的窗口参数
# seq_len+pred_len 应小于样本数，且留足够训练样本
n = len(y)
# 使用较小的 seq_len/pred_len 以适应数据量
seq_len = min(48, n // 5)
pred_len = min(24, n // 10)
pred_len = max(6, pred_len)  # 至少预测6步

print(f"\n模型参数: seq_len={seq_len}, pred_len={pred_len}")

lstm_pred = PatchTSTPredictor()

success, msg = lstm_pred.train(
    X_for_model, y,
    seq_len=seq_len,
    pred_len=pred_len,
    patch_size=8,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    test_size=0.2,
    target_col=target_col,
)

print("\n========== 训练结果 ==========")
print(msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))

# ========== 3. 保存模型 ==========
model_path = OUTPUT_DIR / "patchtst_k_epifluidics.pt"
lstm_pred.save_model(str(model_path))
print(f"模型已保存: {model_path}")

# ========== 4. 生成预测并绘图 ==========
# 使用最后 seq_len 个点作为输入，预测未来步
seq_len_actual = lstm_pred.seq_len
X_last = y[-seq_len_actual:] if len(y) >= seq_len_actual else y
n_future = 50  # 预测未来50步
future_preds = lstm_pred.predict_future(X_last, steps=n_future)

print(f"\n预测步数: {len(future_preds)}")
print(f"预测值范围: {float(future_preds.min()):.4f} ~ {float(future_preds.max()):.4f}")
print(f"预测均值: {float(future_preds.mean()):.4f}")

# ========== 5. 绘图 ==========
last_n_plot = min(100, len(y))
hist = y[-last_n_plot:]

# 时间轴数值用于 X 轴
last_time = time_vals[-last_n_plot]
interval = float(np.median(np.diff(time_vals))) if len(time_vals) > 1 else 0.1

# 历史 x 轴（时间值）
steps_hist = list(time_vals[-last_n_plot:])
# 未来 x 轴（时间值）
steps_fut = [float(time_vals[-1]) + interval * (i + 1) for i in range(len(future_preds))]

# X 轴标签（每隔一段显示）
hist_step = max(1, last_n_plot // 8)
xtick_hist = [f"{steps_hist[i]:.1f}" if i % hist_step == 0 else "" for i in range(len(steps_hist))]
fut_step = max(1, len(future_preds) // 8)
xtick_fut = [f"{steps_fut[i]:.1f}" if i % fut_step == 0 else "" for i in range(len(steps_fut))]

fig, ax = plt.subplots(figsize=(14, 5))

# 分隔线
boundary = float(time_vals[-1])
ax.axvline(x=boundary, color='gray', linestyle=':', linewidth=1.5, label='Forecast Start')

# 历史数据
ax.plot(steps_hist, hist, color='#3B82F6', linewidth=2, label='Historical (K-with epifluidics)')
# 预测
ax.plot(steps_fut, future_preds, color='#FF6B2B', linewidth=2, linestyle='--', label='PatchTST Forecast')

# X 轴
all_steps = steps_hist + steps_fut
all_labels = xtick_hist + xtick_fut
hist_ticks = [(i, xtick_hist[i]) for i in range(len(xtick_hist)) if xtick_hist[i]]
fut_ticks = [(len(steps_hist) + i, xtick_fut[i]) for i in range(len(xtick_fut)) if xtick_fut[i]]
all_ticks = hist_ticks + fut_ticks
if all_ticks:
    tick_pos, tick_lab = zip(*all_ticks)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=30, ha='right', fontsize=9)

ax.set_xlabel('Time/min', fontsize=11)
ax.set_ylabel(target_col, fontsize=11)
ax.set_title(f'PatchTST Forecast - {target_col} (Time/min: {float(time_vals[0]):.1f}-{float(time_vals[-1]):.1f})', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 保存图片
fig_path = OUTPUT_DIR / "patchtst_forecast.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n图表已保存: {fig_path}")

# ========== 6. 绘制损失曲线（如果有） ==========
if hasattr(lstm_pred, 'metrics') and lstm_pred.metrics:
    metrics = lstm_pred.metrics
    r2 = metrics.get('R2', 'N/A')
    rmse = metrics.get('RMSE', 'N/A')
    mae = metrics.get('MAE', 'N/A')
    epochs_trained = metrics.get('epochs', 'N/A')
    final_loss = metrics.get('final_train_loss', 'N/A')

    print(f"\n========== 最终结果汇总 ==========")
    print(f"模型名称: PatchTST")
    print(f"训练轮数: {epochs_trained}")
    print(f"最终训练损失: {final_loss}")
    print(f"R² 分数: {r2}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"预测步数: {len(future_preds)}")
    print(f"预测值范围: {float(future_preds.min()):.4f} ~ {float(future_preds.max()):.4f}")
    print(f"图表路径: {fig_path}")
    print(f"模型路径: {model_path}")

    # 保存汇总到 JSON
    import json
    summary = {
        'model': 'PatchTST',
        'target': target_col,
        'time_col': time_col,
        'seq_len': int(seq_len_actual),
        'pred_len': int(lstm_pred.pred_len),
        'epochs_trained': int(epochs_trained) if isinstance(epochs_trained, (int, float)) else epochs_trained,
        'final_train_loss': float(final_loss) if isinstance(final_loss, (int, float)) else final_loss,
        'R2': float(r2) if isinstance(r2, (int, float)) else r2,
        'RMSE': float(rmse) if isinstance(rmse, (int, float)) else rmse,
        'MAE': float(mae) if isinstance(mae, (int, float)) else mae,
        'n_future_steps': len(future_preds),
        'pred_range': [float(future_preds.min()), float(future_preds.max())],
        'pred_mean': float(future_preds.mean()),
        'fig_path': str(fig_path),
        'model_path': str(model_path),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总已保存: {summary_path}")
else:
    print(f"\n预测步数: {len(future_preds)}")
    print(f"预测值范围: {float(future_preds.min()):.4f} ~ {float(future_preds.max()):.4f}")
    print(f"图表路径: {fig_path}")
    print(f"模型路径: {model_path}")
