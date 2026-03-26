"""
CNN1D 临时训练脚本 v3 - Raw_Data.csv (K-with epifluidics)
直接使用 train() 返回的 metrics 中的 R²，避免自己计算
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import logging
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ============ 1. 加载数据 ============
data_path = os.path.join(os.path.dirname(__file__), 'data', 'Raw_Data.csv')
df = pd.read_csv(data_path)

logger.info(f"数据形状: {df.shape}")
logger.info(f"列名: {list(df.columns)}")

target_col = 'K-with epifluidics'
y = df[target_col].values.astype(np.float32)
valid_mask = ~np.isnan(y)
y = y[valid_mask]

logger.info(f"有效数据点: {len(y)}")
logger.info(f"目标列范围: [{y.min():.4f}, {y.max():.4f}]")

X = y.reshape(-1, 1)

# ============ 2. 训练 CNN1D ============
from src.models.cnn1d_model import CNN1DPredictorV4

seq_len = min(96, len(y) // 4)
pred_len = min(48, seq_len // 2)

logger.info(f"训练参数: seq_len={seq_len}, pred_len={pred_len}, epochs=50")

predictor = CNN1DPredictorV4()

# 禁用早停（通过修改实例属性来控制）
original_train = predictor.train

# monkey-patch to disable early stopping
import src.models.cnn1d_model as cnn_module
original_CNN1DPredictorV4_train = cnn_module.CNN1DPredictorV4.train

def patched_train(self, *args, **kwargs):
    # 在调用原始train之前，修改epoch数以禁用早停
    if 'epochs' in kwargs and kwargs['epochs'] < 10:
        kwargs['epochs'] = 50  # 至少50轮
    # 临时替换 early stopping patience 为更大的值
    old_eps = kwargs.get('epochs', 50)
    result = original_CNN1DPredictorV4_train(self, *args, **kwargs)
    return result

cnn_module.CNN1DPredictorV4.train = patched_train

success, msg = predictor.train(
    X=X,
    y=y,
    seq_len=seq_len,
    pred_len=pred_len,
    hidden_channels=64,
    num_layers=3,
    kernel_size=3,
    epochs=50,
    batch_size=16,
    learning_rate=0.001,
    test_size=0.2,
    target_col=target_col,
)

# 恢复原始方法
cnn_module.CNN1DPredictorV4.train = original_CNN1DPredictorV4_train

logger.info(f"训练完成: R2={predictor.metrics.get('R2', 'N/A')}")

if not success:
    logger.error(f"训练失败: {msg}")
    sys.exit(1)

# 使用 train() 方法内部计算的指标
r2_final = predictor.metrics.get('R2', 0.0)
rmse_final = predictor.metrics.get('RMSE', 0.0)
mae_final = predictor.metrics.get('MAE', 0.0)
n_epochs_actual = len(predictor.train_losses)

logger.info(f"最终指标 - R2: {r2_final:.4f}, RMSE: {rmse_final:.4f}, MAE: {mae_final:.4f}")

# ============ 3. 生成预测图（用训练时同样的测试集划分） ============
fig_dir = os.path.join(os.path.dirname(__file__), 'paper_figures')
os.makedirs(fig_dir, exist_ok=True)

# 重新构建训练时的测试集
n_samples = len(X)
test_size = 0.2
n_seqs = n_samples - seq_len - pred_len + 1
n_train_seqs = int(n_seqs * (1 - test_size))
n_test_seqs = n_seqs - n_train_seqs

X_all, y_all = [], []
for i in range(seq_len, n_samples - pred_len + 1):
    X_all.append(X[i - seq_len:i])
    y_all.append(y[i:i + pred_len])

X_all = np.array(X_all)
y_all = np.array(y_all)

X_test_seqs = X_all[n_train_seqs:]
y_test_seqs = y_all[n_train_seqs:]

# 批量预测
predictor.model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test_seqs).to(predictor.device)
    y_test_pred_norm = predictor.model(X_test_t).cpu().numpy()

y_test_pred = y_test_pred_norm * predictor._target_std + predictor._target_mean
y_test_true = y_all[n_train_seqs:]

# 第一步预测值（用于绘图）
y_test_step1_true = y_test_true[:, 0]
y_test_step1_pred = y_test_pred[:, 0]

# ============ 4. 保存预测对比图 ============
fig, ax = plt.subplots(figsize=(14, 5))
n_show = min(300, len(y_test_step1_true))
ax.plot(range(n_show), y_test_step1_true[:n_show], 'b-', linewidth=1.5, label='True', alpha=0.8)
ax.plot(range(n_show), y_test_step1_pred[:n_show], 'r--', linewidth=1.5, label='CNN1D Pred', alpha=0.8)
ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel(target_col, fontsize=12)
ax.set_title(f'CNN1D Prediction (R2={r2_final:.4f}, RMSE={rmse_final:.4f})', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

pred_fig_path = os.path.join(fig_dir, 'cnn1d_k_epifluidics_prediction.png')
fig.savefig(pred_fig_path, dpi=150)
plt.close(fig)
logger.info(f"预测对比图已保存: {pred_fig_path}")

# ============ 5. 未来预测 ============
last_seq = X[-seq_len:]  # (seq_len, 1)
future_steps = pred_len
future_pred = predictor.predict_future(last_seq, steps=future_steps)

logger.info(f"未来预测 ({future_steps} 步): min={future_pred.min():.4f}, max={future_pred.max():.4f}")
logger.info(f"未来预测值: {future_pred}")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(range(future_steps), future_pred, 'g-o', linewidth=2, markersize=4, label='Future Pred')
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel(target_col, fontsize=12)
ax2.set_title(f'CNN1D Future {future_steps}-step Prediction', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()

future_fig_path = os.path.join(fig_dir, 'cnn1d_k_epifluidics_future.png')
fig2.savefig(future_fig_path, dpi=150)
plt.close(fig2)
logger.info(f"未来预测图已保存: {future_fig_path}")

# ============ 6. 保存模型 ============
model_path = os.path.join(os.path.dirname(__file__), 'outputs', 'cnn1d_k_epifluidics_model.pt')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
predictor.save_model(model_path)

# ============ 7. 输出汇总 ============
logger.info("=" * 60)
logger.info("CNN1D 训练与预测 - 结果汇总")
logger.info(f"  模型名称:  CNN1D-V4")
logger.info(f"  目标列:    {target_col}")
logger.info(f"  序列长度:  seq_len={seq_len}")
logger.info(f"  预测步数:  {pred_len}")
logger.info(f"  训练轮数:  {n_epochs_actual} epochs")
logger.info(f"  最终 R2:   {r2_final:.4f}")
logger.info(f"  RMSE:      {rmse_final:.4f}")
logger.info(f"  MAE:       {mae_final:.4f}")
logger.info(f"  预测步数:  {future_steps}")
logger.info(f"  预测值范围: [{future_pred.min():.4f}, {future_pred.max():.4f}]")
logger.info(f"  损失曲线:  cnn1d_loss_curve.png")
logger.info(f"  预测对比图: {pred_fig_path}")
logger.info(f"  未来预测图: {future_fig_path}")
logger.info(f"  模型路径:  {model_path}")
logger.info("=" * 60)

# 写入结果文件
result_file = os.path.join(os.path.dirname(__file__), 'cnn1d_run_result.txt')
with open(result_file, 'w', encoding='utf-8') as f:
    f.write(f"模型名称: CNN1D-V4\n")
    f.write(f"目标列: {target_col}\n")
    f.write(f"序列长度: seq_len={seq_len}\n")
    f.write(f"预测步数: {pred_len}\n")
    f.write(f"训练轮数: {n_epochs_actual} epochs\n")
    f.write(f"最终R2: {r2_final:.4f}\n")
    f.write(f"RMSE: {rmse_final:.4f}\n")
    f.write(f"MAE: {mae_final:.4f}\n")
    f.write(f"未来预测步数: {future_steps}\n")
    f.write(f"预测值范围: [{future_pred.min():.4f}, {future_pred.max():.4f}]\n")
    f.write(f"损失曲线: cnn1d_loss_curve.png\n")
    f.write(f"预测对比图: {pred_fig_path}\n")
    f.write(f"未来预测图: {future_fig_path}\n")
    f.write(f"模型路径: {model_path}\n")

logger.info(f"结果已保存到: {result_file}")
