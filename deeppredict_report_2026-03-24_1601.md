# DeepPredict 自动调试简报
**时间**: 2026-03-24 16:01
**数据集**: 月度香槟销售 (Monthly Champagne Sales) (n=105)
**数据来源**: https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly_champagne_sales.csv

## 模型结果

| 模型 | R² | RMSE | MAE | 状态 |
|------|-----|------|-----|------|
| CNN1D | 0.1136 | 2558.01 | 1884.00 | ✅ |
| PatchTST | -0.2213 | 3002.49 | 1811.00 | ⚠️ R²为负 |
| LSTM | 0.3147 | 2249.11 | 1635.11 | ✅ |

## 报错修复记录
（本次无报错，所有模型均成功运行）

## 备注
- **LSTM 表现最优**（R²=0.3147），月度香槟销售具有较强季节性，LSTM 能更好地捕捉长周期依赖
- **CNN1D 次之**（R²=0.1136），小样本（105条月度数据）限制了 1D-CNN 的特征提取能力
- **PatchTST R²为负**（-0.2213），PatchTST 在极短序列（seq_len=9）上 patch 划分受限，模型无法有效学习
- 数据量偏小（105条），训练集仅73条，建议后续选用数据量≥500的数据集进一步验证
