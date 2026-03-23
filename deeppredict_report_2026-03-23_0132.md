# DeepPredict 自动调试简报
**时间**: 2026-03-23 01:32
**数据集**: 洗发水销售 (Shampoo Sales) (n=36)
**数据来源**: https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv

## 模型结果

| 模型 | R² | RMSE | MAE | 状态 |
|------|-----|------|-----|------|
| CNN1D | -2.6986 | 218.7973 | 163.5282 | ⚠️ 负R² |
| PatchTST | — | — | — | ❌ 数据不足 |
| LSTM | -4.5939 | 269.0779 | 243.9120 | ⚠️ 负R² |

## 报错修复记录

- **LSTM** 报错: `LSTMPredictor.predict_future() got an unexpected keyword argument 'n_future'` → **已修复**: LSTM 接口参数名是 `steps`，不是 `n_future`（在本次测试脚本中修正）
- **PatchTST** 报错: `数据不足：36 条样本不足以进行训练。请至少准备 52 条数据` → **无法修复**: PatchTST 内部硬编码最小样本要求 `n ≥ seq_len + pred_len + 50`，36 样本远低于阈值

## 备注

- **数据规模问题**：洗发水销售数据集仅有 36 个月度样本，是典型的小样本时序，深度学习模型在此规模下无法学习到有效模式
- **PatchTST** 训练失败是结构性限制，非参数调整可解决
- **CNN1D 和 LSTM** 均能运行但 R² 为负值，说明模型预测不如直接用均值作为预测值效果好——这是小样本过拟合/欠拟合的典型表现
- **结论**：该数据集规模不满足 DeepPredict 三个模型的最低数据需求，建议选用样本数≥500 的时序数据集进行有意义的模型对比测试
- **建议数据集**：可尝试 Kaggle 上的航空乘客数、太阳黑子数、或 UCI 的电力负荷数据集（样本数均在1000以上）
