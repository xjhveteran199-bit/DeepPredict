# DeepPredict 自动调试简报
**时间**: 2026-03-24 12:01
**数据集**: 墨尔本每日最高温度 (Daily Max Temperatures in Melbourne) (n=3650)
**数据来源**: https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-max-temperatures.csv

## 模型结果

| 模型 | R² | RMSE | MAE | 状态 |
|------|-----|------|-----|------|
| CNN1D | 0.2767 | 5.1460 | 3.6309 | ✅ |
| PatchTST | 0.0855 | 5.7862 | 4.2746 | ✅ |
| LSTM | 0.1062 | 5.7202 | 4.7941 | ✅ |

## 修复记录
- **CNN1D**: 类名从 CNN1DPredictor 更新为 CNN1DPredictorV4（API 变化）
- **LSTM**: train() 移除无效的 pred_len 参数；predict_future() 补充缺失的 window 参数

## 备注
- 三个模型全部成功完成，CNN1D 表现最优
- 相比上次太阳黑子数据集（PatchTST 最优），本次最高温度数据集由 CNN1D 主导，说明 CNN1D 对温度类季节性数据有较好适应性
- PatchTST R² 偏低（0.0855），可能需要更多 epochs 或调参
- 数据集总样本 3650，训练 2555，测试 1095（70/30 分割）
- 温度均值 19.85°C，标准差 6.12°C，季节性波动明显
