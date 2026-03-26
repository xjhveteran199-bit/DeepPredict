# ChronoML v1.04 Release Notes

## What's New in v1.04

### 🧠 模型升级
- **CNN1D v4**：Patch embedding + 可学习位置编码 + 改进池化层（Conv1d 替代裸 mean）
- **RevIN Bug 修复**：修复了 per-batch 归一化的数据泄露问题，改为 per-sample instance normalization
- **LSTM 稳定性**：Xavier 初始化 + Gradient Clipping + ReduceLROnPlateau + Early Stopping
- **CNN1D 小数据集保护**：自动调整 seq_len/pred_len，防止除零崩溃

### 🔧 数据解耦 v2
- **智能列类型识别**：CSV 上传后自动区分数值列 / 类别列 / 日期列
- **日期特征自动提取**：year, month, day, dayofweek, hour + sin/cos 周期编码
- **不规则采样检测**：`detect_irregular_sampling()` 自动判断是否需要重采样
- **LabelEncoder 编码**：类别列自动编码处理

### 🌐 变现功能
- **积分制 Credits**：注册送 100 积分，按模型消耗扣费
- **Stripe Checkout**：国际信用卡支付
- **数据库设计**：users / credit_transactions / predictions 表

### 📊 预测性能（最新测试）
| 数据集 | 样本数 | CNN1D R² |
|--------|--------|----------|
| 每日最低温度 | 3650 | 0.5065 ✅ |
| 北京空气质量 | 41757 | 0.5669 ✅ |

---

## v1.05 Roadmap
- 多变量 CNN1D 支持（per-channel 归一化 + 日期特征注入）
- 不规则时序自动重采样
- 1D-CNN 新架构
