# 更新日志 / Changelog

## v1.03 — 2026-03-23

### 新增
- **X 自变量选择**：用户上传 CSV 后可自主选择 X 列（时间/自变量列）和 Y 列（目标列）
- **支付宝收款系统**：国内用户可扫码购买积分，无需 Stripe
- **积分系统**：注册送 100 积分，按次扣积分（PatchTST 20分 / LSTM 15分 / GradientBoosting 5分）
- **FastAPI 后端**：`/api/*` REST 接口，含用户管理、积分、支付、Webhook
- **NanoBnana 风格首页**：暗色科技风，定价 ¥7 / ¥28 / ¥49
- **Docker 部署支持**：Dockerfile + docker-compose，一键部署

### 修复
- LSTM predict_future 参数名修正（`n_future` → `steps`）
- CNN1D 小数据集保护（seq_len < 16 时自动调整）
- 模块 import 路径问题

### 变更
- Gradio 界面步骤编号更新：① X列 → ② Y列 → ③ 模式 → ④ 模型 → ⑤ 需求描述

---

## v1.02 — 2026-03-22

- 初始版本
- 支持 PatchTST / LSTM / GradientBoosting / RandomForest
- 智能数据分析 + 自动模型推荐
