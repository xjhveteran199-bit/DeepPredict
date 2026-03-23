# 更新日志 / Changelog

## v1.04 — 2026-03-23

### 新增

**界面 & 交互**
- **图表定制描述框**：用户可用自然语言描述想要的图表类型（双轴/置信区间/散点/柱状等）
- **5种图表模板**：标准折线图、双轴图、置信区间图、散点图、柱状图，纯规则匹配无需 AI
- **一键下载结果包**：训练完成后自动生成 zip 包，含 forecast.png + forecast_data.csv + metrics.json
- **未来趋势预测图**：训练完成后自动展示历史数据（蓝色）+ 预测数据（橙色）折线图
- **未来 N 步预测值**：用户可自定义预测步数，默认30步，展示具体数值
- **自然语言趋势总结**：自动生成"上升/下降趋势、幅度、预测区间、稳定性"等结论

**数据处理**
- **数据解耦器 DataDecoupler**：自动识别列类型（日期/类别/数值/文本），分别预处理后合并
- **数据结构自动检测**：单变量时序、成组时序、多变量数据自动分类
- **成组时序支持**：自动识别 K-xxx、Glu-xxx 等成组列名，针对性处理

**模型优化（小数据鲁棒性）**
- **CNN1D**：残差连接 + 自适应池化（mean+max）+ Huber Loss + Early Stopping，小样本更稳定
- **LSTM**：LayerNorm + 更深FC头（Skip Connection）+ Gradient Clipping（max_norm=1.0）+ ReduceLROnPlateau + Early Stopping
- **PatchTST**：早期停止 + 自适应模型复杂度（d_model=64/n_layers=2 for 小数据）+ Warmup scheduler
- **EnhancedCNN1D**：多尺度卷积（kernel=3/5/7）+ SE通道注意力 + 残差块×2，适合多变量复杂数据

### 修复
- PatchTST `predict_future()` 缺少 `X` 参数报错问题
- Sklearn 模型新增 `predict_future()` 滚动预测方法
- LSTM `val_mse` 引用在赋值之前的问题
- CNN1D 小数据集（seq_len < 16）除零崩溃问题
- LSTM predict_future 参数名（`steps` vs `n_future`）
- PatchTST 数据不足（n < seq_len+pred_len+50）时的报错信息

---

## v1.03 — 2026-03-23

### 新增
- **X 自变量选择**：用户上传 CSV 后可自主选择 X 列（时间/自变量列）和 Y 列（目标列）
- **支付宝收款系统**：国内用户可扫码购买积分，无需 Stripe
- **积分系统**：注册送 100 积分，按次扣积分
- **FastAPI 后端**：`/api/*` REST 接口，含用户管理、积分、支付、Webhook
- **NanoBnana 风格首页**：暗色科技风，定价 ¥7 / ¥28 / ¥49
- **Docker 部署支持**：Dockerfile + docker-compose

### 修复
- LSTM predict_future 参数名修正
- CNN1D 小数据集保护

---

## v1.02 — 2026-03-22

- 初始版本
- 支持 PatchTST / LSTM / GradientBoosting / RandomForest
