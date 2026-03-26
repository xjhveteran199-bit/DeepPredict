# OpenCode Review 问题修复记录

## Review 执行时间
2026-03-26

## 审查的文件
- `deeppredict_web.py`
- `src/models/predictor.py`
- `src/models/lstm_model.py`
- `src/models/cnn1d_model.py`

---

## 问题清单与修复

### Issue 1: training_history.csv 格式问题
**严重程度**: P0
**位置**: `deeppredict_web.py:1638, 1652`
**问题描述**: `pd.DataFrame(th)` 在某些边缘情况下可能产生非预期格式
**修复**: 改用 `pd.DataFrame.from_dict(th, orient='columns').to_csv(...)`
**状态**: ✅ 已修复

---

### Issue 2: PatchTST 缺少 target_col 参数
**严重程度**: P1
**位置**: `deeppredict_web.py:1197-1215`
**问题描述**: 调用 PatchTSTPredictor.train() 时未传递 target_col 参数
**修复**: 添加 `target_col=first_target`，其中 `first_target = target_cols_list[0] if isinstance(target_cols_list, list) and target_cols_list else target_col`
**状态**: ✅ 已修复

---

### Issue 3: 多目标时 metrics.json 的 target 字段类型不一致
**严重程度**: P1
**位置**: `deeppredict_web.py:1585`
**问题描述**: 当 target_col 是列表时，直接写入 JSON 可能导致不一致
**修复**: 使用 `target_col_display if isinstance(target_col, list) else target_col`
**状态**: ✅ 已修复

---

### Issue 4: gr.Plot() 返回路径字符串
**严重程度**: P1
**位置**: `deeppredict_web.py:1690`
**问题描述**: Gradio 6.x 的 gr.Plot() 组件接受 matplotlib Figure 对象或 plotly 图表，返回 PNG 路径可能无法正确渲染
**说明**: 当前返回 PNG 文件路径，由 Gradio Image 组件处理。gr.Plot() 在 Gradio 6.x 中对路径支持行为因版本而异，建议未来如发现图表不显示，改用 `gr.Image()` 替换 `gr.Plot()`。
**状态**: ⚠️ 待验证（依赖 Gradio 实际运行测试）

---

### Issue 5: 多变量 tensor shape 检查
**严重程度**: P2
**位置**: `src/models/lstm_model.py`, `src/models/cnn1d_model.py`
**问题描述**: OpenCode 建议检查 EnhancedCNN1D 对多变量输入的 tensor shape 处理
**分析**: 
- LSTM: `input_size = X_train.shape[2]` 自动从 3D 输入获取特征数 ✅
- CNN1D: `input_size=n_features` 直接从 DataFrame 列数获取 ✅
- 多目标输出: LSTMModel 的 output_size 参数已正确传递 ✅
**状态**: ✅ 已实现正确

---

### Issue 6: sklearn 模型多目标支持
**严重程度**: Info
**说明**: sklearn 模型（GradientBoosting, RandomForest）不支持多目标输出。当前实现中，sklearn 模型的 target_col 多选时，会使用第一个目标列或退化为单目标。这是设计选择，非缺陷。

---

## 未修改但已确认正确的部分

1. **LSTM 多目标输出**: `LSTMModel.forward()` 输出 shape `(batch, output_size)` ✅
2. **LSTM 归一化**: 使用联合归一化 `[X, y]`，反归一化时正确提取目标列 ✅
3. **CNN1D 训练历史**: `CNN1DPredictorV4.train_history` 格式完整 ✅
4. **target_col 多选 Dropdown**: `multiselect=True` 已设置 ✅
5. **train_btn.click outputs**: 已添加 `training_history_plot` ✅

---

## 测试建议

1. **多目标 LSTM**: 选择 2-3 个 Y 列，验证预测输出形状正确
2. **训练历史 Plot**: 运行训练后，检查 gr.Plot() 组件是否显示训练曲线
3. **ZIP 导出**: 下载 ZIP，确认 training_history.csv 包含完整的 epoch/MAE/R² 数据
4. **PatchTST**: 确认传入 target_col 后不再报 target_col 相关错误
