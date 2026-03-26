# -*- coding: utf-8 -*-
"""
DeepPredict 自动调试脚本 — 月度太阳黑子 (Monthly Sunspots)
数据集: Monthly Sunspots (monthly_sunspots.csv)
来源: https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv
修复: LSTM 使用 predict_future(steps=...) 而非 predict(pred_len=...)
"""
import pandas as pd
import numpy as np
import sys, io, os, json, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print('=' * 60)
print('DeepPredict 自动调试 — 月度太阳黑子 Sunspots')
print('=' * 60)

# ============================================================
# 第一步：加载数据
# ============================================================
print('\n>>> 步骤1: 加载太阳黑子数据集...')
csv_path = r'C:\Users\XJH\DeepPredict\data\monthly_sunspots.csv'
df = pd.read_csv(csv_path)
print(f'数据: {df.shape[0]} 行 x {df.shape[1]} 列')
print(f'列名: {df.columns.tolist()}')

y_col = 'Sunspots'
df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
df[y_col] = df[y_col].ffill().bfill()
y = df[y_col].values.astype(np.float64)
mask = ~np.isnan(y)
y = y[mask]
print(f'清理后样本数: {len(y)}')

train_size = int(len(y) * 0.7)
y_train, y_test = y[:train_size], y[train_size:]
print(f'训练: {len(y_train)}, 测试: {len(y_test)}')
print(f'  训练集 mean={y_train.mean():.2f}, std={y_train.std():.2f}')
print(f'  测试集  mean={y_test.mean():.2f}, std={y_test.std():.2f}')

# 自动参数
n_train = len(y_train)
SEQ_LEN = min(96, max(12, n_train // 10))
PRED_CHUNK = min(48, max(4, SEQ_LEN // 2))
print(f'\n自动参数: seq_len={SEQ_LEN}, pred_chunk={PRED_CHUNK}')

# ============================================================
# 滚动预测函数 (CNN1D / PatchTST)
# ============================================================
def rolling_predict(model, y_train, y_test, seq_len, pred_chunk, model_name):
    window = y_train[-seq_len:].copy()
    all_preds = []
    n_test = len(y_test)
    i = 0
    while i < n_test:
        remaining = n_test - i
        this_pred_len = min(pred_chunk, remaining)
        if len(window) < seq_len:
            print(f'    [{model_name}] window 不足，跳出')
            break
        if this_pred_len <= 0:
            break
        try:
            chunk_preds = model.predict(window, pred_len=this_pred_len)
        except Exception as e:
            print(f'    [{model_name}] 预测失败 at i={i}: {str(e)[:80]}，缩小 pred_len')
            this_pred_len = max(1, this_pred_len // 2)
            try:
                chunk_preds = model.predict(window, pred_len=this_pred_len)
            except Exception as e2:
                print(f'    [{model_name}] 仍失败: {str(e2)[:80]}，跳过 chunk')
                i += max(1, this_pred_len)
                continue
        preds_list = chunk_preds.tolist() if hasattr(chunk_preds, 'tolist') else list(chunk_preds)
        all_preds.extend(preds_list)
        window = np.concatenate([window[this_pred_len:], y_test[i:i + this_pred_len]])
        i += this_pred_len

    preds_arr = np.array(all_preds[:n_test])
    if len(preds_arr) < n_test:
        print(f'    [{model_name}] 警告：仅预测了 {len(preds_arr)}/{n_test} 点')
    return preds_arr

# ============================================================
# 滚动预测函数 (LSTM - 使用 predict_future)
# ============================================================
def rolling_predict_lstm(model, y_train, y_test, seq_len, pred_chunk, model_name="LSTM"):
    window = y_train[-seq_len:].copy()
    all_preds = []
    n_test = len(y_test)
    i = 0
    while i < n_test:
        remaining = n_test - i
        this_pred_len = min(pred_chunk, remaining)
        if len(window) < seq_len:
            print(f'    [{model_name}] window 不足，跳出')
            break
        if this_pred_len <= 0:
            break
        try:
            # LSTM 用 predict_future(steps=...)
            chunk_preds = model.predict_future(window, steps=this_pred_len)
        except Exception as e:
            print(f'    [{model_name}] 预测失败 at i={i}: {str(e)[:80]}，缩小 steps')
            this_pred_len = max(1, this_pred_len // 2)
            try:
                chunk_preds = model.predict_future(window, steps=this_pred_len)
            except Exception as e2:
                print(f'    [{model_name}] 仍失败: {str(e2)[:80]}，跳过 chunk')
                i += max(1, this_pred_len)
                continue
        preds_list = chunk_preds.tolist() if hasattr(chunk_preds, 'tolist') else list(chunk_preds)
        all_preds.extend(preds_list)
        window = np.concatenate([window[this_pred_len:], y_test[i:i + this_pred_len]])
        i += this_pred_len

    preds_arr = np.array(all_preds[:n_test])
    if len(preds_arr) < n_test:
        print(f'    [{model_name}] 警告：仅预测了 {len(preds_arr)}/{n_test} 点')
    return preds_arr

# ============================================================
# 第二步：测试 CNN1D
# ============================================================
print('\n>>> 步骤2: 测试 CNN1D...')
results = {}
fixes = []
try:
    from models.cnn1d_model import CNN1DPredictorV4
    cnn = CNN1DPredictorV4()
    success, msg = cnn.train(y_train, y_train, seq_len=SEQ_LEN, pred_len=PRED_CHUNK, epochs=20)
    if not success:
        results['cnn1d'] = {'status': 'failed', 'error': f'训练失败: {msg}'}
        print(f'    CNN1D 训练失败: {msg}')
    else:
        preds_cnn = rolling_predict(cnn, y_train, y_test, seq_len=SEQ_LEN, pred_chunk=PRED_CHUNK, model_name="CNN1D")
        if len(preds_cnn) > 0:
            r2_cnn = r2_score(y_test[:len(preds_cnn)], preds_cnn)
            rmse_cnn = np.sqrt(mean_squared_error(y_test[:len(preds_cnn)], preds_cnn))
            mae_cnn = mean_absolute_error(y_test[:len(preds_cnn)], preds_cnn)
            print(f'    CNN1D R2: {r2_cnn:.4f}, RMSE: {rmse_cnn:.4f}, MAE: {mae_cnn:.4f}')
            results['cnn1d'] = {'status': 'success', 'r2': round(r2_cnn, 4), 'rmse': round(rmse_cnn, 4), 'mae': round(mae_cnn, 4)}
        else:
            results['cnn1d'] = {'status': 'failed', 'error': '无有效预测'}
except Exception as e:
    error_msg = f'{type(e).__name__}: {str(e)[:120]}'
    print(f'    CNN1D 异常: {error_msg}')
    results['cnn1d'] = {'status': 'failed', 'error': error_msg}

# ============================================================
# 第三步：测试 PatchTST
# ============================================================
print('\n>>> 步骤3: 测试 PatchTST...')
try:
    from models.patchtst_model import PatchTSTPredictor
    pst = PatchTSTPredictor()
    seq_len_pst = min(SEQ_LEN, max(12, len(y_train) // 5))
    pred_len_pst = min(PRED_CHUNK, max(4, seq_len_pst // 2))
    print(f'    PatchTST 自动参数: seq_len={seq_len_pst}, pred_len={pred_len_pst}')
    success, msg = pst.train(y_train, y_train, seq_len=seq_len_pst, pred_len=pred_len_pst, epochs=20)
    if not success:
        results['patchtst'] = {'status': 'failed', 'error': f'训练失败: {msg}'}
        print(f'    PatchTST 训练失败: {msg}')
    else:
        preds_pst = rolling_predict(pst, y_train, y_test, seq_len=seq_len_pst, pred_chunk=pred_len_pst, model_name="PatchTST")
        if len(preds_pst) > 0:
            r2_pst = r2_score(y_test[:len(preds_pst)], preds_pst)
            rmse_pst = np.sqrt(mean_squared_error(y_test[:len(preds_pst)], preds_pst))
            mae_pst = mean_absolute_error(y_test[:len(preds_pst)], preds_pst)
            print(f'    PatchTST R2: {r2_pst:.4f}, RMSE: {rmse_pst:.4f}, MAE: {mae_pst:.4f}')
            results['patchtst'] = {'status': 'success', 'r2': round(r2_pst, 4), 'rmse': round(rmse_pst, 4), 'mae': round(mae_pst, 4)}
        else:
            results['patchtst'] = {'status': 'failed', 'error': '无有效预测'}
except Exception as e:
    error_msg = f'{type(e).__name__}: {str(e)[:120]}'
    print(f'    PatchTST 异常: {error_msg}')
    results['patchtst'] = {'status': 'failed', 'error': error_msg}

# ============================================================
# 第四步：测试 LSTM (修复版)
# ============================================================
print('\n>>> 步骤4: 测试 LSTM (修复 predict_future)...')
try:
    from models.lstm_model import LSTMPredictor
    lstm = LSTMPredictor()
    # LSTM.train() 不接受 pred_len，使用 seq_len
    success, msg = lstm.train(y_train, y_train, seq_len=SEQ_LEN, epochs=20)
    if not success:
        results['lstm'] = {'status': 'failed', 'error': f'训练失败: {msg}'}
        print(f'    LSTM 训练失败: {msg}')
    else:
        preds_lstm = rolling_predict_lstm(lstm, y_train, y_test, seq_len=SEQ_LEN, pred_chunk=PRED_CHUNK, model_name="LSTM")
        if len(preds_lstm) > 0:
            r2_lstm = r2_score(y_test[:len(preds_lstm)], preds_lstm)
            rmse_lstm = np.sqrt(mean_squared_error(y_test[:len(preds_lstm)], preds_lstm))
            mae_lstm = mean_absolute_error(y_test[:len(preds_lstm)], preds_lstm)
            print(f'    LSTM R2: {r2_lstm:.4f}, RMSE: {rmse_lstm:.4f}, MAE: {mae_lstm:.4f}')
            results['lstm'] = {'status': 'success', 'r2': round(r2_lstm, 4), 'rmse': round(rmse_lstm, 4), 'mae': round(mae_lstm, 4)}
        else:
            results['lstm'] = {'status': 'failed', 'error': '无有效预测'}
except Exception as e:
    error_msg = f'{type(e).__name__}: {str(e)[:120]}'
    print(f'    LSTM 异常: {error_msg}')
    results['lstm'] = {'status': 'failed', 'error': error_msg}

# ============================================================
# 保存结果
# ============================================================
output = {
    'dataset': 'monthly_sunspots',
    'source': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv',
    'n_samples': int(len(y)),
    'n_train': int(len(y_train)),
    'n_test': int(len(y_test)),
    'seq_len': SEQ_LEN,
    'pred_chunk': PRED_CHUNK,
    'results': results,
    'fixes': fixes
}
print('\n=== 完成 ===')
print(json.dumps(output, ensure_ascii=False, indent=2))
with open(r'C:\Users\XJH\DeepPredict\auto_test_sunspots_run2.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print('结果已保存到 auto_test_sunspots_run2.json')
