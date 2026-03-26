# -*- coding: utf-8 -*-
"""
DeepPredict 自动调试脚本 — 月度太阳黑子 (Monthly Sunspots)
数据集: Monthly Sunspots (monthly_sunspots.csv)
来源: https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv
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

# 去除0值（太阳黑子数为0可能是数据问题）
# 保持原样，不处理

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
# 滚动预测函数
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
    return preds_arr

# ============================================================
# 第二步：测试 CNN1D
# ============================================================
results = {}
print('\n>>> 步骤2: 测试 CNN1D...')
try:
    from models.cnn1d_model import CNN1DPredictor
    cnn = CNN1DPredictor()
    cnn_seq = min(SEQ_LEN, max(12, len(y_train) // 5))
    cnn_pred = min(PRED_CHUNK, max(4, cnn_seq // 2))
    print(f'    自动参数: seq_len={cnn_seq}, pred_len={cnn_pred}')
    
    # 数据归一化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).ravel()
    
    success, msg = cnn.train(y_train_scaled, y_train_scaled, seq_len=cnn_seq, pred_len=cnn_pred, epochs=20)
    if not success:
        print(f'    CNN1D 训练失败: {msg}')
        results['cnn1d'] = {'status': 'failed', 'error': msg}
    else:
        preds_cnn = rolling_predict(cnn, y_train_scaled, y_test_scaled, cnn_seq, cnn_pred, 'CNN1D')
        if len(preds_cnn) > 0:
            # 反归一化
            preds_cnn_orig = scaler.inverse_transform(preds_cnn.reshape(-1, 1)).ravel()
            y_test_orig = y_test[:len(preds_cnn)]
            r2_cnn = r2_score(y_test_orig, preds_cnn_orig)
            rmse_cnn = np.sqrt(mean_squared_error(y_test_orig, preds_cnn_orig))
            mae_cnn = mean_absolute_error(y_test_orig, preds_cnn_orig)
            print(f'    CNN1D 结果: R2={r2_cnn:.4f}, RMSE={rmse_cnn:.4f}, MAE={mae_cnn:.4f}')
            results['cnn1d'] = {'status': 'ok', 'r2': round(r2_cnn,4), 'rmse': round(rmse_cnn,4), 'mae': round(mae_cnn,4)}
        else:
            results['cnn1d'] = {'status': 'failed', 'error': 'no predictions'}
except Exception as e:
    print(f'    CNN1D 异常: {str(e)[:100]}')
    print(traceback.format_exc())
    results['cnn1d'] = {'status': 'failed', 'error': str(e)[:100]}

# ============================================================
# 第三步：测试 PatchTST
# ============================================================
print('\n>>> 步骤3: 测试 PatchTST...')
try:
    from models.patchtst_model import PatchTSTPredictor
    pst = PatchTSTPredictor()
    pst_seq = min(SEQ_LEN, max(12, len(y_train) // 5))
    pst_pred = min(PRED_CHUNK, max(4, pst_seq // 2))
    print(f'    自动参数: seq_len={pst_seq}, pred_len={pst_pred}')
    
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).ravel()
    
    success, msg = pst.train(y_train_scaled, y_train_scaled, seq_len=pst_seq, pred_len=pst_pred, epochs=20)
    if not success:
        print(f'    PatchTST 训练失败: {msg}')
        results['patchtst'] = {'status': 'failed', 'error': msg}
    else:
        preds_pst = rolling_predict(pst, y_train_scaled, y_test_scaled, pst_seq, pst_pred, 'PatchTST')
        if len(preds_pst) > 0:
            preds_pst_orig = scaler.inverse_transform(preds_pst.reshape(-1, 1)).ravel()
            y_test_orig = y_test[:len(preds_pst)]
            r2_pst = r2_score(y_test_orig, preds_pst_orig)
            rmse_pst = np.sqrt(mean_squared_error(y_test_orig, preds_pst_orig))
            mae_pst = mean_absolute_error(y_test_orig, preds_pst_orig)
            print(f'    PatchTST 结果: R2={r2_pst:.4f}, RMSE={rmse_pst:.4f}, MAE={mae_pst:.4f}')
            results['patchtst'] = {'status': 'ok', 'r2': round(r2_pst,4), 'rmse': round(rmse_pst,4), 'mae': round(mae_pst,4)}
        else:
            results['patchtst'] = {'status': 'failed', 'error': 'no predictions'}
except Exception as e:
    print(f'    PatchTST 异常: {str(e)[:100]}')
    print(traceback.format_exc())
    results['patchtst'] = {'status': 'failed', 'error': str(e)[:100]}

# ============================================================
# 第四步：测试 LSTM
# ============================================================
print('\n>>> 步骤4: 测试 LSTM...')
try:
    from models.lstm_model import LSTMPredictor
    lstm = LSTMPredictor()
    lstm_seq = min(SEQ_LEN, max(12, len(y_train) // 5))
    lstm_pred = min(PRED_CHUNK, max(4, lstm_seq // 2))
    print(f'    自动参数: seq_len={lstm_seq}, pred_len={lstm_pred}')
    
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).ravel()
    
    success, msg = lstm.train(y_train_scaled, y_train_scaled, seq_len=lstm_seq, pred_len=lstm_pred, epochs=20)
    if not success:
        print(f'    LSTM 训练失败: {msg}')
        results['lstm'] = {'status': 'failed', 'error': msg}
    else:
        # LSTM 用 predict_future (steps 参数)
        window = y_train_scaled[-lstm_seq:].copy()
        all_preds = []
        n_test = len(y_test_scaled)
        i = 0
        while i < n_test:
            remaining = n_test - i
            this_pred_len = min(lstm_pred, remaining)
            if len(window) < lstm_seq:
                print(f'    [LSTM] window 不足，跳出')
                break
            if this_pred_len <= 0:
                break
            try:
                chunk_preds = lstm.predict_future(steps=this_pred_len)
            except Exception as e:
                print(f'    [LSTM] 预测失败 at i={i}: {str(e)[:80]}，缩小 steps')
                this_pred_len = max(1, this_pred_len // 2)
                try:
                    chunk_preds = lstm.predict_future(steps=this_pred_len)
                except Exception as e2:
                    print(f'    [LSTM] 仍失败: {str(e2)[:80]}，跳过 chunk')
                    i += max(1, this_pred_len)
                    continue
            preds_list = chunk_preds.tolist() if hasattr(chunk_preds, 'tolist') else list(chunk_preds)
            all_preds.extend(preds_list)
            window = np.concatenate([window[this_pred_len:], y_test_scaled[i:i + this_pred_len]])
            i += this_pred_len
        
        preds_lstm = np.array(all_preds[:n_test])
        if len(preds_lstm) > 0:
            preds_lstm_orig = scaler.inverse_transform(preds_lstm.reshape(-1, 1)).ravel()
            y_test_orig = y_test[:len(preds_lstm)]
            r2_lstm = r2_score(y_test_orig, preds_lstm_orig)
            rmse_lstm = np.sqrt(mean_squared_error(y_test_orig, preds_lstm_orig))
            mae_lstm = mean_absolute_error(y_test_orig, preds_lstm_orig)
            print(f'    LSTM 结果: R2={r2_lstm:.4f}, RMSE={rmse_lstm:.4f}, MAE={mae_lstm:.4f}')
            results['lstm'] = {'status': 'ok', 'r2': round(r2_lstm,4), 'rmse': round(rmse_lstm,4), 'mae': round(mae_lstm,4)}
        else:
            results['lstm'] = {'status': 'failed', 'error': 'no predictions'}
except Exception as e:
    print(f'    LSTM 异常: {str(e)[:100]}')
    print(traceback.format_exc())
    results['lstm'] = {'status': 'failed', 'error': str(e)[:100]}

# ============================================================
# 第五步：输出汇总
# ============================================================
print('\n' + '=' * 60)
print('模型结果汇总')
print('=' * 60)
for name, r in results.items():
    if r['status'] == 'ok':
        print(f'{name}: R2={r["r2"]}, RMSE={r["rmse"]}, MAE={r["mae"]}')
    else:
        print(f'{name}: FAILED — {r["error"]}')

# 保存结果
results_file = r'C:\Users\XJH\DeepPredict\auto_test_sunspots_results.json'
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump({
        'dataset': 'monthly_sunspots',
        'source': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv',
        'n_samples': int(len(y)),
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'results': results
    }, f, ensure_ascii=False, indent=2)
print(f'\n结果已保存: {results_file}')
