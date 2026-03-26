# -*- coding: utf-8 -*-
"""Full DeepPredict test on Beijing Hourly Pollution - rolling evaluation"""
import pandas as pd
import numpy as np
import sys, os, json
sys.path.insert(0, 'src')

LOG = r'C:\Users\XJH\DeepPredict\test_data\pollution_log.txt'

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    log('=' * 60)
    log('DeepPredict 北京空气质量 PM2.5 完整测试')
    log('=' * 60)

    # Load data
    df = pd.read_csv(r'C:\Users\XJH\DeepPredict\data\pollution.csv')
    y_col = 'pm2.5'
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce').ffill().bfill()
    y = df[y_col].dropna().values.astype(np.float64)
    log(f'Loaded {len(y)} samples, mean={y.mean():.2f}, std={y.std():.2f}')

    train_size = int(len(y) * 0.7)
    y_train, y_test = y[:train_size], y[train_size:]
    log(f'Train: {len(y_train)}, Test: {len(y_test)}')

    SEQ_LEN = min(96, max(24, len(y_train) // 10))
    PRED_CHUNK = min(48, max(6, SEQ_LEN // 2))
    EPOCHS = 20
    log(f'Params: seq_len={SEQ_LEN}, pred_chunk={PRED_CHUNK}, epochs={EPOCHS}')

    def rolling_predict(model, y_train, y_test, seq_len, pred_chunk, name):
        window = y_train[-seq_len:].copy()
        all_preds = []
        n_test = len(y_test)
        i = 0
        while i < n_test:
            remaining = n_test - i
            this_len = min(pred_chunk, remaining)
            if len(window) < seq_len or this_len <= 0:
                break
            try:
                chunk = model.predict(window, pred_len=this_len)
            except Exception as e:
                log(f'  [{name}] predict error at {i}: {str(e)[:50]}, retry smaller...')
                this_len = max(1, this_len // 2)
                try:
                    chunk = model.predict(window, pred_len=this_len)
                except:
                    i += this_len
                    continue
            preds = chunk.tolist() if hasattr(chunk, 'tolist') else list(chunk)
            all_preds.extend(preds)
            window = np.concatenate([window[this_len:], y_test[i:i+this_len]])
            i += this_len
        return np.array(all_preds[:n_test])

    results = {}

    # --- CNN1D ---
    log('\n>>> CNN1D (epochs=20)...')
    try:
        from models.cnn1d_model import CNN1DPredictorV4
        m = CNN1DPredictorV4()
        ok, msg = m.train(y_train, y_train, seq_len=SEQ_LEN, pred_len=PRED_CHUNK, epochs=EPOCHS)
        log(f'CNN1D train: {ok}')
        if ok:
            preds = rolling_predict(m, y_train, y_test, SEQ_LEN, PRED_CHUNK, 'CNN1D')
            n = min(len(preds), len(y_test))
            r2 = r2_score(y_test[:n], preds[:n])
            rmse = np.sqrt(mean_squared_error(y_test[:n], preds[:n]))
            mae = mean_absolute_error(y_test[:n], preds[:n])
            log(f'CNN1D: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}')
            results['CNN1D'] = {'status': 'ok', 'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae)}
        else:
            results['CNN1D'] = {'status': 'fail', 'error': msg[:80]}
    except Exception as e:
        import traceback
        log(f'CNN1D ERROR: {e}')
        traceback.print_exc()
        results['CNN1D'] = {'status': 'error', 'error': str(e)[:100]}

    # --- PatchTST ---
    log('\n>>> PatchTST (epochs=20)...')
    try:
        from models.patchtst_model import PatchTSTPredictor
        m = PatchTSTPredictor()
        ok, msg = m.train(y_train, y_train, seq_len=SEQ_LEN, pred_len=PRED_CHUNK, epochs=EPOCHS)
        log(f'PatchTST train: {ok}')
        if ok:
            preds = rolling_predict(m, y_train, y_test, SEQ_LEN, PRED_CHUNK, 'PatchTST')
            n = min(len(preds), len(y_test))
            r2 = r2_score(y_test[:n], preds[:n])
            rmse = np.sqrt(mean_squared_error(y_test[:n], preds[:n]))
            mae = mean_absolute_error(y_test[:n], preds[:n])
            log(f'PatchTST: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}')
            results['PatchTST'] = {'status': 'ok', 'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae)}
        else:
            results['PatchTST'] = {'status': 'fail', 'error': msg[:80]}
    except Exception as e:
        import traceback
        log(f'PatchTST ERROR: {e}')
        traceback.print_exc()
        results['PatchTST'] = {'status': 'error', 'error': str(e)[:100]}

    # --- LSTM ---
    log('\n>>> LSTM (epochs=20)...')
    try:
        from models.lstm_model import LSTMPredictor
        m = LSTMPredictor()
        ok, msg = m.train(y_train, y_train, seq_len=SEQ_LEN, epochs=EPOCHS)
        log(f'LSTM train: {ok}')
        if ok:
            # LSTM uses predict_future
            window = y_train[-SEQ_LEN:].copy()
            all_preds = []
            n_test = len(y_test)
            i = 0
            while i < n_test:
                remaining = n_test - i
                this_len = min(PRED_CHUNK, remaining)
                if len(window) < SEQ_LEN or this_len <= 0:
                    break
                try:
                    chunk = m.predict_future(window, steps=this_len)
                except Exception as e:
                    log(f'  [LSTM] predict error at {i}: {str(e)[:50]}, retry smaller...')
                    this_len = max(1, this_len // 2)
                    try:
                        chunk = m.predict_future(window, steps=this_len)
                    except:
                        i += this_len
                        continue
                preds = chunk.tolist() if hasattr(chunk, 'tolist') else list(chunk)
                all_preds.extend(preds)
                window = np.concatenate([window[this_len:], y_test[i:i+this_len]])
                i += this_len
            preds_arr = np.array(all_preds[:n_test])
            n = min(len(preds_arr), len(y_test))
            r2 = r2_score(y_test[:n], preds_arr[:n])
            rmse = np.sqrt(mean_squared_error(y_test[:n], preds_arr[:n]))
            mae = mean_absolute_error(y_test[:n], preds_arr[:n])
            log(f'LSTM: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}')
            results['LSTM'] = {'status': 'ok', 'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae)}
        else:
            results['LSTM'] = {'status': 'fail', 'error': msg[:80]}
    except Exception as e:
        import traceback
        log(f'LSTM ERROR: {e}')
        traceback.print_exc()
        results['LSTM'] = {'status': 'error', 'error': str(e)[:100]}

    # Summary
    log('\n' + '=' * 60)
    log('模型汇总结果:')
    for name, res in results.items():
        if res['status'] == 'ok':
            log(f'  {name}: R2={res["r2"]:.4f}, RMSE={res["rmse"]:.4f}, MAE={res["mae"]:.4f} ✅')
        else:
            log(f'  {name}: ❌ {res.get("error","")[:50]}')
    log('=' * 60)

    # Save
    with open(r'C:\Users\XJH\DeepPredict\test_data\pollution_result.json', 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': 'Beijing Hourly Pollution',
            'n': int(len(y)),
            'n_train': int(train_size),
            'n_test': int(len(y) - train_size),
            'target': y_col,
            'source': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv',
            'seq_len': SEQ_LEN,
            'pred_chunk': PRED_CHUNK,
            'epochs': EPOCHS,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    log('DONE')

except Exception as e:
    import traceback
    log(f'FATAL: {e}')
    traceback.print_exc()
