# -*- coding: utf-8 -*-
"""Quick test of all three models on pollution data"""
import pandas as pd
import numpy as np
import sys, os, json
sys.path.insert(0, 'src')

LOG = r'C:\Users\XJH\DeepPredict\test_data\pollution_log.txt'

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')

try:
    # Load data
    df = pd.read_csv(r'C:\Users\XJH\DeepPredict\data\pollution.csv')
    y_col = 'pm2.5'
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce').ffill().bfill()
    y = df[y_col].dropna().values
    log(f'Loaded {len(y)} samples, mean={y.mean():.2f}')

    train_size = int(len(y) * 0.7)
    y_train, y_test = y[:train_size], y[train_size:]
    SEQ_LEN = 24  # smaller for speed
    PRED_CHUNK = 6

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    results = {}

    # --- CNN1D ---
    log('\n>>> CNN1D (epochs=5)...')
    try:
        log('Importing CNN1DPredictorV4...')
        from models.cnn1d_model import CNN1DPredictorV4
        log('Creating instance...')
        m = CNN1DPredictorV4()
        log('Calling train()...')
        ok, msg = m.train(y_train, y_train, seq_len=SEQ_LEN, pred_len=PRED_CHUNK, epochs=5)
        log(f'CNN1D train: {ok} - {msg[:80]}')
        if ok:
            p = m.predict(y_train[-SEQ_LEN:], pred_len=min(6, len(y_test)))
            r2 = r2_score(y_test[:len(p)], p)
            log(f'CNN1D quick R2={r2:.4f}')
            results['CNN1D'] = {'status': 'ok', 'r2': float(r2), 'rmse': float(np.sqrt(mean_squared_error(y_test[:len(p)], p))), 'mae': float(mean_absolute_error(y_test[:len(p)], p))}
        else:
            results['CNN1D'] = {'status': 'fail', 'error': msg[:50]}
    except Exception as e:
        import traceback
        log(f'CNN1D ERROR: {e}')
        traceback.print_exc()
        results['CNN1D'] = {'status': 'error', 'error': str(e)[:100]}

    # --- PatchTST ---
    log('\n>>> PatchTST (epochs=5)...')
    try:
        from models.patchtst_model import PatchTSTPredictor
        m = PatchTSTPredictor()
        ok, msg = m.train(y_train, y_train, seq_len=SEQ_LEN, pred_len=PRED_CHUNK, epochs=5)
        log(f'PatchTST train: {ok} - {msg[:80]}')
        if ok:
            p = m.predict(y_train[-SEQ_LEN:], pred_len=min(6, len(y_test)))
            r2 = r2_score(y_test[:len(p)], p)
            log(f'PatchTST quick R2={r2:.4f}')
            results['PatchTST'] = {'status': 'ok', 'r2': float(r2), 'rmse': float(np.sqrt(mean_squared_error(y_test[:len(p)], p))), 'mae': float(mean_absolute_error(y_test[:len(p)], p))}
        else:
            results['PatchTST'] = {'status': 'fail', 'error': msg[:50]}
    except Exception as e:
        import traceback
        log(f'PatchTST ERROR: {e}')
        traceback.print_exc()
        results['PatchTST'] = {'status': 'error', 'error': str(e)[:100]}

    # --- LSTM ---
    log('\n>>> LSTM (epochs=5)...')
    try:
        from models.lstm_model import LSTMPredictor
        m = LSTMPredictor()
        ok, msg = m.train(y_train, y_train, seq_len=SEQ_LEN, epochs=5)
        log(f'LSTM train: {ok} - {msg[:80]}')
        if ok:
            p = m.predict_future(y_train[-SEQ_LEN:], steps=min(6, len(y_test)))
            r2 = r2_score(y_test[:len(p)], p)
            log(f'LSTM quick R2={r2:.4f}')
            results['LSTM'] = {'status': 'ok', 'r2': float(r2), 'rmse': float(np.sqrt(mean_squared_error(y_test[:len(p)], p))), 'mae': float(mean_absolute_error(y_test[:len(p)], p))}
        else:
            results['LSTM'] = {'status': 'fail', 'error': msg[:50]}
    except Exception as e:
        import traceback
        log(f'LSTM ERROR: {e}')
        traceback.print_exc()
        results['LSTM'] = {'status': 'error', 'error': str(e)[:100]}

    # Save results
    with open(r'C:\Users\XJH\DeepPredict\test_data\pollution_result.json', 'w', encoding='utf-8') as f:
        json.dump({'dataset': 'Beijing Hourly Pollution', 'n': len(y), 'target': y_col, 'source': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv', 'seq_len': SEQ_LEN, 'pred_chunk': PRED_CHUNK, 'results': results}, f, ensure_ascii=False, indent=2)
    log('\nDone! Results saved.')
    log(json.dumps(results, ensure_ascii=False, indent=2))

except Exception as e:
    import traceback
    log(f'FATAL: {e}')
    traceback.print_exc()
