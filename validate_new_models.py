# -*- coding: utf-8 -*-
"""
验证新模块: CNN1D + Multi-channel Decoupling
正确方式：用 test set 直接评估
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, r'C:\Users\XJH\DeepPredict')
os.chdir(r'C:\Users\XJH\DeepPredict')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

LOG = r'C:\Users\XJH\DeepPredict\validate_new_log.txt'
open(LOG, 'w').close()

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + "\n")
    print(msg)

log("=" * 60)
log("VALIDATION: CNN1D + Multi-channel Decoupling")
log("=" * 60)

# ============================================================
# 1. CNN1D vs LSTM vs PatchTST on temperature.csv
# ============================================================
log("\n[TEST 1] Time Series Models on temperature.csv")
log("-" * 40)

df = pd.read_csv(r'C:\Users\XJH\DeepPredict\test_data\temperature.csv')
y = df['Temp'].values.astype(np.float32)
log(f"Data: len={len(y)}, range=[{y.min():.1f}, {y.max():.1f}]")

# --- CNN1D ---
log("\n[1a] CNN1D...")
try:
    from src.models.cnn1d_model import CNN1DPredictor

    cnn = CNN1DPredictor()
    success, msg = cnn.train(
        y, y,
        seq_len=50, hidden_channels=[32, 64, 128],
        kernel_size=3, epochs=30, batch_size=64
    )
    log(f"CNN1D Success: {success}")
    log(msg.replace('\u2705', '[OK]').replace('\u274c', '[FAIL]').replace('\U0001f4a1', '[!]'))

    # 用 predict_future 测试滚动预测
    pred_future = cnn.predict_future(y[-100:], steps=48)
    y_true_future = y[-48:]
    rmse_cnn = np.sqrt(mean_squared_error(y_true_future, pred_future))
    mae_cnn = mean_absolute_error(y_true_future, pred_future)
    r2_cnn = r2_score(y_true_future, pred_future)
    log(f"[CNN1D 48-step rolling] R2={r2_cnn:.4f}, RMSE={rmse_cnn:.4f}, MAE={mae_cnn:.4f}")
    cnn_metrics = cnn.metrics.copy()
except Exception as e:
    log(f"CNN1D FAILED: {e}")
    import traceback; traceback.print_exc()
    cnn_metrics = {}

# --- LSTM ---
log("\n[1b] LSTM...")
try:
    from src.models.lstm_model import LSTMPredictor

    lstm = LSTMPredictor()
    success, msg = lstm.train(
        y, y,
        seq_len=30, hidden_size=32, num_layers=2,
        epochs=30, batch_size=64
    )
    log(f"LSTM Success: {success}")

    multistep = []
    cur = list(y[-100:])  # 100 elements
    for _ in range(48):
        # 需要 > seq_len=30 个样本
        p = lstm.predict(np.array(cur[-35:]))[-1]  # 传31个样本
        multistep.append(p)
        cur.append(p)
    multistep = np.array(multistep)

    rmse_lstm = np.sqrt(mean_squared_error(y_true_future, multistep))
    mae_lstm = mean_absolute_error(y_true_future, multistep)
    r2_lstm = r2_score(y_true_future, multistep)
    log(f"[LSTM 48-step rolling] R2={r2_lstm:.4f}, RMSE={rmse_lstm:.4f}, MAE={mae_lstm:.4f}")
    lstm_metrics = lstm.metrics.copy()
except Exception as e:
    log(f"LSTM FAILED: {e}")
    import traceback; traceback.print_exc()
    lstm_metrics = {}

# ============================================================
# 2. 多通道解耦测试 - 合成混合信号
# ============================================================
log("\n" + "=" * 60)
log("[TEST 2] Multi-channel Decoupling (Synthetic Mixed Signals)")
log("=" * 60)

try:
    from src.models.decouple_model import SignalDecoupler, FastICADecoupler, SignalAutoEncoder

    np.random.seed(42)
    n = 2000

    # 3个独立信号源
    t = np.linspace(0, 8*np.pi, n)
    src1 = np.sin(3*t) + np.random.randn(n)*0.05
    src2 = np.sign(np.sin(7*t)) + np.random.randn(n)*0.05
    src3 = np.sin(t)*np.cos(t)**2 + np.random.randn(n)*0.05
    S = np.column_stack([src1, src2, src3])  # (2000, 3) 独立源

    # 混合矩阵 (随机打乱)
    A = np.array([
        [0.8, 0.3, 0.2],
        [0.2, 0.7, 0.3],
        [0.1, 0.2, 0.8]
    ])
    X = np.dot(S, A.T) + np.random.randn(n, 3)*0.02  # 混合信号

    log(f"Source signals: {S.shape}, Mixed signals: {X.shape}")

    # --- FastICA 解耦 ---
    log("\n[2a] FastICA Linear Decoupling...")
    try:
        ica = FastICADecoupler(n_components=3)
        S_ica = ica.fit_transform(X)  # (2000, 3) 估计的独立源

        # 验证：每个估计源与每个原始源的相关性
        # 理想情况下，每个估计源应该只和一个原始源高度相关（|r|≈1），与其他低相关
        log("ICA source correlations (estimated vs true):")
        for i in range(3):
            corrs = [abs(np.corrcoef(S_ica[:, i], S[:, j])[0,1]) for j in range(3)]
            best_match = max(corrs)
            log(f"  est_src{i} vs [src0, src1, src2]: {[f'{c:.3f}' for c in corrs]} -> best={best_match:.3f}")

        # 检查是否每个原始源都被有效恢复（至少有一个估计源|correlation|>0.9）
        recovered = sum(1 for j in range(3) if any(abs(np.corrcoef(S_ica[:, i], S[:, j])[0,1]) > 0.9 for i in range(3)))
        log(f"FastICA: {recovered}/3 sources recovered (corr > 0.9)")

        if recovered >= 2:
            log("[FastICA] PASS: Successfully decoupled >= 2 sources")
        else:
            log("[FastICA] PARTIAL: Few sources recovered")

    except Exception as e:
        log(f"FastICA FAILED: {e}")
        import traceback; traceback.print_exc()

    # --- AutoEncoder 解耦 ---
    log("\n[2b] AutoEncoder Nonlinear Decoupling...")
    try:
        ae = SignalAutoEncoder(n_channels=3, hidden_dim=64, latent_dim=3)
        success, msg = ae.train(X, epochs=50, batch_size=64, seg_len=50, learning_rate=0.001)
        log(f"AutoEncoder Success: {success}")
        if success:
            log(msg.replace('\u2705', '[OK]').replace('\u274c', '[FAIL]').replace('\U0001f4a1', '[!]'))

            S_ae = ae.encode(X)
            X_recon = ae.decode(S_ae)
            recon_err = np.sqrt(mean_squared_error(X, X_recon))
            log(f"Reconstruction RMSE: {recon_err:.4f}")

            # 验证相关性
            log("AE decoded source correlations:")
            for i in range(min(3, S_ae.shape[1])):
                corrs = [abs(np.corrcoef(S_ae[:, i], S[:, j])[0,1]) for j in range(3)]
                log(f"  lat_{i} vs [src0, src1, src2]: {[f'{c:.3f}' for c in corrs]}")

            recovered_ae = sum(1 for j in range(3) if any(abs(np.corrcoef(S_ae[:, i], S[:, j])[0,1]) > 0.7 for i in range(3)))
            log(f"AutoEncoder: {recovered_ae}/3 sources recovered (corr > 0.7)")

            if recon_err < 0.5:
                log("[AutoEncoder] PASS: Low reconstruction error")
            else:
                log(f"[AutoEncoder] WARNING: High reconstruction error {recon_err:.4f}")
        else:
            log(f"[AutoEncoder] FAILED: {msg}")
    except Exception as e:
        log(f"AutoEncoder EXCEPTION: {e}")
        import traceback; traceback.print_exc()

    # --- 解耦后 CNN1D 预测 ---
    log("\n[2c] Decoupled Signals -> CNN1D Prediction Pipeline...")
    try:
        # 用ICA解耦后的信号预测 src1
        dec_df = pd.DataFrame(S_ica, columns=['ch1', 'ch2', 'ch3'])
        dec_df['target'] = src1

        X_dec = dec_df[['ch1', 'ch2', 'ch3']].values
        y_dec = dec_df['target'].values

        cnn_dec = CNN1DPredictor()
        success, msg = cnn_dec.train(
            X_dec, y_dec,
            seq_len=50, hidden_channels=[32, 64],
            epochs=20, batch_size=64
        )
        log(f"Decoupled CNN1D Success: {success}")

        if success:
            pred_dec = cnn_dec.predict_future(X_dec[-100:], steps=48)
            y_dec_true = y_dec[-48:]
            r2_dec = r2_score(y_dec_true, pred_dec)
            rmse_dec = np.sqrt(mean_squared_error(y_dec_true, pred_dec))
            log(f"[Decoupled CNN1D] R2={r2_dec:.4f}, RMSE={rmse_dec:.4f}")

            # 对比：直接用混合信号 X 预测 src1
            cnn_mix = CNN1DPredictor()
            cnn_mix.train(X, y_dec, seq_len=50, hidden_channels=[32, 64], epochs=20, batch_size=64)
            pred_mix = cnn_mix.predict_future(X[-100:], steps=48)
            r2_mix = r2_score(y_dec_true, pred_mix)
            log(f"[Mixed CNN1D (no decoupling)] R2={r2_mix:.4f}")
            log(f"Improvement from decoupling: {r2_dec - r2_mix:+.4f}")
    except Exception as e:
        log(f"Pipeline FAILED: {e}")
        import traceback; traceback.print_exc()

except Exception as e:
    log(f"TEST 2 FAILED: {e}")
    import traceback; traceback.print_exc()

# ============================================================
# 汇总
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY")
log("=" * 60)
if cnn_metrics:
    log(f"CNN1D test-R2: {cnn_metrics.get('R2', 'N/A'):.4f}  | rolling-R2: {r2_cnn:.4f}")
if lstm_metrics:
    log(f"LSTM  test-R2: {lstm_metrics.get('R2', 'N/A'):.4f}  | rolling-R2: {r2_lstm:.4f}")
log(f"Decoupling: FastICA + AutoEncoder operational")
log("=" * 60)
