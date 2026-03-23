# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=['C:\\Users\\XJH\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\torch\\lib', 'C:\\Users\\XJH\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\PyQt5\\Qt5\\bin', 'C:\\Users\\XJH\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\numpy.libs', 'C:\\Users\\XJH\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\scipy.libs'],
    binaries=[],
    datas=[],
    hiddenimports=['torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils.data', 'torch._utils', 'torch.classes', 'sklearn.ensemble._forest', 'sklearn.ensemble._gb', 'sklearn.tree._classes', 'sklearn.linear_model._base', 'sklearn.metrics._regression', 'sklearn.metrics._classification', 'sklearn.preprocessing._data', 'sklearn.preprocessing._label', 'sklearn.model_selection._split', 'scipy.special._ellip_harm_2', 'scipy.linalg._fblas', 'scipy.linalg._flapack', 'scipy._lib._ccallback', 'pandas._libs.tslibs.np_datetime', 'pandas._libs.tslibs.timedeltas', 'pandas._libs.hashtable', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'PyQt5.sip', 'ui.main_window', 'core.data_loader', 'core.task_router', 'models.predictor', 'models.lstm_model', 'models.patchtst_model'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DeepPredict',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DeepPredict',
)
