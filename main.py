"""
DeepPredict - 低门槛深度学习预测工具
主入口文件
"""

import sys
import os
from pathlib import Path

# ====== 关键修复：PyInstaller frozen 环境下 DLL 搜索路径 ======
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    torch_lib = Path(BASE_DIR) / "torch" / "lib"
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    python_dir = Path(sys.executable).parent
    os.add_dll_directory(str(python_dir))
    os.environ["PYTORCH_JIT"] = "0"

# 解决 Windows 高DPI问题
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "Round"

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# frozen=True 表示 PyInstaller 打包环境
if getattr(sys, 'frozen', False):
    APP_ROOT = Path(sys._MEIPASS)
else:
    APP_ROOT = Path(__file__).parent

SRC_ROOT = APP_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(APP_ROOT))

from ui.main_window import MainWindow


def setup_logging():
    """初始化日志"""
    log_dir = APP_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "deeppredict.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()
    logger.info("DeepPredict v1.01 启动")
    
    # 启用高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("DeepPredict")
    app.setApplicationVersion("1.0.0")
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    logger.info("主窗口已显示")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
