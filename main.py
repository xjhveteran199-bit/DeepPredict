"""
DeepPredict - 低门槛深度学习预测工具
主入口文件
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# 解决 Windows 上的高DPI问题
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "Round"

# 添加 src 目录到路径
APP_ROOT = Path(__file__).parent
sys.path.insert(0, str(APP_ROOT / "src"))

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
    logger.info("DeepPredict v1.0 启动")
    
    # 启用高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("DeepPredict")
    app.setApplicationVersion("1.0.0")
    
    # 设置样式
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    logger.info("主窗口已显示")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
