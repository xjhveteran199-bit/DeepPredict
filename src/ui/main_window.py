"""
DeepPredict 主窗口UI
"""

import os
import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QFileDialog,
    QTableWidget, QTableWidgetItem, QComboBox, QGroupBox,
    QProgressBar, QMessageBox, QSplitter, QFrame,
    QApplication, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

from core.data_loader import DataLoader
from core.task_router import TaskRouter
from models.predictor import Predictor

logger = logging.getLogger(__name__)


class WorkerThread(QThread):
    """后台工作线程"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(True, str(result) if result else "完成")
        except Exception as e:
            self.finished.emit(False, str(e))


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 核心组件
        self.data_loader = DataLoader()
        self.task_router = TaskRouter()
        self.predictor = Predictor()
        
        # 状态
        self.current_data: Optional[dict] = None
        
        self._init_ui()
        self._init_styles()
        
        logger.info("主窗口初始化完成")
    
    def _init_ui(self):
        """初始化UI"""
        self.setWindowTitle("DeepPredict - 智能预测工具 v1.04 (PatchTST + CNN1D-V4 + Decouple)")
        self.setMinimumSize(1200, 800)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # ===== 顶部区域：Logo + 标题 =====
        header_layout = QHBoxLayout()
        title_label = QLabel("🧠 DeepPredict")
        title_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        subtitle_label = QLabel("低门槛深度学习预测工具 | 研究生友好版")
        subtitle_label.setFont(QFont("Microsoft YaHei", 10))
        subtitle_label.setStyleSheet("color: gray;")
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # ===== 分割器：左右区域 =====
        splitter = QSplitter(Qt.Horizontal)
        
        # --- 左侧面板 ---
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # --- 右侧面板 ---
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
        # ===== 底部状态栏 =====
        self.status_label = QLabel("📁 请导入CSV数据文件")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        status_layout.addStretch()
        main_layout.addLayout(status_layout)
    
    def _create_left_panel(self) -> QFrame:
        """创建左侧面板"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 1. 数据导入区
        import_group = QGroupBox("📂 数据导入")
        import_layout = QVBoxLayout(import_group)
        
        self.import_btn = QPushButton("选择 CSV 文件")
        self.import_btn.setMinimumHeight(40)
        self.import_btn.clicked.connect(self._on_import_file)
        
        self.file_label = QLabel("未选择文件")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: gray; padding: 5px;")
        
        import_layout.addWidget(self.import_btn)
        import_layout.addWidget(self.file_label)
        layout.addWidget(import_group)
        
        # 2. 数据预览区
        preview_group = QGroupBox("📊 数据预览")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(200)
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        preview_layout.addWidget(self.preview_table)
        
        self.data_info_label = QLabel("暂无数据")
        self.data_info_label.setStyleSheet("color: #666; font-size: 12px;")
        preview_layout.addWidget(self.data_info_label)
        
        layout.addWidget(preview_group)
        
        # 3. 目标列选择
        target_group = QGroupBox("🎯 目标列选择")
        target_layout = QVBoxLayout(target_group)
        
        self.target_combo = QComboBox()
        self.target_combo.setMinimumHeight(30)
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        
        target_layout.addWidget(QLabel("选择要预测的列:"))
        target_layout.addWidget(self.target_combo)
        layout.addWidget(target_group)
        
        # 4. 需求描述
        requirement_group = QGroupBox("📝 预测需求描述")
        requirement_layout = QVBoxLayout(requirement_group)
        
        self.requirement_edit = QTextEdit()
        self.requirement_edit.setPlaceholderText(
            "输入你的预测需求，例如：\n"
            "• 预测下个月的销售额\n"
            "• 分类判断用户是否会流失\n"
            "• 时序预测未来7天的流量趋势"
        )
        self.requirement_edit.setMinimumHeight(100)
        self.requirement_edit.setMaximumHeight(120)
        self.requirement_edit.textChanged.connect(self._on_requirement_changed)
        
        requirement_layout.addWidget(self.requirement_edit)
        layout.addWidget(requirement_group)
        
        # 5. 开始训练按钮
        self.train_btn = QPushButton("🚀 开始训练模型")
        self.train_btn.setMinimumHeight(45)
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self._on_train)
        layout.addWidget(self.train_btn)
        
        layout.addStretch()
        return frame
    
    def _create_right_panel(self) -> QFrame:
        """创建右侧面板"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 1. 任务配置
        config_group = QGroupBox("⚙️ 任务配置")
        config_layout = QVBoxLayout(config_group)
        
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setMaximumHeight(100)
        self.config_text.setPlaceholderText("选择数据和描述需求后，这里显示自动识别的任务配置")
        
        config_layout.addWidget(self.config_text)
        layout.addWidget(config_group)
        
        # 2. 训练进度
        progress_group = QGroupBox("📈 训练进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.train_progress = QProgressBar()
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumHeight(100)
        self.train_log.setStyleSheet("background: #1e1e1e; color: #0f0; font-family: Consolas;")
        
        progress_layout.addWidget(self.train_progress)
        progress_layout.addWidget(self.train_log)
        layout.addWidget(progress_group)
        
        # 3. 预测结果
        result_group = QGroupBox("📋 预测结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_table = QTableWidget()
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setAlternatingRowColors(True)
        
        result_layout.addWidget(self.result_table)
        layout.addWidget(result_group)
        
        # 4. 特征重要性
        importance_group = QGroupBox("🔍 特征重要性 Top10")
        importance_layout = QVBoxLayout(importance_group)
        
        self.importance_text = QTextEdit()
        self.importance_text.setReadOnly(True)
        self.importance_text.setMaximumHeight(100)
        
        importance_layout.addWidget(self.importance_text)
        layout.addWidget(importance_group)
        
        # 5. 操作按钮
        btn_layout = QHBoxLayout()
        
        self.predict_btn = QPushButton("🔮 新数据预测")
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self._on_predict)
        
        self.save_btn = QPushButton("💾 保存模型")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save_model)
        
        btn_layout.addWidget(self.predict_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        
        return frame
    
    def _init_styles(self):
        """初始化样式"""
        self.setStyleSheet("""
            QMainWindow {
                background: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #106ebe;
            }
            QPushButton:disabled {
                background: #ccc;
            }
            QPushButton#secondary {
                background: #6c6c6c;
            }
            QTableWidget {
                border: 1px solid #ddd;
                gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background: #0078d4;
                color: white;
                padding: 5px;
                border: none;
            }
        """)
    
    def _on_import_file(self):
        """导入文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择CSV数据文件",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self._log(f"正在加载: {file_path}")
        
        success, msg = self.data_loader.load_csv(file_path)
        
        if success:
            self.file_label.setText(Path(file_path).name)
            self._update_data_preview()
            self._update_target_combo()
            self.status_label.setText(f"✅ {msg}")
            self._log(msg)
        else:
            QMessageBox.warning(self, "加载失败", msg)
            self._log(f"❌ {msg}")
    
    def _update_data_preview(self):
        """更新数据预览"""
        preview = self.data_loader.get_preview(50)
        summary = self.data_loader.get_summary()
        
        # 设置表格
        self.preview_table.setRowCount(len(preview))
        self.preview_table.setColumnCount(len(preview.columns))
        self.preview_table.setHorizontalHeaderLabels(list(preview.columns))
        
        for i, row in preview.iterrows():
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                self.preview_table.setItem(i, j, item)
        
        # 调整列宽
        self.preview_table.resizeColumnsToContents()
        
        # 更新信息
        info = f"形状: {summary['shape'][0]}行 × {summary['shape'][1]}列 | "
        info += f"数值列: {len(summary['numeric_cols'])} | "
        info += f"类别列: {len(summary['categorical_cols'])}"
        self.data_info_label.setText(info)
        
        self.current_data = summary
        self._update_train_button_state()
    
    def _update_target_combo(self):
        """更新目标列下拉框"""
        self.target_combo.clear()
        
        if self.data_loader.df is not None:
            self.target_combo.addItems(list(self.data_loader.df.columns))
    
    def _on_target_changed(self):
        """目标列变化"""
        target = self.target_combo.currentText()
        if not target:
            return
        
        success, msg = self.data_loader.select_target(target)
        if success:
            self._log(msg)
            self._update_task_config()
            self._update_train_button_state()

    def _on_requirement_changed(self):
        """需求描述变化"""
        self._update_task_config()
        self._update_train_button_state()

    def _update_train_button_state(self):
        """更新训练按钮状态"""
        has_data = self.data_loader.df is not None and self.target_combo.currentText()
        has_requirement = self.requirement_edit.toPlainText().strip() != ""
        self.train_btn.setEnabled(bool(has_data and has_requirement))
    
    def _update_task_config(self):
        """更新任务配置显示"""
        requirement = self.requirement_edit.toPlainText().strip()
        
        if not requirement or not self.current_data:
            self.config_text.setPlainText("请先选择目标列并描述预测需求")
            return
        
        task_config = self.task_router.parse_requirement(requirement, self.current_data)
        self.config_text.setPlainText(self.task_router.explain_task())
    
    def _on_train(self):
        """开始训练"""
        requirement = self.requirement_edit.toPlainText().strip()
        if not requirement:
            QMessageBox.warning(self, "提示", "请输入预测需求描述")
            return
        
        target_col = self.target_combo.currentText()
        if not target_col:
            QMessageBox.warning(self, "提示", "请选择目标列")
            return
        
        # 获取特征和目标
        feature_df = self.data_loader.get_feature_matrix(exclude_cols=[target_col])
        target_series = self.data_loader.df[target_col]
        
        if feature_df.empty:
            QMessageBox.warning(self, "错误", "特征为空，请确保有除目标列外的其他列作为特征")
            return
        
        # 解析任务
        task_config = self.task_router.parse_requirement(requirement, self.current_data)
        self.config_text.setPlainText(self.task_router.explain_task())
        
        # 禁用按钮
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self._log(f"🚀 开始训练: {task_config.model_name}")
        
        # 训练
        success, msg = self.predictor.train(
            X=feature_df,
            y=target_series,
            task_type=task_config.task_type,
            model_name=task_config.model_name,
            model_params=task_config.model_params
        )
        
        self.progress_bar.setValue(100)
        
        if success:
            self._log("✅ " + msg.replace("\n", "\n✅ "))
            
            # 更新特征重要性
            importance = self.predictor.get_feature_importance()
            if importance:
                sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                imp_text = "\n".join([f"{name}: {val:.4f}" for name, val in sorted_imp])
                self.importance_text.setPlainText(imp_text)
            
            self.predict_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.status_label.setText("✅ 训练完成！可以点击「新数据预测」或「保存模型」")
        else:
            self._log(f"❌ {msg}")
            QMessageBox.critical(self, "训练失败", msg)
        
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def _on_predict(self):
        """预测新数据"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择预测数据文件",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # 加载新数据
        success, msg = self.data_loader.load_csv(file_path)
        if not success:
            QMessageBox.warning(self, "加载失败", msg)
            return
        
        target_col = self.target_combo.currentText()
        feature_df = self.data_loader.get_feature_matrix(exclude_cols=[target_col])
        
        # 预测
        try:
            predictions = self.predictor.predict(feature_df)
            
            # 显示结果
            result_df = self.data_loader.df.copy()
            result_df['预测值'] = predictions
            
            self.result_table.setRowCount(len(result_df))
            self.result_table.setColumnCount(len(result_df.columns))
            self.result_table.setHorizontalHeaderLabels(list(result_df.columns))
            
            for i, row in result_df.iterrows():
                for j, val in enumerate(row):
                    self.result_table.setItem(i, j, QTableWidgetItem(str(val)))
            
            self.result_table.resizeColumnsToContents()
            self._log(f"✅ 预测完成，共 {len(predictions)} 条结果")
            self.status_label.setText(f"✅ 预测完成！")
            
        except Exception as e:
            QMessageBox.critical(self, "预测失败", str(e))
            self._log(f"❌ 预测失败: {e}")
    
    def _on_save_model(self):
        """保存模型"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存模型",
            "deeppredict_model.pkl",
            "Model Files (*.pkl)"
        )
        
        if not file_path:
            return
        
        try:
            self.predictor.save_model(file_path)
            self._log(f"✅ 模型已保存: {file_path}")
            QMessageBox.information(self, "保存成功", f"模型已保存到:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
    
    def _log(self, message: str):
        """添加日志"""
        self.train_log.append(message)
        
        # 滚动到底部
        cursor = self.train_log.textCursor()
        cursor.movePosition(cursor.End)
        self.train_log.setTextCursor(cursor)
