# -*- coding: utf-8 -*-
import subprocess
import sys

prompt = """你是 ChronoML 的核心开发者。请彻底修复 Error 13 和 ZIP 创建问题。

根因：
1. Gradio 6.x 的 postprocess_data 对 gr.File 组件返回值调用 hash_file()。当返回空字符串 "" 时，Gradio 把 "" 转成当前工作目录 C:\Users\XJH\ChronoML，然后 open(目录, 'rb') 导致 PermissionError。
2. 整个 ZIP 创建在一个大 try-except 中，任何一步出错 ZIP 就没了。
3. plot_training_history 只查找 train_loss/val_loss，但 sklearn 模型存的是 loss/accuracy。

修复任务：
1. on_train 最终 return 前：forecast_plot/zip_path/training_hist_plot 的空值必须返回 None 而非 ""
2. 重构 ZIP 创建为三层结构：核心文件独立 try → 可选文件独立 try → ZIP 必须创建
3. 更新 plot_training_history 支持 sklearn 格式

请直接修改 C:\Users\XJH\ChronoML\deeppredict_web.py，完成后：
- 启动服务器（端口 7861）
- 验证 outputs 有 ZIP 文件
- git commit 并 push origin master"""

result = subprocess.run(
    [r"C:\Users\XJH\AppData\Roaming\npm\opencode.cmd", "run", prompt],
    cwd=r"C:\Users\XJH\ChronoML",
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=600
)
print("STDOUT:", result.stdout[:8000] if result.stdout else "")
print("STDERR:", result.stderr[:2000] if result.stderr else "")
print("RETURN CODE:", result.returncode)
