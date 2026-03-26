# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import subprocess

# Get command line for process 13428
result = subprocess.run(
    ['powershell', '-Command', 
     'Get-CimInstance Win32_Process -Filter "ProcessId=13428" | Select-Object -ExpandProperty CommandLine'],
    capture_output=True, text=True, encoding='utf-8', errors='replace'
)
print("Process 13428 command line:")
print(result.stdout)
print(result.stderr)
