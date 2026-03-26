# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Find deep_research_web.py
search_paths = [
    r'C:\Users\XJH\DeepPredict',
    r'C:\Users\XJH',
]

for base in search_paths:
    try:
        for root, dirs, files in os.walk(base):
            for f in files:
                if 'deep_research' in f.lower() and f.endswith('.py'):
                    full = os.path.join(root, f)
                    print(f"Found: {full}")
    except Exception as e:
        print(f"Error searching {base}: {e}")
