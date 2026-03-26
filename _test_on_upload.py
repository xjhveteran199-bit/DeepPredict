"""
Verify on_file_upload returns exactly 8 correct values with actual data
"""
import sys, os
sys.path.insert(0, r'C:\Users\XJH\DeepPredict')
os.chdir(r'C:\Users\XJH\DeepPredict')

src = open(r'C:\Users\XJH\DeepPredict\deeppredict_web.py', encoding='utf-8').read()

# Comment out the demo.launch call (it's at the end of the file)
lines = src.split('\n')
for i, line in enumerate(lines):
    if line.strip().startswith('demo.launch('):
        depth = 0
        for j in range(i, min(i+20, len(lines))):
            depth += lines[j].count('(') - lines[j].count(')')
            lines[j] = '# ' + lines[j]
            if depth <= 0:
                break
        break
src = '\n'.join(lines)

# Execute to get the function
code = compile(src, 'dp.py', 'exec')
namespace = {'__file__': __file__}
exec(code, namespace)

on_file_upload = namespace['on_file_upload']

# Create mock FileData
import gradio
from pathlib import Path

fd = gradio.data_classes.FileData(
    path=str(Path(r'C:\Users\XJH\DeepResearch\test_data\demo\iris_header.csv')),
    orig_name='iris_header.csv',
    is_file=True
)

print("Testing on_file_upload with iris CSV...")
result = on_file_upload(fd)
print(f"Returned {len(result)} values:")
for i, r in enumerate(result):
    if hasattr(r, 'choices'):
        c = getattr(r, 'choices', [])
        print(f"  [{i}] gr.update: choices={c[:5] if c else 'EMPTY'}...")
    elif isinstance(r, str):
        print(f"  [{i}] str ({len(r)} chars): {r[:60]}...")
    else:
        print(f"  [{i}] {type(r).__name__}")

print("\nAll 8 outputs:", "PASS" if len(result) == 8 else f"FAIL ({len(result)} values)")
