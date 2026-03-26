"""
End-to-end test: upload iris -> on_file_upload -> on_train -> check ZIP
"""
import os
os.chdir(r"C:\Users\XJH\DeepPredict")

import requests, json, zipfile, shutil
from pathlib import Path

API = "http://localhost:7861"

# ========== Step 1: Upload file ==========
print("=== Step 1: Uploading file ===")
with open(r"C:\Users\XJH\DeepResearch\test_data\demo\iris_header.csv", "rb") as f:
    r = requests.post(f"{API}/gradio_api/upload", files={"files": f})
print("Upload:", r.status_code)
fp = json.loads(r.text)[0]
print("File path:", fp)

# ========== Step 2: Call on_file_upload ==========
print("\n=== Step 2: Calling on_file_upload ===")
filedata = {
    "path": fp,
    "url": None,
    "size": None,
    "orig_name": "iris_header.csv",
    "mime_type": "text/csv",
    "is_stream": False,
    "meta": {"_type": "gradio.FileData"}
}

r2 = requests.post(f"{API}/gradio_api/call/on_file_upload",
                   json={"data": [filedata]}, timeout=30)
print("on_file_upload call:", r2.status_code)
event_id = json.loads(r2.text)["event_id"]
print("Event ID:", event_id)

# Get on_file_upload result
import threading, queue
q = queue.Queue()
def get_result():
    try:
        with requests.get(f"{API}/gradio_api/call/on_file_upload/{event_id}",
                         stream=True, timeout=15) as r3:
            for line in r3.iter_lines():
                if line:
                    q.put(line.decode())
    except Exception as e:
        q.put(f"Error: {e}")

t = threading.Thread(target=get_result)
t.start()
t.join(timeout=12)

# Parse the data_preview output (index 0)
data_preview = None
for _ in range(100):
    try:
        line = q.get_nowait()
        if line.startswith("data:"):
            result = json.loads(line[5:])
            print(f"on_file_upload returned {len(result)} values")
            # data_preview is index 0
            if result and len(result) > 0:
                headers = result[0].get("headers", [])
                print(f"Headers: {headers}")
            # feature_col dropdown is index 4 (gr.update choices=...)
            if result and len(result) > 4:
                feat_choices = result[4] if isinstance(result[4], list) else []
                print(f"Feature choices: {feat_choices}")
            # target_col dropdown is index 5
            if result and len(result) > 5:
                targ = result[5] if isinstance(result[5], list) else []
                print(f"Target choices: {targ}")
            break
    except queue.Empty:
        break

# ========== Step 3: Call on_train ==========
print("\n=== Step 3: Calling on_train ===")
print("Using GradientBoosting regression on sepal_length")

# Clean outputs dir
shutil.rmtree("outputs", ignore_errors=True)

train_inputs = [
    None,           # feature_col
    "sepal_length", # target_col
    "regression",   # predict_mode
    "GradientBoosting",  # model_select
    "",             # requirement
    "10",           # n_future_val
    "分钟",          # n_future_unit
    "",             # chart_requirement
]

r4 = requests.post(f"{API}/gradio_api/call/on_train",
                   json={"data": train_inputs}, timeout=120)
print("on_train call:", r4.status_code)
event_id2 = json.loads(r4.text)["event_id"]
print("Event ID:", event_id2)

# Get on_train result (with progress polling)
def get_train_result():
    try:
        with requests.get(f"{API}/gradio_api/call/on_train/{event_id2}",
                         stream=True, timeout=120) as r5:
            for line in r5.iter_lines():
                if line:
                    decoded = line.decode()
                    if decoded.startswith("data:"):
                        q.put(("train_result", json.loads(decoded[5:])))
                    elif "progress" in decoded.lower() or "event" in decoded.lower():
                        q.put(("progress", decoded[:100]))
                    else:
                        q.put(("other", decoded[:100]))
                    print(f"SSE: {decoded[:120]}")
    except Exception as e:
        q.put(f"Error: {e}")

t2 = threading.Thread(target=get_train_result)
t2.start()
t2.join(timeout=90)

# Check outputs dir
print("\n=== Step 4: Checking outputs ===")
outputs = Path("outputs")
if outputs.exists():
    zips = list(outputs.glob("*.zip"))
    print(f"ZIP files: {len(zips)}")
    if zips:
        latest_zip = sorted(zips, key=lambda p: p.stat().st_mtime)[-1]
        print(f"Latest ZIP: {latest_zip.name}")
        
        # Extract and list contents
        extract_dir = Path("outputs/extracted")
        shutil.rmtree(extract_dir, ignore_errors=True)
        with zipfile.ZipFile(latest_zip, "r") as zf:
            zf.extractall(extract_dir)
            print("\nZIP contents:")
            for name in sorted(zf.namelist()):
                print(f"  {name}")
else:
    print("outputs/ does not exist!")

print("\nDone!")
